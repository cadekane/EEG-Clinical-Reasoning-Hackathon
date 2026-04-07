"""
preprocess.py
─────────────
Converts raw .set EEG files into tensors ready for EEGPT.
Run this ONCE to generate data/processed_eeg.pt, then use train.py.

Usage:
    python preprocess.py
"""

import mne
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

# ── Configuration ─────────────────────────────────────────────────────────────

# Old 10-20 names in your dataset → new names expected by EEGPT
CHANNEL_RENAME = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8',
}

# The 19 channels we keep, in the exact order EEGPT expects
USE_CHANNELS = [
    'FP1', 'FP2',
    'F7',  'F3', 'FZ', 'F4', 'F8',
    'T7',                             # renamed from T3
    'C3',  'CZ', 'C4',
    'T8',                             # renamed from T4
    'P7',                             # renamed from T5
    'P3',  'PZ', 'P4',
    'P8',                             # renamed from T6
    'O1',  'O2',
]

# Numeric labels for PyTorch (must be integers)
LABEL_MAP = {
    'A': 0,   # Alzheimer's
    'C': 1,   # Control
    'F': 2,   # Frontotemporal dementia (ready for later)
}

SFREQ     = 256   # Hz — resample to match EEGPT pretraining rate
EPOCH_LEN = 4.0   # seconds — matches EEGPT pretraining window length
OVERLAP   = 0.5   # 50% overlap between windows to maximize epochs


# ── Step 1: Load and clean a single .set file ─────────────────────────────────

def load_and_preprocess(set_path):
    """
    Load a raw .set file and return a cleaned MNE Raw object.
    Handles renaming, filtering, and resampling.
    """
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    # Rename old 10-20 channel names to EEGPT equivalents
    rename_map = {ch: CHANNEL_RENAME[ch]
                  for ch in raw.ch_names if ch in CHANNEL_RENAME}
    if rename_map:
        raw.rename_channels(rename_map)

    # Uppercase all names (EEGPT expects FP1 not Fp1, FZ not Fz, etc.)
    raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})

    # Drop A1, A2 (reference electrodes) and any other unwanted channels
    drop = [ch for ch in raw.ch_names if ch not in USE_CHANNELS]
    if drop:
        raw.drop_channels(drop)

    # Enforce consistent channel ordering across all subjects
    raw.reorder_channels(USE_CHANNELS)

    # Bandpass filter 0.5–45 Hz: removes slow drift and high-freq noise
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)

    # Notch filter at 50 Hz: removes European power line interference
    raw.notch_filter(freqs=50.0, verbose=False)

    # Resample from 500 Hz down to 256 Hz to match EEGPT pretraining
    if raw.info['sfreq'] != SFREQ:
        raw.resample(SFREQ, verbose=False)

    return raw


# ── Step 2: Slice into epochs and normalize ───────────────────────────────────

def epoch_and_normalize(raw):
    """
    Slice continuous EEG into overlapping 4-second windows.
    Returns normalized array of shape (n_epochs, 19, 1024).
    """
    data          = raw.get_data()                       # (19, n_timepoints)
    epoch_samples = int(EPOCH_LEN * SFREQ)               # 4s × 256Hz = 1024
    step_samples  = int(epoch_samples * (1 - OVERLAP))   # 50% overlap = 512

    epochs = []
    start  = 0
    while start + epoch_samples <= data.shape[1]:
        epochs.append(data[:, start:start + epoch_samples])
        start += step_samples

    epochs = np.array(epochs)  # (n_epochs, 19, 1024)

    # Z-score normalize each channel within each epoch independently.
    # Removes differences in baseline voltage across electrodes/subjects.
    mean   = epochs.mean(axis=-1, keepdims=True)
    std    = epochs.std( axis=-1, keepdims=True)
    std    = np.where(std < 1e-6, 1e-6, std)    # avoid division by zero
    epochs = (epochs - mean) / std

    return epochs


# ── Step 3: Subject-level train/val split ─────────────────────────────────────

def subject_level_split(df, val_fraction=0.2, seed=42):
    """
    Split subjects into train/val sets, stratified by diagnosis group.
    Keeps entire subjects in one split to prevent data leakage.

    Returns train_df, val_df
    """
    np.random.seed(seed)
    train_rows, val_rows = [], []

    for group in df['Group'].unique():
        group_df = df[df['Group'] == group].copy()
        group_df = group_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_val   = max(1, int(len(group_df) * val_fraction))
        n_train = len(group_df) - n_val

        train_rows.append(group_df.iloc[:n_train])
        val_rows.append(  group_df.iloc[n_train:])

        print(f"  Group {group}: {n_train} train subjects, {n_val} val subjects")

    return (pd.concat(train_rows).reset_index(drop=True),
            pd.concat(val_rows  ).reset_index(drop=True))


# ── Step 4: Process a list of subjects into tensors ───────────────────────────

def process_subjects(df, dataset_dir):
    """
    Loop through subjects in a DataFrame, preprocess each one,
    and concatenate into a single tensor.

    Returns:
        X    : (total_epochs, 19, 1024) float32 tensor
        y    : (total_epochs,) long tensor
        meta : list of dicts with subject info per epoch
    """
    dataset_dir    = Path(dataset_dir)
    all_x, all_y   = [], []
    all_meta       = []

    for _, row in df.iterrows():
        subject_id = row['participant_id']
        group      = row['Group']
        label      = LABEL_MAP[group]

        set_path = (dataset_dir / subject_id / "eeg" /
                    f"{subject_id}_task-eyesclosed_eeg.set")

        if not set_path.exists():
            print(f"  WARNING: {set_path} not found — skipping")
            continue

        print(f"  Processing {subject_id} (Group={group})...")

        try:
            raw    = load_and_preprocess(set_path)
            epochs = epoch_and_normalize(raw)           # (n_epochs, 19, 1024)

            x = torch.tensor(epochs, dtype=torch.float32)
            y = torch.full((len(epochs),), label, dtype=torch.long)

            all_x.append(x)
            all_y.append(y)
            all_meta.extend([{
                'subject_id': subject_id,
                'group':      group,
                'age':        row['Age'],
                'gender':     row['Gender'],
                'mmse':       row['MMSE'],
                'epoch_idx':  i,
            } for i in range(len(epochs))])

            print(f"    → {len(epochs)} epochs")

        except Exception as e:
            print(f"  ERROR on {subject_id}: {e}")
            continue

    X = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)
    return X, y, all_meta


# ── Step 5: PyTorch Dataset ───────────────────────────────────────────────────

class EEGDataset(Dataset):
    """Wraps X and y tensors for use with PyTorch DataLoader."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATASET_DIR      = "datasets/"
    PARTICIPANTS_TSV = "datasets/participants.tsv"
    OUTPUT_PATH      = "data/processed_eeg.pt"
    GROUPS           = ('A', 'C')    # change to ('A', 'C', 'F') to include FTD

    # Load participants file
    df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    df = df[df['Group'].isin(GROUPS)].reset_index(drop=True)

    print(f"Total subjects : {len(df)}")
    print(f"Group breakdown: {df['Group'].value_counts().to_dict()}\n")

    # Split at subject level BEFORE any processing
    print("Splitting subjects into train/val...")
    train_df, val_df = subject_level_split(df, val_fraction=0.2)

    # Process each split
    print(f"\nProcessing {len(train_df)} train subjects...")
    X_train, y_train, meta_train = process_subjects(train_df, DATASET_DIR)

    print(f"\nProcessing {len(val_df)} val subjects...")
    X_val, y_val, meta_val = process_subjects(val_df, DATASET_DIR)

    # Print summary
    print(f"\n── Summary ──────────────────────────────────")
    print(f"Train : {len(X_train):5d} epochs | "
          f"AD={(y_train==0).sum():4d} | Control={(y_train==1).sum():4d}")
    print(f"Val   : {len(X_val):5d} epochs | "
          f"AD={(y_val==0).sum():4d} | Control={(y_val==1).sum():4d}")
    print(f"X shape: {X_train.shape}  (n_epochs, channels, timepoints)")

    # Save to disk
    Path("data").mkdir(exist_ok=True)
    torch.save({
        'X_train':    X_train,
        'y_train':    y_train,
        'meta_train': meta_train,
        'X_val':      X_val,
        'y_val':      y_val,
        'meta_val':   meta_val,
    }, OUTPUT_PATH)

    print(f"\nSaved to {OUTPUT_PATH}")