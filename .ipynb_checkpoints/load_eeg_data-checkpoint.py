import mne
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
# These are the channel names that need to be renamed because your dataset uses
# the OLD 10-20 naming system (T3, T4, T5, T6) but EEGPT expects the NEW naming
# system (T7, T8, P7, P8). They refer to the same physical electrode locations.
CHANNEL_RENAME = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8',
}

# This is the exact list of channels we will keep, in the exact order EEGPT
# expects. A1 and A2 are reference electrodes (placed on the ears) and carry
# no brain signal, so we drop them.
USE_CHANNELS = [
    'FP1', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7',   # was T3
    'C3', 'CZ', 'C4',
    'T8',   # was T4
    'P7',   # was T5
    'P3', 'PZ', 'P4',
    'P8',   # was T6
    'O1', 'O2'
]

# Numeric labels for each group. PyTorch needs numbers not strings.
# F is included now so it's easy to add later without changing the script.
LABEL_MAP = {
    'A': 0,   # Alzheimer's
    'C': 1,   # Control
    'F': 2,   # Frontotemporal dementia
}

SFREQ     = 256   # EEGPT was pretrained on 256 Hz data. Your data is 500 Hz
                  # so we resample down to match what the model expects.
EPOCH_LEN = 4.0   # Length of each window in seconds. EEGPT was pretrained on
                  # 4 second windows so we match that exactly.
                  # 4s × 256Hz = 1024 timepoints per epoch.
OVERLAP   = 0.5   # Each window overlaps the previous by 50%. This gives us
                  # more epochs from each recording which helps with small
                  # datasets. e.g. a 600s recording gives ~150 epochs at 50%
                  # overlap vs ~75 with no overlap.

# ── Step 1: Load and preprocess a single .set file ───────────────────────────
# This function takes one subject's .set file and returns a clean MNE Raw
# object ready for epoching. It handles all the cleaning steps in order.

def load_and_preprocess(set_path):
    """Load a .set file and return a cleaned MNE Raw object."""
    
    # MNE is a Python library for EEG/MEG processing. read_raw_eeglab loads
    # EEGLAB .set files specifically. preload=True loads all data into RAM
    # immediately rather than reading from disk on demand.
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    
    # ── Rename old channel names to EEGPT equivalents ──
    # We only rename channels that actually exist in the file to avoid errors.
    # e.g. if the file has T3, it becomes T7. If it already has T7, no change.
    rename_map = {
        ch: CHANNEL_RENAME[ch]
        for ch in raw.ch_names
        if ch in CHANNEL_RENAME
    }
    if rename_map:
        raw.rename_channels(rename_map)
    
    # ── Uppercase all channel names ──
    # EEGPT's CHANNEL_DICT uses all uppercase (FP1, FZ, CZ etc).
    # Your files use mixed case (Fp1, Fz, Cz etc) so we standardize here.
    raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})
    
    # ── Drop unwanted channels ──
    # Remove anything not in our USE_CHANNELS list.
    # This drops A1, A2 (reference electrodes) and any other extra channels.
    drop = [ch for ch in raw.ch_names if ch not in USE_CHANNELS]
    if drop:
        raw.drop_channels(drop)
    
    # ── Reorder channels ──
    # Ensure channels are always in the same order across all subjects.
    # This is critical — if subject 1 has [FP1, FP2, F3...] and subject 2
    # has [F3, FP1, FP2...] the model will be confused because it maps
    # channel position to brain region.
    raw.reorder_channels(USE_CHANNELS)
    
    # ── Bandpass filter: 0.5–45 Hz ──
    # EEG brain signals we care about live between 0.5 and 45 Hz.
    # Below 0.5 Hz: slow drifts from sweat, movement, electrode settling.
    # Above 45 Hz: muscle artifacts, high frequency noise.
    # Filtering removes these so the model sees clean brain signal.
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
    
    # ── Notch filter: 50 Hz ──
    # Power lines in Greece (and Europe) run at 50 Hz and inductively
    # couple into EEG electrodes creating a strong 50 Hz artifact.
    # The notch filter specifically removes just that frequency.
    # (US uses 60 Hz — always match to where the data was recorded.)
    raw.notch_filter(freqs=50.0, verbose=False)
    
    # ── Resample from 500 Hz → 256 Hz ──
    # Your device recorded at 500 Hz (500 samples per second per channel).
    # EEGPT was pretrained at 256 Hz, so we downsample to match.
    # MNE handles the anti-aliasing filter automatically before resampling.
    raw.resample(SFREQ, verbose=False)
    
    return raw


# ── Step 2: Epoch and normalize ───────────────────────────────────────────────
# After preprocessing we have one long continuous recording (~600 seconds).
# EEGPT expects fixed-length 4-second windows, so we slice it up here.

def epoch_and_normalize(raw):
    """Slice continuous EEG into overlapping 4s epochs and normalize."""
    
    # Get the raw data as a numpy array of shape (n_channels, n_timepoints)
    # i.e. (19, ~153600) for a ~600s recording at 256Hz
    data = raw.get_data()
    
    # Calculate window sizes in samples
    epoch_samples = int(EPOCH_LEN * SFREQ)           # 4 × 256 = 1024 samples
    step_samples  = int(epoch_samples * (1 - OVERLAP)) # 1024 × 0.5 = 512 samples
    
    # Slide a window across the recording, stepping by step_samples each time
    # e.g. window 1: samples 0–1023
    #      window 2: samples 512–1535
    #      window 3: samples 1024–2047  etc.
    epochs = []
    start  = 0
    while start + epoch_samples <= data.shape[1]:
        epoch = data[:, start:start + epoch_samples]  # (19, 1024)
        epochs.append(epoch)
        start += step_samples
    
    epochs = np.array(epochs)  # stack into (n_epochs, 19, 1024)
    
    # ── Normalize per channel per epoch ──
    # Each channel has different baseline voltage levels depending on electrode
    # placement, impedance, etc. Normalization removes these differences so
    # the model focuses on patterns rather than absolute voltage levels.
    # We normalize each channel within each epoch independently.
    mean   = epochs.mean(axis=-1, keepdims=True)  # mean across time: (n_epochs, 19, 1)
    std    = epochs.std( axis=-1, keepdims=True)  # std  across time: (n_epochs, 19, 1)
    std    = np.where(std < 1e-6, 1e-6, std)      # prevent division by zero for flat channels
    epochs = (epochs - mean) / std                # z-score normalize
    
    return epochs  # (n_epochs, 19, 1024)


# ── Step 3: Subject-level train/val split ─────────────────────────────────────
# WHY this matters: each subject contributes ~150 epochs. If we split randomly,
# the same subject's epochs appear in both train and val. The model could then
# "memorize" that subject's brain patterns rather than learning generalizable
# features — making validation accuracy misleadingly high. By splitting at the
# subject level, every subject is either entirely in train or entirely in val.

def subject_level_split(df, val_fraction=0.2, seed=42):
    """
    Split subjects into train and val groups, stratified by diagnosis.
    Stratified means we maintain the same A/C ratio in both train and val.
    
    Returns two DataFrames: train_df, val_df
    """
    np.random.seed(seed)
    
    train_rows, val_rows = [], []
    
    # Split each group (A, C) separately to maintain class balance
    for group in df['Group'].unique():
        group_df = df[df['Group'] == group].copy()
        
        # Shuffle subjects within this group
        group_df = group_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Calculate how many subjects go to val
        n_val   = max(1, int(len(group_df) * val_fraction))
        n_train = len(group_df) - n_val
        
        train_rows.append(group_df.iloc[:n_train])
        val_rows.append(  group_df.iloc[n_train:])
        
        print(f"  Group {group}: {n_train} train subjects, "
              f"{n_val} val subjects")
    
    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df   = pd.concat(val_rows  ).reset_index(drop=True)
    
    return train_df, val_df


# ── Step 4: Process a list of subjects into tensors ───────────────────────────
# This function takes a DataFrame of subjects, loads and processes each one,
# and concatenates everything into a single tensor ready for the model.

def process_subjects(df, dataset_dir):
    """
    Process all subjects in a DataFrame.
    Returns X tensor (n_epochs, 19, 1024), y tensor (n_epochs,), metadata list.
    """
    dataset_dir = Path(dataset_dir)
    all_x, all_y, all_meta = [], [], []
    
    for _, row in df.iterrows():
        subject_id = row['participant_id']  # e.g. "sub-001"
        group      = row['Group']           # "A", "C", or "F"
        label      = LABEL_MAP[group]       # 0, 1, or 2
        
        # Build the path to this subject's .set file
        # Structure: datasets/sub-001/eeg/sub-001_task-eyesclosed_eeg.set
        set_path = (dataset_dir / subject_id / "eeg" /
                    f"{subject_id}_task-eyesclosed_eeg.set")
        
        if not set_path.exists():
            print(f"  WARNING: {set_path} not found, skipping")
            continue
        
        print(f"  Processing {subject_id} (Group={group})...")
        
        try:
            raw    = load_and_preprocess(set_path)
            epochs = epoch_and_normalize(raw)       # (n_epochs, 19, 1024)
            
            # Convert numpy arrays to PyTorch tensors
            x = torch.tensor(epochs, dtype=torch.float32)
            y = torch.full((len(epochs),), label, dtype=torch.long)
            
            all_x.append(x)
            all_y.append(y)
            
            # Store metadata for each epoch so we can trace back
            # which subject/group each epoch came from during analysis
            all_meta.extend([{
                'subject_id': subject_id,
                'group':      group,
                'age':        row['Age'],
                'gender':     row['Gender'],
                'mmse':       row['MMSE'],
                'epoch_idx':  i,
            } for i in range(len(epochs))])
            
            print(f"    → {len(epochs)} epochs extracted")
            
        except Exception as e:
            print(f"  ERROR on {subject_id}: {e}")
            continue
    
    X = torch.cat(all_x, dim=0)  # (total_epochs, 19, 1024)
    y = torch.cat(all_y, dim=0)  # (total_epochs,)
    
    return X, y, all_meta


# ── Step 5: PyTorch Dataset and DataLoader ────────────────────────────────────
# PyTorch's Dataset and DataLoader are the standard way to feed data to a model
# during training. Dataset wraps your tensors and DataLoader handles batching,
# shuffling, and parallel loading automatically.

class EEGDataset(Dataset):
    def __init__(self, X, y):
        # Store the full tensors
        self.X = X  # (n_epochs, 19, 1024)
        self.y = y  # (n_epochs,)
    
    def __len__(self):
        # Required by PyTorch — tells DataLoader how many samples exist
        return len(self.X)
    
    def __getitem__(self, idx):
        # Required by PyTorch — returns one sample given an index
        # DataLoader calls this repeatedly to build each batch
        return self.X[idx], self.y[idx]


# ── Main: putting it all together ─────────────────────────────────────────────

def build_dataloaders(
    dataset_dir,
    participants_tsv,
    groups       = ('A', 'C'),  # which groups to include
    val_fraction = 0.2,         # 20% of subjects held out for validation
    batch_size   = 32,
    seed         = 42
):
    """
    Full pipeline from raw files to DataLoaders ready for model training.
    """
    
    # Load the participants TSV and filter to requested groups
    df = pd.read_csv(participants_tsv, sep='\t')
    df = df[df['Group'].isin(groups)].reset_index(drop=True)
    
    print(f"Total subjects: {len(df)}")
    print(f"Group breakdown: {df['Group'].value_counts().to_dict()}\n")
    
    # Split subjects into train and val BEFORE processing
    # This is the key step that prevents data leakage
    print("Splitting subjects into train/val...")
    train_df, val_df = subject_level_split(df, val_fraction, seed)
    
    # Process train subjects
    print(f"\nProcessing {len(train_df)} train subjects...")
    X_train, y_train, meta_train = process_subjects(train_df, dataset_dir)
    
    # Process val subjects
    print(f"\nProcessing {len(val_df)} val subjects...")
    X_val, y_val, meta_val = process_subjects(val_df, dataset_dir)
    
    # Summary
    print(f"\n── Dataset Summary ──────────────────────────")
    print(f"Train: {len(X_train)} epochs | "
          f"AD={( y_train==0).sum()} | Control={(y_train==1).sum()}")
    print(f"Val:   {len(X_val)}   epochs | "
          f"AD={( y_val  ==0).sum()} | Control={(y_val  ==1).sum()}")
    
    # Save processed data so you don't have to rerun preprocessing each time
    # Loading from .pt is much faster than reprocessing all the .set files
    torch.save({
        'X_train': X_train, 'y_train': y_train, 'meta_train': meta_train,
        'X_val':   X_val,   'y_val':   y_val,   'meta_val':   meta_val,
    }, "processed_eeg.pt")
    print(f"\nSaved to processed_eeg.pt")
    
    # Wrap in Dataset and DataLoader
    # shuffle=True for train so the model sees subjects in random order
    # shuffle=False for val so results are reproducible
    train_loader = DataLoader(
        EEGDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        EEGDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    return train_loader, val_loader


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    train_loader, val_loader = build_dataloaders(
        dataset_dir      = "datasets/ds004504_eeg",
        participants_tsv = "datasets/ds004504_eeg/participants.tsv",
        groups           = ('A', 'C'),
        val_fraction     = 0.2,
        batch_size       = 32,
    )
    
    # Sanity check — inspect one batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch X shape : {x_batch.shape}")   # (32, 19, 1024)
    print(f"Batch y shape : {y_batch.shape}")     # (32,)
    print(f"Unique labels : {y_batch.unique()}")  # tensor([0, 1])
    print(f"Value range   : {x_batch.min():.2f} to {x_batch.max():.2f}")