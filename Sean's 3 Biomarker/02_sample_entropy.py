"""
Sample Entropy (SampEn) Extractor
Dataset: OpenNeuro ds004504 (Alzheimer's/FTD/CN resting-state EEG)

Input:  {BASE_DIR}/derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
Output: {BASE_DIR}/results/SampleEntropy.csv

Usage:
    python compute_sampen.py --base_dir /path/to/ds004504
    python compute_sampen.py --base_dir /path/to/ds004504 --subject sub-001

Dependencies:
    pip install mne antropy numpy pandas scipy
"""

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import mne
import antropy

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Channel groups for regional analysis
CHANNEL_GROUPS = {
    "frontal":    ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central":    ["C3", "Cz", "C4"],
    "temporal":   ["T3", "T4", "T5", "T6"],
    "parietal":   ["P3", "Pz", "P4"],
    "occipital":  ["O1", "O2"],
    "posterior":  ["O1", "O2", "P3", "Pz", "P4"],   # primary region of interest
}

# SampEn parameters (standard EEG literature values)
SAMPEN_M = 2          # Template length (embedding dimension)
SAMPEN_R = 0.2        # Tolerance as fraction of signal std: r = SAMPEN_R * std(signal)

# Epoching — SampEn is sensitive to segment length; keep consistent across subjects
EPOCH_LENGTH_SEC  = 5.0    # Length of each epoch (seconds)
EPOCH_OVERLAP     = 0.0    # No overlap — independent epochs
MIN_EPOCHS        = 5      # Discard subject if fewer than this many clean epochs

# Epoch rejection — discard epochs with extreme amplitude
AMPLITUDE_THRESH_UV = 150.0   # µV peak-to-peak threshold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_participants(base_dir: str) -> pd.DataFrame:
    tsv_path = os.path.join(base_dir, "participants.tsv")
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t")
        df.columns = df.columns.str.lower()
        return df
    return pd.DataFrame()


def find_set_files(base_dir: str, subject: str = None) -> list:
    if subject:
        pattern = os.path.join(
            base_dir, "derivatives", subject, "eeg",
            f"{subject}_task-eyesclosed_eeg.set"
        )
    else:
        pattern = os.path.join(
            base_dir, "derivatives", "sub-*", "eeg",
            "sub-*_task-eyesclosed_eeg.set"
        )
    return sorted(glob.glob(pattern))


def pick_channels_available(raw: mne.io.BaseRaw, channel_list: list) -> list:
    """Return subset of channel_list that exists in raw."""
    return [ch for ch in channel_list if ch in raw.ch_names]


def make_epochs(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Segment continuous data into fixed-length epochs and reject bad ones.

    Returns
    -------
    epochs : np.ndarray  shape (n_epochs, n_channels, n_samples)
    """
    sfreq      = raw.info["sfreq"]
    n_samples  = int(EPOCH_LENGTH_SEC * sfreq)
    data       = raw.get_data(units="uV")   # (n_channels, n_times)
    n_channels, n_times = data.shape

    n_epochs = n_times // n_samples
    epochs   = []

    for i in range(n_epochs):
        start = i * n_samples
        epoch = data[:, start : start + n_samples]

        # Amplitude rejection
        ptp = np.ptp(epoch, axis=1)   # peak-to-peak per channel
        if np.any(ptp > AMPLITUDE_THRESH_UV):
            continue

        # Flat signal rejection (std < 0.1 µV on any channel)
        if np.any(epoch.std(axis=1) < 0.1):
            continue

        epochs.append(epoch)

    return np.array(epochs) if epochs else np.empty((0, n_channels, n_samples))


def sampen_single(signal: np.ndarray) -> float:
    """
    Compute Sample Entropy for a 1D signal.
    r is set to SAMPEN_R * std(signal) per standard practice.
    Returns np.nan if computation fails.
    """
    std = signal.std()
    if std < 1e-10:
        return np.nan
    try:
        se = antropy.sample_entropy(signal, order=SAMPEN_M, metric="chebyshev")
        # antropy uses fixed r=0.2*std internally when metric='chebyshev'
        # For explicit control, use the formula below instead:
        # se = antropy.sample_entropy(signal, order=SAMPEN_M,
        #                             metric="chebyshev")
        return float(se) if np.isfinite(se) else np.nan
    except Exception:
        return np.nan


def compute_regional_sampen(epochs: np.ndarray,
                             raw: mne.io.BaseRaw) -> dict:
    """
    Compute mean SampEn per channel group across all clean epochs.

    Returns dict of {region_sampen: value, region_sampen_std: value, ...}
    """
    ch_names = raw.ch_names
    results  = {}

    for region, ch_list in CHANNEL_GROUPS.items():
        available_idx = [ch_names.index(ch) for ch in ch_list
                         if ch in ch_names]
        if not available_idx:
            results[f"{region}_sampen_mean"] = np.nan
            results[f"{region}_sampen_std"]  = np.nan
            results[f"{region}_n_channels"]  = 0
            continue

        # SampEn per epoch per channel → mean across both
        epoch_ch_vals = []
        for epoch in epochs:                     # (n_channels, n_samples)
            for idx in available_idx:
                val = sampen_single(epoch[idx])
                if not np.isnan(val):
                    epoch_ch_vals.append(val)

        if epoch_ch_vals:
            results[f"{region}_sampen_mean"] = round(float(np.mean(epoch_ch_vals)), 5)
            results[f"{region}_sampen_std"]  = round(float(np.std(epoch_ch_vals)),  5)
        else:
            results[f"{region}_sampen_mean"] = np.nan
            results[f"{region}_sampen_std"]  = np.nan

        results[f"{region}_n_channels"] = len(available_idx)

    return results


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(set_path: str) -> dict:
    subject_id = os.path.basename(set_path).split("_")[0]
    print(f"  Processing {subject_id} ...")

    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception as e:
        print(f"    [ERROR] Could not load: {e}")
        return {"subject_id": subject_id, "error": str(e)}

    # Light notch filter (data already bandpass filtered by dataset authors)
    raw.notch_filter(freqs=[50, 60], verbose=False)

    # Crop edges for stationarity
    duration = raw.times[-1]
    t_start  = min(30.0, duration * 0.1)
    t_end    = max(duration - 30.0, duration * 0.9)
    if t_end > t_start + 10.0:
        raw.crop(tmin=t_start, tmax=t_end)

    # Pick only EEG channels
    raw.pick("eeg", verbose=False)

    epochs = make_epochs(raw)
    n_epochs = len(epochs)

    if n_epochs < MIN_EPOCHS:
        msg = f"Only {n_epochs} clean epochs (min={MIN_EPOCHS}) — skipping."
        print(f"    [WARN] {msg}")
        return {"subject_id": subject_id, "n_clean_epochs": n_epochs, "error": msg}

    print(f"    {n_epochs} clean epochs × {len(raw.ch_names)} channels")

    regional = compute_regional_sampen(epochs, raw)

    return {
        "subject_id":         subject_id,
        "recording_dur_s":    round(duration, 1),
        "sfreq_hz":           raw.info["sfreq"],
        "n_clean_epochs":     n_epochs,
        "epoch_length_s":     EPOCH_LENGTH_SEC,
        "sampen_m":           SAMPEN_M,
        "sampen_r_fraction":  SAMPEN_R,
        **regional,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Sample Entropy from ds004504 EEG dataset."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--subject",  default=None,
                        help="Single subject ID, e.g. sub-001. Omit for all.")
    parser.add_argument("--output",   default=None,
                        help="Output CSV path. Default: {BASE_DIR}/results/SampleEntropy.csv")
    args = parser.parse_args()

    base_dir   = os.path.abspath(args.base_dir)
    output_csv = args.output or os.path.join(base_dir, "results", "SampleEntropy.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    participants = load_participants(base_dir)
    set_files    = find_set_files(base_dir, subject=args.subject)

    if not set_files:
        print(f"[ERROR] No .set files found under {base_dir}/derivatives/")
        return

    print(f"Found {len(set_files)} subject(s). Starting SampEn extraction...\n")

    records = []
    for set_path in set_files:
        result = process_subject(set_path)
        records.append(result)

    df = pd.DataFrame(records)

    # Merge with participants.tsv
    if not participants.empty:
        id_col = next((c for c in participants.columns
                       if "participant" in c or c == "subject_id"), None)
        if id_col:
            participants = participants.rename(columns={id_col: "subject_id"})
            if not participants["subject_id"].str.startswith("sub-").all():
                participants["subject_id"] = "sub-" + participants["subject_id"].str.zfill(3)
            df = df.merge(participants, on="subject_id", how="left")

    df = df.sort_values("subject_id").reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print(f"\nDone. Results saved to: {output_csv}")
    print(f"Subjects processed: {len(df)}")

    # Summary by group
    group_col = next((c for c in df.columns
                      if c.lower() in ("group", "dx", "diagnosis")), None)
    metric    = "posterior_sampen_mean"
    if group_col and metric in df.columns:
        print(f"\n--- Posterior SampEn by {group_col} ---")
        summary = (df.groupby(group_col)[metric]
                     .agg(["count", "mean", "std"])
                     .rename(columns={"count": "n",
                                      "mean":  "mean_sampen",
                                      "std":   "std_sampen"}))
        summary["mean_sampen"] = summary["mean_sampen"].round(4)
        summary["std_sampen"]  = summary["std_sampen"].round(4)
        print(summary.to_string())


if __name__ == "__main__":
    main()
