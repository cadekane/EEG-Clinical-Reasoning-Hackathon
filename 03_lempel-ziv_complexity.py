"""
Lempel-Ziv Complexity (LZC) Extractor
Dataset: OpenNeuro ds004504 (Alzheimer's/FTD/CN resting-state EEG)

Input:  {BASE_DIR}/derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
Output: {BASE_DIR}/results/LempelZiv.csv

Usage:
    python compute_lzc.py --base_dir /path/to/ds004504
    python compute_lzc.py --base_dir /path/to/ds004504 --subject sub-001

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
    "frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central":   ["C3", "Cz", "C4"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "parietal":  ["P3", "Pz", "P4"],
    "occipital": ["O1", "O2"],
    "posterior": ["O1", "O2", "P3", "Pz", "P4"],
}

# Binarisation method:
#   "median"  — 1 if above signal median, 0 otherwise (most common in literature)
#   "mean"    — 1 if above signal mean
#   "zero"    — 1 if above 0 (appropriate for zero-mean signals)
BINARISE_METHOD = "median"

# Epoching — use same length as SampEn script for direct comparability
EPOCH_LENGTH_SEC = 5.0
MIN_EPOCHS       = 5

# Amplitude rejection threshold (µV peak-to-peak)
AMPLITUDE_THRESH_UV = 150.0

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


def binarise(signal: np.ndarray, method: str = "median") -> np.ndarray:
    """
    Convert a continuous EEG signal to a binary sequence for LZC.

    Parameters
    ----------
    signal : 1D array
    method : "median" | "mean" | "zero"

    Returns
    -------
    binary : 1D bool array
    """
    if method == "median":
        threshold = np.median(signal)
    elif method == "mean":
        threshold = np.mean(signal)
    elif method == "zero":
        threshold = 0.0
    else:
        raise ValueError(f"Unknown binarise method: {method}")
    return (signal > threshold).astype(int)


def lzc_single(signal: np.ndarray) -> float:
    """
    Compute normalised Lempel-Ziv Complexity for a 1D signal.
    Returns np.nan if the signal is flat or computation fails.
    """
    if signal.std() < 1e-10:
        return np.nan
    try:
        binary = binarise(signal, method=BINARISE_METHOD)
        lzc = antropy.lziv_complexity(binary, normalize=True)
        return float(lzc) if np.isfinite(lzc) else np.nan
    except Exception:
        return np.nan


def make_epochs(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Segment continuous data into fixed-length epochs and reject bad ones.

    Returns
    -------
    epochs : np.ndarray  shape (n_epochs, n_channels, n_samples)
    """
    sfreq     = raw.info["sfreq"]
    n_samples = int(EPOCH_LENGTH_SEC * sfreq)
    data      = raw.get_data(units="uV")   # (n_channels, n_times)
    n_channels, n_times = data.shape

    n_epochs = n_times // n_samples
    epochs   = []

    for i in range(n_epochs):
        start = i * n_samples
        epoch = data[:, start : start + n_samples]

        # Peak-to-peak amplitude rejection
        if np.any(np.ptp(epoch, axis=1) > AMPLITUDE_THRESH_UV):
            continue

        # Flat signal rejection
        if np.any(epoch.std(axis=1) < 0.1):
            continue

        epochs.append(epoch)

    return np.array(epochs) if epochs else np.empty((0, n_channels, n_samples))


def compute_regional_lzc(epochs: np.ndarray,
                          raw: mne.io.BaseRaw) -> dict:
    """
    Compute mean LZC per channel group across all clean epochs.

    Returns dict of {region_lzc_mean: value, region_lzc_std: value, ...}
    """
    ch_names = raw.ch_names
    results  = {}

    for region, ch_list in CHANNEL_GROUPS.items():
        available_idx = [ch_names.index(ch) for ch in ch_list
                         if ch in ch_names]

        if not available_idx:
            results[f"{region}_lzc_mean"] = np.nan
            results[f"{region}_lzc_std"]  = np.nan
            results[f"{region}_n_channels"] = 0
            continue

        epoch_ch_vals = []
        for epoch in epochs:                   # (n_channels, n_samples)
            for idx in available_idx:
                val = lzc_single(epoch[idx])
                if not np.isnan(val):
                    epoch_ch_vals.append(val)

        if epoch_ch_vals:
            results[f"{region}_lzc_mean"] = round(float(np.mean(epoch_ch_vals)), 5)
            results[f"{region}_lzc_std"]  = round(float(np.std(epoch_ch_vals)),  5)
        else:
            results[f"{region}_lzc_mean"] = np.nan
            results[f"{region}_lzc_std"]  = np.nan

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

    # Light notch filter only — data already preprocessed by dataset authors
    raw.notch_filter(freqs=[50, 60], verbose=False)

    # Crop edges for stationarity
    duration = raw.times[-1]
    t_start  = min(30.0, duration * 0.1)
    t_end    = max(duration - 30.0, duration * 0.9)
    if t_end > t_start + 10.0:
        raw.crop(tmin=t_start, tmax=t_end)

    raw.pick("eeg", verbose=False)

    epochs   = make_epochs(raw)
    n_epochs = len(epochs)

    if n_epochs < MIN_EPOCHS:
        msg = f"Only {n_epochs} clean epochs (min={MIN_EPOCHS}) — skipping."
        print(f"    [WARN] {msg}")
        return {"subject_id": subject_id, "n_clean_epochs": n_epochs, "error": msg}

    print(f"    {n_epochs} clean epochs × {len(raw.ch_names)} channels")

    regional = compute_regional_lzc(epochs, raw)

    return {
        "subject_id":        subject_id,
        "recording_dur_s":   round(duration, 1),
        "sfreq_hz":          raw.info["sfreq"],
        "n_clean_epochs":    n_epochs,
        "epoch_length_s":    EPOCH_LENGTH_SEC,
        "binarise_method":   BINARISE_METHOD,
        **regional,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Lempel-Ziv Complexity from ds004504 EEG dataset."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--subject",  default=None,
                        help="Single subject ID, e.g. sub-001. Omit for all.")
    parser.add_argument("--output",   default=None,
                        help="Output CSV path. Default: {BASE_DIR}/results/LempelZiv.csv")
    args = parser.parse_args()

    base_dir   = os.path.abspath(args.base_dir)
    output_csv = args.output or os.path.join(base_dir, "results", "LempelZiv.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    participants = load_participants(base_dir)
    set_files    = find_set_files(base_dir, subject=args.subject)

    if not set_files:
        print(f"[ERROR] No .set files found under {base_dir}/derivatives/")
        return

    print(f"Found {len(set_files)} subject(s). Starting LZC extraction...\n")

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
    metric    = "posterior_lzc_mean"
    if group_col and metric in df.columns:
        print(f"\n--- Posterior LZC by {group_col} ---")
        summary = (df.groupby(group_col)[metric]
                     .agg(["count", "mean", "std"])
                     .rename(columns={"count": "n",
                                      "mean":  "mean_lzc",
                                      "std":   "std_lzc"}))
        summary["mean_lzc"] = summary["mean_lzc"].round(4)
        summary["std_lzc"]  = summary["std_lzc"].round(4)
        print(summary.to_string())


if __name__ == "__main__":
    main()
