"""
Lempel-Ziv Complexity (LZC) Extractor
Dataset: OpenNeuro ds005048 (40 Hz auditory entrainment, AD/dementia + controls)

⚠️  IMPORTANT NOTE ON VALIDITY:
    This dataset contains 40 Hz auditory entrainment, structured as
    40 s ON (stim) / 20 s OFF (rest) trials. Events file marks: 1 = rest,
    2 = stimulus.

    LZC, like SampEn, is highly sensitive to driven periodic activity.
    Strong 40 Hz ASSR will lower LZC for reasons unrelated to disease.
    Computing LZC on the full recording confounds stimulation effects with
    group differences. Use rest (OFF) epochs only.

    Modes (--mode):
      'rest'   — use only OFF epochs from events.tsv  (RECOMMENDED, default)
      'full'   — use the entire recording (legacy)

Key differences vs ds006036/ds004504 versions:
  - Task name      : "40HzAuditoryEntrainment"
  - Layout         : sub-XX/eeg/  (NO derivatives/ subdir)
  - Subject IDs    : zero-padded to 2 digits (sub-01..sub-35)
  - Sampling rate  : 250 Hz
  - Preprocessing  : Makoto pipeline already applied

Input:  {BASE_DIR}/sub-XX/eeg/sub-XX_task-40HzAuditoryEntrainment_eeg.set
Output: {BASE_DIR}/results/LempelZiv_ds005048.csv

Usage:
    python 03_lempel-ziv_complexity_ds005048.py --base_dir /path/to/ds005048
    python 03_lempel-ziv_complexity_ds005048.py --base_dir /path/to/ds005048 --subject sub-01
    python 03_lempel-ziv_complexity_ds005048.py --base_dir /path/to/ds005048 --mode full

Dependencies:
    pip install mne antropy numpy pandas scipy
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import mne
import antropy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eeglab_v73_loader import read_raw_eeglab_any

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHANNEL_GROUPS = {
    "frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central":   ["C3", "Cz", "C4"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "parietal":  ["P3", "Pz", "P4"],
    "occipital": ["O1", "O2"],
    "posterior": ["O1", "O2", "P3", "Pz", "P4"],
}

BINARISE_METHOD = "median"     # "median" | "mean" | "zero"
EPOCH_LENGTH_SEC = 5.0
MIN_EPOCHS       = 5
AMPLITUDE_THRESH_UV = 150.0

TASK_NAME      = "40HzAuditoryEntrainment"
ANALYSIS_MODE  = "rest"        # 'rest' | 'full'
REST_CODE      = "1"
STIM_CODE      = "2"

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
    """ds005048 stores .set directly under sub-XX/eeg/, not under derivatives/."""
    if subject:
        pattern = os.path.join(
            base_dir, subject, "eeg",
            f"{subject}_task-{TASK_NAME}_eeg.set"
        )
    else:
        pattern = os.path.join(
            base_dir, "sub-*", "eeg",
            f"sub-*_task-{TASK_NAME}_eeg.set"
        )
    return sorted(glob.glob(pattern))


def load_events_tsv(set_path: str) -> pd.DataFrame:
    eeg_dir = os.path.dirname(set_path)
    base = os.path.basename(set_path).replace("_eeg.set", "_events.tsv")
    events_path = os.path.join(eeg_dir, base)
    if os.path.exists(events_path):
        return pd.read_csv(events_path, sep="\t")
    return pd.DataFrame()


def get_rest_intervals(events_df: pd.DataFrame, recording_dur: float) -> list:
    if events_df.empty:
        return []
    label_col = None
    for c in ("value", "trial_type", "stim_type", "event_type"):
        if c in events_df.columns:
            label_col = c
            break
    if label_col is None:
        return []
    labels = events_df[label_col].astype(str).str.strip()
    onsets = events_df["onset"].astype(float).values

    intervals = []
    for i, lab in enumerate(labels):
        if lab == REST_CODE:
            start = float(onsets[i]) + 1.0
            end   = float(onsets[i + 1]) - 1.0 if i + 1 < len(onsets) else recording_dur - 1.0
            if end - start >= 5.0:
                intervals.append((start, end))
    return intervals


def select_segments(raw: mne.io.BaseRaw, set_path: str, mode: str):
    duration = raw.times[-1]
    actual_mode = mode

    if mode == "rest":
        events_df = load_events_tsv(set_path)
        rest_intervals = get_rest_intervals(events_df, duration)
        if rest_intervals:
            rest_raws = [raw.copy().crop(tmin=t0, tmax=t1) for t0, t1 in rest_intervals]
            raw_out = mne.concatenate_raws(rest_raws, verbose=False)
            total = sum(t1 - t0 for t0, t1 in rest_intervals)
            seg_used = f"rest_{len(rest_intervals)}segs_{total:.0f}s"
            return raw_out, seg_used, actual_mode
        else:
            print(f"    [WARN] No rest intervals found — falling back to 'full' mode.")
            actual_mode = "full"

    t_start = min(30.0, duration * 0.1)
    t_end   = max(duration - 30.0, duration * 0.9)
    if t_end > t_start + 10.0:
        raw_out = raw.copy().crop(tmin=t_start, tmax=t_end)
    else:
        raw_out = raw.copy()
    seg_used = f"full_{t_start:.0f}-{t_end:.0f}s"
    return raw_out, seg_used, actual_mode


def binarise(signal: np.ndarray, method: str = "median") -> np.ndarray:
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
    if signal.std() < 1e-10:
        return np.nan
    try:
        binary = binarise(signal, method=BINARISE_METHOD)
        lzc = antropy.lziv_complexity(binary, normalize=True)
        return float(lzc) if np.isfinite(lzc) else np.nan
    except Exception:
        return np.nan


def make_epochs(raw: mne.io.BaseRaw) -> np.ndarray:
    sfreq     = raw.info["sfreq"]
    n_samples = int(EPOCH_LENGTH_SEC * sfreq)
    data      = raw.get_data(units="uV")
    n_channels, n_times = data.shape

    n_epochs = n_times // n_samples
    epochs   = []

    for i in range(n_epochs):
        start = i * n_samples
        epoch = data[:, start : start + n_samples]

        if np.any(np.ptp(epoch, axis=1) > AMPLITUDE_THRESH_UV):
            continue
        if np.any(epoch.std(axis=1) < 0.1):
            continue

        epochs.append(epoch)

    return np.array(epochs) if epochs else np.empty((0, n_channels, n_samples))


def compute_regional_lzc(epochs: np.ndarray, raw: mne.io.BaseRaw) -> dict:
    ch_names = raw.ch_names
    results  = {}

    for region, ch_list in CHANNEL_GROUPS.items():
        available_idx = [ch_names.index(ch) for ch in ch_list if ch in ch_names]

        if not available_idx:
            results[f"{region}_lzc_mean"]  = np.nan
            results[f"{region}_lzc_std"]   = np.nan
            results[f"{region}_n_channels"] = 0
            continue

        epoch_ch_vals = []
        for epoch in epochs:
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

def process_subject(set_path: str, mode: str = "rest") -> dict:
    subject_id = os.path.basename(set_path).split("_")[0]
    print(f"  Processing {subject_id} (mode={mode}) ...")

    try:
        raw = read_raw_eeglab_any(set_path, preload=True)
    except Exception as e:
        print(f"    [ERROR] Could not load: {e}")
        return {"subject_id": subject_id, "error": str(e)}

    duration = raw.times[-1]

    raw.notch_filter(freqs=[50, 60], verbose=False)
    raw.pick("eeg", verbose=False)

    raw_seg, segment_used, actual_mode = select_segments(raw, set_path, mode)

    epochs   = make_epochs(raw_seg)
    n_epochs = len(epochs)

    if n_epochs < MIN_EPOCHS:
        msg = f"Only {n_epochs} clean epochs (min={MIN_EPOCHS}) — skipping."
        print(f"    [WARN] {msg}")
        return {
            "subject_id":     subject_id,
            "n_clean_epochs": n_epochs,
            "segment_used":   segment_used,
            "analysis_mode":  actual_mode,
            "error":          msg,
        }

    print(f"    {n_epochs} clean epochs × {len(raw_seg.ch_names)} channels  [{segment_used}]")

    regional = compute_regional_lzc(epochs, raw_seg)

    return {
        "subject_id":        subject_id,
        "recording_dur_s":   round(duration, 1),
        "segment_used":      segment_used,
        "analysis_mode":     actual_mode,
        "sfreq_hz":          raw_seg.info["sfreq"],
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
        description="Extract Lempel-Ziv Complexity from ds005048 EEG dataset."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Root directory of ds005048.")
    parser.add_argument("--subject", default=None,
                        help="Single subject ID, e.g. sub-01. Omit for all.")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Default: {BASE_DIR}/results/LempelZiv_ds005048.csv")
    parser.add_argument("--mode", default=ANALYSIS_MODE, choices=["rest", "full"],
                        help="'rest' uses OFF epochs from events.tsv (recommended). "
                             "'full' uses the entire recording.")
    args = parser.parse_args()

    base_dir   = os.path.abspath(args.base_dir)
    output_csv = args.output or os.path.join(base_dir, "results", "LempelZiv_ds005048.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    participants = load_participants(base_dir)
    set_files    = find_set_files(base_dir, subject=args.subject)

    if not set_files:
        print(f"[ERROR] No .set files found.")
        print(f"        Expected: {base_dir}/sub-*/eeg/sub-*_task-{TASK_NAME}_eeg.set")
        print(f"        If you used datalad clone, run: datalad get sub-*/eeg/*.set")
        return

    print(f"Found {len(set_files)} subject(s). Mode: {args.mode}. Starting LZC extraction...\n")

    records = []
    for set_path in set_files:
        result = process_subject(set_path, mode=args.mode)
        records.append(result)

    df = pd.DataFrame(records)

    if not participants.empty:
        id_col = next((c for c in participants.columns
                       if "participant" in c or c == "subject_id"), None)
        if id_col:
            participants = participants.rename(columns={id_col: "subject_id"})
            if not participants["subject_id"].astype(str).str.startswith("sub-").all():
                participants["subject_id"] = "sub-" + participants["subject_id"].astype(str).str.zfill(2)
            df = df.merge(participants, on="subject_id", how="left")

    df = df.sort_values("subject_id").reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print(f"\nDone. Results saved to: {output_csv}")
    print(f"Subjects processed: {len(df)}")

    group_col = next((c for c in df.columns
                      if c.lower() in ("group", "dx", "diagnosis")), None)
    metric    = "posterior_lzc_mean"
    if group_col and metric in df.columns:
        print(f"\n--- Posterior LZC by {group_col} (mode={args.mode}) ---")
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
