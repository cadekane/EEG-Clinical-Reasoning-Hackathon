"""
Alpha Peak Frequency (APF) Extractor — adapted for ds005048
Dataset: OpenNeuro ds005048 (40 Hz auditory entrainment, AD/dementia + controls)

⚠️  IMPORTANT NOTES ON VALIDITY:
    ds005048 is an EYES-OPEN recording with 40 Hz amplitude-modulated auditory
    stimulation, structured as 40 s ON (stim) / 20 s OFF (rest) trials.
    Events file marks: 1 = rest start, 2 = stimulus start.

    Compared to ds006036 (photic flicker) the situation for APF is BETTER
    because 40 Hz is outside the 6–13 Hz alpha band — but it is still safest
    to compute APF on rest (OFF) epochs only, because (a) the 40 Hz ASSR has
    sub-harmonics (20 Hz, 13.3 Hz) that can leak into the upper alpha band,
    and (b) attentional engagement during stimulation can shift alpha.

    Three modes are available (--mode):
      'rest'     — use only OFF epochs from events.tsv  (RECOMMENDED, default)
      'baseline' — use first BASELINE_DURATION_S of recording (fallback if no events)
      'full'     — use the entire recording (fastest, most contamination)

Input layout (BIDS, NO derivatives subdir for this dataset):
    {BASE_DIR}/sub-XX/eeg/sub-XX_task-40HzAuditoryEntrainment_eeg.set
    {BASE_DIR}/sub-XX/eeg/sub-XX_task-40HzAuditoryEntrainment_events.tsv
Output:
    {BASE_DIR}/results/APF_ds005048.csv

Usage:
    python 01_alpha_peak_frequency_ds005048.py --base_dir /path/to/ds005048
    python 01_alpha_peak_frequency_ds005048.py --base_dir /path/to/ds005048 --subject sub-01
    python 01_alpha_peak_frequency_ds005048.py --base_dir /path/to/ds005048 --mode full
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# Local helper: handles MATLAB v7.3 (HDF5) .set files that mne.io.read_raw_eeglab
# cannot read because it falls back to scipy.io.loadmat under the hood.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eeglab_v73_loader import read_raw_eeglab_any

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Posterior electrodes — best for APF
POSTERIOR_CHANNELS = ["O1", "O2", "P3", "Pz", "P4"]

# Alpha band search window (Hz)
ALPHA_BAND = (6.0, 13.0)

# Welch PSD parameters
WELCH_WINDOW_SEC = 4.0
WELCH_OVERLAP    = 0.5

# Smoothing bins before peak picking
SMOOTH_BINS = 3

# Analysis mode: 'rest' | 'baseline' | 'full'
ANALYSIS_MODE       = "rest"
BASELINE_DURATION_S = 30.0   # used only in 'baseline' mode

# Auditory stimulation harmonics to flag in output (for QC).
# 40 Hz is outside alpha but its sub-harmonics can contaminate.
ASSR_FREQS_TO_CHECK = [13.33, 20.0, 40.0]

# ds005048 BIDS specifics
TASK_NAME = "40HzAuditoryEntrainment"
# Event codes: 1 = rest start, 2 = stimulus start
REST_CODE = "1"
STIM_CODE = "2"

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
    """
    ds005048 stores the .set files directly under sub-XX/eeg/, not under derivatives/.
    """
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
    """Load companion *_events.tsv for the same subject/task."""
    eeg_dir = os.path.dirname(set_path)
    base = os.path.basename(set_path).replace("_eeg.set", "_events.tsv")
    events_path = os.path.join(eeg_dir, base)
    if os.path.exists(events_path):
        return pd.read_csv(events_path, sep="\t")
    return pd.DataFrame()


def get_rest_intervals(events_df: pd.DataFrame, recording_dur: float) -> list:
    """
    From the events.tsv (BIDS), return list of (onset, offset) tuples for
    REST (OFF) periods. Trials alternate rest (code 1, 20 s) / stim (code 2, 40 s).

    Strategy: take every event whose value/trial_type is the rest code, and
    define its end as the next event onset (or recording end).
    """
    if events_df.empty:
        return []

    # The label column may be 'value', 'trial_type', or similar.
    label_col = None
    for c in ("value", "trial_type", "stim_type", "event_type"):
        if c in events_df.columns:
            label_col = c
            break
    if label_col is None:
        return []

    # Coerce labels to strings for comparison
    labels = events_df[label_col].astype(str).str.strip()
    onsets = events_df["onset"].astype(float).values

    intervals = []
    for i, lab in enumerate(labels):
        if lab == REST_CODE:
            start = float(onsets[i])
            end   = float(onsets[i + 1]) if i + 1 < len(onsets) else recording_dur
            # Trim 1 s from each edge to avoid event-onset transients
            start += 1.0
            end   -= 1.0
            if end - start >= 5.0:
                intervals.append((start, end))
    return intervals


def pick_posterior(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    available = [ch for ch in POSTERIOR_CHANNELS if ch in raw.ch_names]
    if not available:
        available = raw.ch_names
        print("    [WARN] No posterior channels found — using all EEG channels.")
    return raw.pick(available)


def detect_assr_contamination(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """Power at 40 Hz ASSR & sub-harmonics relative to mean alpha power."""
    results = {}
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha_power = float(np.mean(psd[alpha_mask])) if alpha_mask.any() else 1.0
    for sf in ASSR_FREQS_TO_CHECK:
        mask = (freqs >= sf - 0.5) & (freqs <= sf + 0.5)
        if mask.any():
            assr_power = float(np.mean(psd[mask]))
            key = f"assr_{str(sf).replace('.', 'p')}hz_ratio"
            results[key] = round(assr_power / alpha_power, 3) if alpha_power > 0 else np.nan
    return results


def compute_psd_welch(raw: mne.io.BaseRaw) -> tuple:
    sfreq = raw.info["sfreq"]
    nperseg = int(WELCH_WINDOW_SEC * sfreq)
    noverlap = int(nperseg * WELCH_OVERLAP)

    data = raw.get_data(units="uV")
    psds = []
    for ch_data in data:
        f, p = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                     window="hann", detrend="linear")
        psds.append(p)
    return f, np.mean(psds, axis=0)


def extract_apf(freqs: np.ndarray, psd: np.ndarray) -> dict:
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha_freqs = freqs[alpha_mask]
    alpha_psd   = psd[alpha_mask]

    alpha_psd_smooth = uniform_filter1d(alpha_psd, size=SMOOTH_BINS)

    peak_idx = np.argmax(alpha_psd_smooth)
    apf      = float(alpha_freqs[peak_idx])
    peak_amp = float(alpha_psd_smooth[peak_idx])

    iaf_cog = float(
        np.sum(alpha_freqs * alpha_psd_smooth) / np.sum(alpha_psd_smooth)
    )
    alpha_power = float(np.mean(alpha_psd))

    theta_mask  = (freqs >= 4.0) & (freqs < 8.0)
    theta_power = float(np.mean(psd[theta_mask])) if theta_mask.any() else np.nan
    theta_alpha = theta_power / alpha_power if alpha_power > 0 else np.nan

    delta_mask  = (freqs >= 0.5) & (freqs < 4.0)
    delta_power = float(np.mean(psd[delta_mask])) if delta_mask.any() else np.nan

    beta_mask  = (freqs >= 13.0) & (freqs <= 30.0)
    beta_power = float(np.mean(psd[beta_mask])) if beta_mask.any() else np.nan

    return {
        "apf_hz":             round(apf, 3),
        "iaf_cog_hz":         round(iaf_cog, 3),
        "alpha_power_uv2hz":  round(alpha_power, 4),
        "peak_amplitude":     round(peak_amp, 4),
        "theta_power_uv2hz":  round(theta_power, 4) if not np.isnan(theta_power) else np.nan,
        "delta_power_uv2hz":  round(delta_power, 4) if not np.isnan(delta_power) else np.nan,
        "beta_power_uv2hz":   round(beta_power, 4)  if not np.isnan(beta_power)  else np.nan,
        "theta_alpha_ratio":  round(theta_alpha, 4) if not np.isnan(theta_alpha) else np.nan,
        "channels_used":      ", ".join(POSTERIOR_CHANNELS),
    }


def process_subject(set_path: str, mode: str = "rest") -> dict:
    subject_id = os.path.basename(set_path).split("_")[0]
    print(f"  Processing {subject_id} (mode={mode}) ...")

    try:
        raw = read_raw_eeglab_any(set_path, preload=True)
    except Exception as e:
        print(f"    [ERROR] Could not load {set_path}: {e}")
        return {"subject_id": subject_id, "error": str(e)}

    duration = raw.times[-1]

    # Notch filter (Iran mains = 50 Hz; harmless to also apply 60 Hz)
    raw.notch_filter(freqs=[50, 60], verbose=False)
    raw = pick_posterior(raw)

    actual_mode = mode

    if mode == "rest":
        events_df = load_events_tsv(set_path)
        rest_intervals = get_rest_intervals(events_df, duration)
        if not rest_intervals:
            print(f"    [WARN] No rest intervals found in events.tsv — falling back to baseline mode.")
            actual_mode = "baseline"
        else:
            # Concatenate all rest segments
            rest_raws = [raw.copy().crop(tmin=t0, tmax=t1) for t0, t1 in rest_intervals]
            raw = mne.concatenate_raws(rest_raws, verbose=False)
            total_rest = sum(t1 - t0 for t0, t1 in rest_intervals)
            segment_used = f"rest_{len(rest_intervals)}segs_{total_rest:.0f}s"

    if actual_mode == "baseline":
        t_end = min(BASELINE_DURATION_S, duration - 5.0)
        if t_end < 10.0:
            print(f"    [WARN] Recording too short ({duration:.1f}s) — using full recording.")
            t_start, t_end = 0.0, duration
            actual_mode = "full"
        else:
            t_start = 0.0
        raw.crop(tmin=t_start, tmax=t_end)
        segment_used = f"baseline_0-{t_end:.0f}s"

    if actual_mode == "full":
        skip = min(10.0, duration * 0.05)
        t_start = skip
        t_end   = duration - skip
        if t_end <= t_start + 5.0:
            t_start, t_end = 0.0, duration
        raw.crop(tmin=t_start, tmax=t_end)
        segment_used = f"full_{t_start:.0f}-{t_end:.0f}s"

    freqs, psd  = compute_psd_welch(raw)
    metrics     = extract_apf(freqs, psd)
    assr_flags  = detect_assr_contamination(freqs, psd)

    result = {
        "subject_id":      subject_id,
        "recording_dur_s": round(duration, 1),
        "segment_used":    segment_used,
        "analysis_mode":   actual_mode,
        "sfreq_hz":        raw.info["sfreq"],
        **metrics,
        **assr_flags,
        "error":           "",
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract APF from ds005048 (40 Hz auditory entrainment EEG)."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Root directory of ds005048.")
    parser.add_argument("--subject", default=None,
                        help="Process a single subject, e.g. sub-01.")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Default: {BASE_DIR}/results/APF_ds005048.csv")
    parser.add_argument("--mode", default=ANALYSIS_MODE,
                        choices=["rest", "baseline", "full"],
                        help="'rest' uses OFF epochs from events.tsv (recommended). "
                             "'baseline' uses first 30 s. 'full' uses entire recording.")
    args = parser.parse_args()

    base_dir   = os.path.abspath(args.base_dir)
    output_csv = args.output or os.path.join(base_dir, "results", "APF_ds005048.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    participants = load_participants(base_dir)
    set_files = find_set_files(base_dir, subject=args.subject)

    if not set_files:
        print(f"[ERROR] No .set files found. Tried:")
        print(f"  {base_dir}/sub-*/eeg/sub-*_task-{TASK_NAME}_eeg.set")
        print("Check that the dataset is downloaded with the .set files actually fetched")
        print("(if you used datalad clone, you may need: datalad get sub-*/eeg/*.set).")
        return

    print(f"Found {len(set_files)} subject(s). Mode: {args.mode}\n")

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
            # ds005048 IDs are sub-01, sub-02, ... (zero-padded to 2 digits)
            if not participants["subject_id"].astype(str).str.startswith("sub-").all():
                participants["subject_id"] = "sub-" + participants["subject_id"].astype(str).str.zfill(2)
            df = df.merge(participants, on="subject_id", how="left")

    df = df.sort_values("subject_id").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"\nDone. Results saved to: {output_csv}")
    print(f"Subjects processed: {len(df)}")

    group_col = next((c for c in df.columns if c.lower() in ("group", "dx", "diagnosis")), None)
    if group_col and "apf_hz" in df.columns:
        print(f"\n--- APF summary by {group_col} (mode={args.mode}) ---")
        summary = df.groupby(group_col)["apf_hz"].agg(["count", "mean", "std"])
        summary.columns = ["n", "mean_apf_hz", "std_apf_hz"]
        summary["mean_apf_hz"] = summary["mean_apf_hz"].round(3)
        summary["std_apf_hz"]  = summary["std_apf_hz"].round(3)
        print(summary.to_string())

        assr_cols = [c for c in df.columns if c.startswith("assr_")]
        if assr_cols:
            print(f"\n--- ASSR contamination check (ratio > 2.0 = likely contaminated) ---")
            for col in assr_cols:
                high = (df[col] > 2.0).sum()
                if high > 0:
                    print(f"  {col}: {high}/{len(df)} subjects may be contaminated")


if __name__ == "__main__":
    main()
