"""
Alpha Peak Frequency (APF) Extractor — adapted for ds006036
Dataset: OpenNeuro ds006036 (eyes-open photic stimulation EEG, AD/FTD/CN)

⚠️  IMPORTANT NOTE ON VALIDITY:
    ds006036 is an eyes-open dataset with photic stimulation (5, 10, 15, 30 Hz).
    Photic stimulation drives SSVEPs that fall directly inside the alpha search
    window (especially 10 Hz and 15 Hz). APF extracted during stimulation epochs
    reflects the SSVEP, NOT endogenous alpha.
    
    Two modes are available (set ANALYSIS_MODE below):
      'full'     — run APF on the entire recording (fast, but SSVEP-contaminated)
      'baseline' — attempt to use only pre-stimulation baseline (recommended)

Input:  {BASE_DIR}/derivatives/eeglab/sub-XXX/eeg/sub-XXX_task-photomark_eeg.set
Output: {BASE_DIR}/results/APF_ds006036.csv

Usage:
    python 01_alpha_peak_frequency_ds006036.py --base_dir /path/to/ds006036
    python 01_alpha_peak_frequency_ds006036.py --base_dir /path/to/ds006036 --subject sub-001
    python 01_alpha_peak_frequency_ds006036.py --base_dir /path/to/ds006036 --mode full
"""

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

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

# Analysis mode: 'full' or 'baseline'
# In 'baseline' mode, only the first BASELINE_DURATION_S seconds are used,
# which likely precede photic stimulation onset (verify against events file).
ANALYSIS_MODE       = "baseline"
BASELINE_DURATION_S = 30.0   # seconds to use in baseline mode

# Photic stimulation frequencies to flag in output (for QC)
SSVEP_FREQS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

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
    Glob preprocessed .set files under derivatives/eeglab/.
    ds006036 structure: derivatives/eeglab/sub-XXX/eeg/sub-XXX_task-photomark_eeg.set
    """
    if subject:
        pattern = os.path.join(
            base_dir, "derivatives", "eeglab", subject, "eeg",
            f"{subject}_task-photomark_eeg.set"
        )
    else:
        pattern = os.path.join(
            base_dir, "derivatives", "eeglab", "sub-*", "eeg",
            "sub-*_task-photomark_eeg.set"
        )
    files = sorted(glob.glob(pattern))

    # Fallback: try flat derivatives layout (in case structure differs)
    if not files:
        if subject:
            pattern = os.path.join(
                base_dir, "derivatives", subject, "eeg",
                f"{subject}_task-photomark_eeg.set"
            )
        else:
            pattern = os.path.join(
                base_dir, "derivatives", "sub-*", "eeg",
                "sub-*_task-photomark_eeg.set"
            )
        files = sorted(glob.glob(pattern))

    return files


def pick_posterior(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    available = [ch for ch in POSTERIOR_CHANNELS if ch in raw.ch_names]
    if not available:
        available = raw.ch_names
        print("    [WARN] No posterior channels found — using all EEG channels.")
    return raw.pick(available)


def detect_ssvep_contamination(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """
    Check if any photic stimulation harmonics dominate the PSD.
    Returns ratio of SSVEP-band power to mean alpha power for each stim freq.
    Ratio > 2.0 suggests contamination.
    """
    results = {}
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha_power = float(np.mean(psd[alpha_mask])) if alpha_mask.any() else 1.0
    for sf in SSVEP_FREQS:
        mask = (freqs >= sf - 0.5) & (freqs <= sf + 0.5)
        if mask.any():
            ssvep_power = float(np.mean(psd[mask]))
            results[f"ssvep_{int(sf)}hz_ratio"] = round(ssvep_power / alpha_power, 3) if alpha_power > 0 else np.nan
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


def process_subject(set_path: str, mode: str = "baseline") -> dict:
    subject_id = os.path.basename(set_path).split("_")[0]

    # Resolve symlinks so MNE can find the file (datalad annex uses relative symlinks)
    set_path = os.path.realpath(set_path)
    
    print(f"  Processing {subject_id} (mode={mode}) ...")

    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception as e:
        print(f"    [ERROR] Could not load {set_path}: {e}")
        return {"subject_id": subject_id, "error": str(e)}

    duration = raw.times[-1]

    raw.notch_filter(freqs=[50, 60], verbose=False)
    raw = pick_posterior(raw)

    if mode == "baseline":
        # Use the first BASELINE_DURATION_S seconds as pre-stimulation baseline.
        # This assumes the recording starts before photic stimulation begins,
        # which is typical for this clinical protocol. Verify with events file.
        t_end = min(BASELINE_DURATION_S, duration - 5.0)
        if t_end < 10.0:
            print(f"    [WARN] Recording too short ({duration:.1f}s) for baseline mode — using full recording.")
            t_start, t_end = 0.0, duration
        else:
            t_start = 0.0
        raw.crop(tmin=t_start, tmax=t_end)
        segment_used = f"baseline_0-{t_end:.0f}s"
    else:
        # Full recording mode — skip first/last 10s (reduced from 30s for short recordings)
        skip = min(10.0, duration * 0.05)
        t_start = skip
        t_end   = duration - skip
        if t_end <= t_start + 5.0:
            t_start, t_end = 0.0, duration
        raw.crop(tmin=t_start, tmax=t_end)
        segment_used = f"full_{t_start:.0f}-{t_end:.0f}s"

    freqs, psd  = compute_psd_welch(raw)
    metrics     = extract_apf(freqs, psd)
    ssvep_flags = detect_ssvep_contamination(freqs, psd)

    result = {
        "subject_id":      subject_id,
        "recording_dur_s": round(duration, 1),
        "segment_used":    segment_used,
        "analysis_mode":   mode,
        "sfreq_hz":        raw.info["sfreq"],
        **metrics,
        **ssvep_flags,
        "error":           "",
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract APF from ds006036 (eyes-open photic stimulation EEG)."
    )
    parser.add_argument("--base_dir", required=True,
                        help="Root directory of ds006036.")
    parser.add_argument("--subject", default=None,
                        help="Process a single subject, e.g. sub-001.")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Default: {BASE_DIR}/results/APF_ds006036.csv")
    parser.add_argument("--mode", default=ANALYSIS_MODE, choices=["full", "baseline"],
                        help="'baseline' uses only pre-stimulation segment (recommended). "
                             "'full' uses the entire recording.")
    args = parser.parse_args()

    base_dir   = os.path.abspath(args.base_dir)
    output_csv = args.output or os.path.join(base_dir, "results", "APF_ds006036.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    participants = load_participants(base_dir)
    set_files = find_set_files(base_dir, subject=args.subject)

    if not set_files:
        print(f"[ERROR] No .set files found. Tried:")
        print(f"  {base_dir}/derivatives/eeglab/sub-*/eeg/sub-*_task-photomark_eeg.set")
        print(f"  {base_dir}/derivatives/sub-*/eeg/sub-*_task-photomark_eeg.set")
        print("Check your derivatives folder structure and adjust the path in find_set_files().")
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
            if not participants["subject_id"].str.startswith("sub-").all():
                participants["subject_id"] = "sub-" + participants["subject_id"].str.zfill(3)
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

        # Warn if SSVEP contamination is likely
        ssvep_cols = [c for c in df.columns if c.startswith("ssvep_")]
        if ssvep_cols:
            print(f"\n--- SSVEP contamination check (ratio > 2.0 = likely contaminated) ---")
            for col in ssvep_cols:
                high = (df[col] > 2.0).sum()
                if high > 0:
                    print(f"  {col}: {high}/{len(df)} subjects may be contaminated")


if __name__ == "__main__":
    main()
