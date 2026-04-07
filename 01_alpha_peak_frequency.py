"""
Alpha Peak Frequency (APF) Extractor
Dataset: OpenNeuro ds004504 (Alzheimer's/FTD/CN resting-state EEG)

Input:  {BASE_DIR}/derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
Output: {BASE_DIR}/results/APF.csv

Usage:
    python compute_apf.py --base_dir /path/to/ds004504
    python compute_apf.py --base_dir /path/to/ds004504 --subject sub-001
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

# Posterior electrodes — best for APF (maximally express alpha rhythm)
POSTERIOR_CHANNELS = ["O1", "O2", "P3", "Pz", "P4"]

# Alpha band search window (Hz)
# Wider than classic 8–13 to catch slowed peaks in AD
ALPHA_BAND = (6.0, 13.0)

# Welch PSD parameters
WELCH_WINDOW_SEC = 4.0    # Window length in seconds
WELCH_OVERLAP    = 0.5    # 50% overlap
FREQ_RESOLUTION  = 0.25  # Target frequency resolution (Hz)

# Smoothing: number of frequency bins for moving average before peak picking
SMOOTH_BINS = 3

# Group labels derived from MMSE ranges (fallback if participants.tsv unavailable)
# Primarily loaded from participants.tsv
GROUP_MMSE_THRESHOLDS = {"CN": (27, 30), "MCI": (18, 26), "AD": (0, 17)}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_participants(base_dir: str) -> pd.DataFrame:
    """Load participants.tsv for group labels and MMSE scores."""
    tsv_path = os.path.join(base_dir, "participants.tsv")
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t")
        df.columns = df.columns.str.lower()
        return df
    return pd.DataFrame()


def find_set_files(base_dir: str, subject: str = None) -> list:
    """Glob all preprocessed .set files under derivatives/."""
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
    files = sorted(glob.glob(pattern))
    return files


def pick_posterior(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Select posterior channels that are present in this recording."""
    available = [ch for ch in POSTERIOR_CHANNELS if ch in raw.ch_names]
    if not available:
        # Fall back to all EEG channels
        available = raw.ch_names
        print("    [WARN] No posterior channels found — using all EEG channels.")
    return raw.pick(available)


def compute_psd_welch(raw: mne.io.BaseRaw) -> tuple:
    """
    Compute Welch PSD averaged across selected channels.

    Returns
    -------
    freqs : np.ndarray  shape (n_freqs,)
    psd   : np.ndarray  shape (n_freqs,)  — mean across channels, in µV²/Hz
    """
    sfreq = raw.info["sfreq"]
    nperseg = int(WELCH_WINDOW_SEC * sfreq)
    noverlap = int(nperseg * WELCH_OVERLAP)

    data = raw.get_data(units="uV")  # (n_channels, n_times)

    psds = []
    for ch_data in data:
        f, p = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                     window="hann", detrend="linear")
        psds.append(p)

    psd_mean = np.mean(psds, axis=0)
    return f, psd_mean


def extract_apf(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """
    Extract Alpha Peak Frequency and related metrics.

    Returns a dict with:
        apf           : peak frequency in alpha band (Hz)
        alpha_power   : mean power in alpha band (µV²/Hz)
        peak_amplitude: PSD value at the peak
        iaf_cog       : Individual Alpha Frequency via centre of gravity
        theta_alpha_ratio : theta-band power / alpha-band power
    """
    # Restrict to alpha search window
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha_freqs = freqs[alpha_mask]
    alpha_psd   = psd[alpha_mask]

    # Smooth PSD to reduce spurious peaks
    alpha_psd_smooth = uniform_filter1d(alpha_psd, size=SMOOTH_BINS)

    # Peak frequency (maximum power in alpha band)
    peak_idx  = np.argmax(alpha_psd_smooth)
    apf       = float(alpha_freqs[peak_idx])
    peak_amp  = float(alpha_psd_smooth[peak_idx])

    # Centre of gravity (weighted mean frequency) — more robust alternative
    iaf_cog = float(
        np.sum(alpha_freqs * alpha_psd_smooth) / np.sum(alpha_psd_smooth)
    )

    # Mean alpha power
    alpha_power = float(np.mean(alpha_psd[alpha_mask]))

    # Theta band (4–8 Hz) power for theta/alpha ratio
    theta_mask  = (freqs >= 4.0) & (freqs < 8.0)
    theta_power = float(np.mean(psd[theta_mask])) if theta_mask.any() else np.nan
    theta_alpha = theta_power / alpha_power if alpha_power > 0 else np.nan

    # Delta band (0.5–4 Hz)
    delta_mask  = (freqs >= 0.5) & (freqs < 4.0)
    delta_power = float(np.mean(psd[delta_mask])) if delta_mask.any() else np.nan

    # Beta band (13–30 Hz)
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


def process_subject(set_path: str) -> dict:
    """Load, preprocess minimally, and extract APF for one subject."""
    subject_id = os.path.basename(set_path).split("_")[0]  # e.g. sub-001

    print(f"  Processing {subject_id} ...")

    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception as e:
        print(f"    [ERROR] Could not load {set_path}: {e}")
        return {"subject_id": subject_id, "error": str(e)}

    # The derivatives files are already bandpass-filtered (0.5–45 Hz) and
    # ICA-cleaned by the dataset authors. We apply a light notch just in case,
    # then pick posterior channels.
    # Do NOT re-bandpass aggressively — honour the dataset's preprocessing.
    raw.notch_filter(freqs=[50, 60], verbose=False)   # power line

    raw = pick_posterior(raw)

    # Crop to a stable middle segment (skip first/last 30 s for stationarity)
    duration = raw.times[-1]
    t_start  = min(30.0, duration * 0.1)
    t_end    = max(duration - 30.0, duration * 0.9)
    if t_end > t_start + 10.0:
        raw.crop(tmin=t_start, tmax=t_end)

    freqs, psd = compute_psd_welch(raw)
    metrics    = extract_apf(freqs, psd)

    result = {
        "subject_id":       subject_id,
        "recording_dur_s":  round(duration, 1),
        "sfreq_hz":         raw.info["sfreq"],
        **metrics,
        "error":            "",
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Alpha Peak Frequency from ds004504 EEG dataset."
    )
    parser.add_argument(
        "--base_dir", required=True,
        help="Root directory of the dataset (contains participants.tsv, derivatives/, etc.)"
    )
    parser.add_argument(
        "--subject", default=None,
        help="Process a single subject, e.g. sub-001. Omit to process all."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Default: {BASE_DIR}/results/APF.csv"
    )
    args = parser.parse_args()

    base_dir   = os.path.abspath(args.base_dir)
    output_csv = args.output or os.path.join(base_dir, "results", "APF.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load participant metadata
    participants = load_participants(base_dir)

    # Find .set files
    set_files = find_set_files(base_dir, subject=args.subject)
    if not set_files:
        print(f"[ERROR] No .set files found under {base_dir}/derivatives/")
        return

    print(f"Found {len(set_files)} subject(s). Starting APF extraction...\n")

    records = []
    for set_path in set_files:
        result = process_subject(set_path)
        records.append(result)

    df = pd.DataFrame(records)

    # Merge with participants.tsv if available
    if not participants.empty:
        # Normalise subject column name
        id_col = next((c for c in participants.columns
                       if "participant" in c or c == "subject_id"), None)
        if id_col:
            participants = participants.rename(columns={id_col: "subject_id"})
            # Ensure matching format (some tsv files use 'sub-001', others '001')
            if not participants["subject_id"].str.startswith("sub-").all():
                participants["subject_id"] = "sub-" + participants["subject_id"].str.zfill(3)
            df = df.merge(participants, on="subject_id", how="left")

    # Sort by subject_id
    df = df.sort_values("subject_id").reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"\nDone. Results saved to: {output_csv}")
    print(f"Subjects processed: {len(df)}")

    # Quick summary statistics per group if group column is available
    group_col = next((c for c in df.columns if c.lower() in ("group", "dx", "diagnosis")), None)
    if group_col and "apf_hz" in df.columns:
        print(f"\n--- APF summary by {group_col} ---")
        summary = df.groupby(group_col)["apf_hz"].agg(["count", "mean", "std"])
        summary.columns = ["n", "mean_apf_hz", "std_apf_hz"]
        summary["mean_apf_hz"] = summary["mean_apf_hz"].round(3)
        summary["std_apf_hz"]  = summary["std_apf_hz"].round(3)
        print(summary.to_string())


if __name__ == "__main__":
    main()
