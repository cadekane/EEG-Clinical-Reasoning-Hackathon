"""
02_spectral_biomarkers.py
──────────────────────────
Biomarker 1 – Spectral Power Ratios

Computes per subject:
  • Theta/Alpha ratio  (theta 4–8 Hz  / alpha 8–13 Hz)
  • DTABR              (delta+theta) / (alpha+beta)

Both globally (mean across all channels) and per brain region.

Input:  {BASE_DIR}/derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
Output: {BASE_DIR}/results/spectral_biomarkers.csv
"""

import mne
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR         = "/mnt/lustre/koa/scratch/rikimacm/ds004504_eeg"
DATA_DIR         = os.path.join(BASE_DIR, "derivatives")   # preprocessed data
PARTICIPANTS_FILE = os.path.join(BASE_DIR, "participants.tsv")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Frequency bands (Hz)
BANDS = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha1": (8.0, 10.0),
    "alpha2": (10.0, 13.0),
    "alpha":  (8.0, 13.0),
    "beta":   (13.0, 30.0),
    "gamma":  (30.0, 45.0),
}

# 10-20 regional channel groups
REGIONS = {
    "frontal":   ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "central":   ["C3", "C4", "Cz"],
    "parietal":  ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}
# ──────────────────────────────────────────────────────────────────────────────


def compute_psd(raw: mne.io.BaseRaw):
    """Return (psds, freqs) using Welch with 2-s windows, version-safe."""
    sfreq  = raw.info["sfreq"]
    n_fft  = int(2 * sfreq)
    data   = raw.get_data(picks="eeg")

    from mne.time_frequency import psd_array_welch
    psds, freqs = psd_array_welch(
        data, sfreq=sfreq,
        fmin=0.5, fmax=45.0,
        n_fft=n_fft, n_overlap=n_fft // 2,
    )
    return psds, freqs   # (n_ch, n_freqs)


def band_power(psds: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Integrated PSD power in [fmin, fmax] for every channel (trapz). Returns (n_ch,)."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psds[:, mask], freqs[mask], axis=1)


def compute_spectral_features(raw: mne.io.BaseRaw) -> dict:
    psds, freqs  = compute_psd(raw)
    ch_names     = raw.copy().pick("eeg").ch_names

    # Per-channel band powers
    bp = {b: band_power(psds, freqs, lo, hi) for b, (lo, hi) in BANDS.items()}

    results = {}

    # ── Global features ──────────────────────────────────────────────────────
    for b, arr in bp.items():
        results[f"global_{b}_power"] = float(np.mean(arr))

    g_theta = np.mean(bp["theta"])
    g_alpha = np.mean(bp["alpha"])
    g_delta = np.mean(bp["delta"])
    g_beta  = np.mean(bp["beta"])

    results["global_theta_alpha_ratio"] = g_theta / (g_alpha + 1e-30)
    results["global_DTABR"]             = (g_delta + g_theta) / (g_alpha + g_beta + 1e-30)

    # ── Regional features ────────────────────────────────────────────────────
    for region, chs in REGIONS.items():
        valid = [ch for ch in chs if ch in ch_names]
        if not valid:
            continue
        idx   = [ch_names.index(ch) for ch in valid]

        r_theta = np.mean(bp["theta"][idx])
        r_alpha = np.mean(bp["alpha"][idx])
        r_delta = np.mean(bp["delta"][idx])
        r_beta  = np.mean(bp["beta"][idx])

        results[f"{region}_theta_alpha_ratio"] = r_theta / (r_alpha + 1e-30)
        results[f"{region}_DTABR"]             = (r_delta + r_theta) / (r_alpha + r_beta + 1e-30)

        for b, arr in bp.items():
            results[f"{region}_{b}_power"] = float(np.mean(arr[idx]))

    return results


def main():
    participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")
    rows = []

    for _, row in participants.iterrows():
        sub_id = row["participant_id"]
        group  = row["Group"]
        age    = row["Age"]
        mmse   = row["MMSE"]

        in_path = os.path.join(DATA_DIR, sub_id, "eeg",
                               f"{sub_id}_task-eyesclosed_eeg.set")
        if not os.path.exists(in_path):
            print(f"[SKIP] {sub_id}: file not found")
            continue

        try:
            print(f"[PROC] {sub_id} (Group={group}) ...")
            raw      = mne.io.read_raw_eeglab(in_path, preload=True)
            features = compute_spectral_features(raw)

            entry = {"subject": sub_id, "group": group, "age": age, "mmse": mmse}
            entry.update(features)
            rows.append(entry)

            ta  = features["global_theta_alpha_ratio"]
            dta = features["global_DTABR"]
            print(f"  theta/alpha={ta:.3f}  DTABR={dta:.3f}")

        except Exception as exc:
            print(f"[ERROR] {sub_id}: {exc}")

    if not rows:
        print("No results collected. Check DATA_DIR path.")
        return

    df       = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "spectral_biomarkers.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # ── Group summary ─────────────────────────────────────────────────────────
    print("\n=== Group Summary ===")
    group_labels = [("A", "Alzheimer"), ("F", "FTD"), ("C", "Healthy")]
    for g, label in group_labels:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        ta  = sub["global_theta_alpha_ratio"]
        dta = sub["global_DTABR"]
        print(f"\n{label} (n={len(sub)}):")
        print(f"  Theta/Alpha : {ta.mean():.3f} ± {ta.std():.3f}")
        print(f"  DTABR       : {dta.mean():.3f} ± {dta.std():.3f}")


if __name__ == "__main__":
    main()
