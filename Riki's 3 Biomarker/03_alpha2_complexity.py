"""
03_alpha2_complexity.py
────────────────────────
Biomarker 2 – Decreased Alpha-2 Complexity

Computes per subject:
  • Alpha 2 absolute power     (10–13 Hz)
  • Alpha 2 relative power     (alpha2 / total power)
  • Spectral entropy in alpha 2 band  (higher = more complex)
  • Permutation entropy on alpha-2 band-filtered signal
    (higher = more complex; reduced in AD)

Input:  {BASE_DIR}/derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
Output: {BASE_DIR}/results/alpha2_complexity.csv
"""

import mne
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR         = "/mnt/lustre/koa/scratch/rikimacm/ds004504_eeg"
DATA_DIR         = os.path.join(BASE_DIR, "derivatives")
PARTICIPANTS_FILE = os.path.join(BASE_DIR, "participants.tsv")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA2_LO = 10.0   # Hz
ALPHA2_HI = 13.0   # Hz
TOTAL_LO  = 0.5
TOTAL_HI  = 45.0

# Permutation entropy parameters
PE_ORDER  = 3      # embedding dimension m
PE_DELAY  = 1      # time delay
# Max samples used for PE (subsampled for speed on long recordings)
PE_MAX_SAMPLES = 6000

REGIONS = {
    "frontal":   ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "central":   ["C3", "C4", "Cz"],
    "parietal":  ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}
# ──────────────────────────────────────────────────────────────────────────────


def compute_psd(raw: mne.io.BaseRaw):
    """Return (psds, freqs) using Welch with 2-s windows."""
    sfreq = raw.info["sfreq"]
    n_fft = int(2 * sfreq)
    data  = raw.get_data(picks="eeg")

    from mne.time_frequency import psd_array_welch
    psds, freqs = psd_array_welch(
        data, sfreq=sfreq,
        fmin=TOTAL_LO, fmax=TOTAL_HI,
        n_fft=n_fft, n_overlap=n_fft // 2,
    )
    return psds, freqs   # (n_ch, n_freqs)


def bandpass(data: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq  = sfreq / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def spectral_entropy(psd_1ch: np.ndarray, freqs: np.ndarray,
                     fmin: float, fmax: float) -> float:
    """Shannon entropy of the normalised PSD within [fmin, fmax].
    Reduced in AD (less complex spectral distribution)."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    p    = psd_1ch[mask]
    s    = p.sum()
    if s == 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def permutation_entropy(signal: np.ndarray, m: int = 3, delay: int = 1) -> float:
    """
    Compute Permutation Entropy (Bandt & Pompe 2002).
    O(N · m · log m) – fast even on long signals.
    Normalised to [0, 1] by dividing by log2(m!).
    """
    N  = len(signal)
    if N < PE_MAX_SAMPLES:
        sig = signal
    else:
        step = N // PE_MAX_SAMPLES
        sig  = signal[::step]
        N    = len(sig)

    counts: dict = {}
    n_patt = N - (m - 1) * delay
    if n_patt <= 0:
        return 0.0

    for i in range(n_patt):
        pattern = tuple(np.argsort(np.argsort([sig[i + j * delay] for j in range(m)])))
        counts[pattern] = counts.get(pattern, 0) + 1

    probs = np.array(list(counts.values()), dtype=float) / n_patt
    H     = -np.sum(probs * np.log2(probs + 1e-12))
    H_max = np.log2(np.math.factorial(m))
    return float(H / H_max) if H_max > 0 else 0.0


def compute_alpha2_features(raw: mne.io.BaseRaw) -> dict:
    sfreq    = raw.info["sfreq"]
    ch_names = raw.copy().pick("eeg").ch_names
    data     = raw.get_data(picks="eeg")   # (n_ch, n_times)

    psds, freqs = compute_psd(raw)

    # Masks
    a2_mask    = (freqs >= ALPHA2_LO) & (freqs <= ALPHA2_HI)
    total_mask = (freqs >= TOTAL_LO)  & (freqs <= TOTAL_HI)

    # Per-channel absolute alpha 2 power (trapz integration, units: V²)
    a2_power    = np.trapz(psds[:, a2_mask],    freqs[a2_mask],    axis=1)  # (n_ch,)
    total_power = np.trapz(psds[:, total_mask], freqs[total_mask], axis=1)  # (n_ch,)

    # Band-filtered data for permutation entropy
    a2_data = bandpass(data, sfreq, ALPHA2_LO, ALPHA2_HI)

    # Per-channel spectral entropy
    sp_ent = np.array([spectral_entropy(psds[i], freqs, ALPHA2_LO, ALPHA2_HI)
                       for i in range(len(ch_names))])

    results = {}

    # ── Global ───────────────────────────────────────────────────────────────
    results["global_alpha2_abs_power"]      = float(np.mean(a2_power))
    results["global_alpha2_rel_power"]      = float(np.mean(a2_power / (total_power + 1e-30)))
    results["global_alpha2_spec_entropy"]   = float(np.mean(sp_ent))

    # Global permutation entropy on spatially-averaged alpha-2 signal
    mean_a2 = np.mean(a2_data, axis=0)
    results["global_alpha2_perm_entropy"]   = permutation_entropy(mean_a2, PE_ORDER, PE_DELAY)

    # ── Regional ─────────────────────────────────────────────────────────────
    for region, chs in REGIONS.items():
        valid = [ch for ch in chs if ch in ch_names]
        if not valid:
            continue
        idx = [ch_names.index(ch) for ch in valid]

        results[f"{region}_alpha2_abs_power"]    = float(np.mean(a2_power[idx]))
        results[f"{region}_alpha2_rel_power"]    = float(
            np.mean(a2_power[idx] / (total_power[idx] + 1e-30)))
        results[f"{region}_alpha2_spec_entropy"] = float(np.mean(sp_ent[idx]))

        region_a2 = np.mean(a2_data[idx], axis=0)
        results[f"{region}_alpha2_perm_entropy"] = permutation_entropy(
            region_a2, PE_ORDER, PE_DELAY)

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
            features = compute_alpha2_features(raw)

            entry = {"subject": sub_id, "group": group, "age": age, "mmse": mmse}
            entry.update(features)
            rows.append(entry)

            pwr = features["global_alpha2_abs_power"]
            se  = features["global_alpha2_spec_entropy"]
            pe  = features["global_alpha2_perm_entropy"]
            print(f"  alpha2_power={pwr:.4e}  spec_entropy={se:.3f}  perm_entropy={pe:.3f}")

        except Exception as exc:
            print(f"[ERROR] {sub_id}: {exc}")

    if not rows:
        print("No results collected. Check DATA_DIR path.")
        return

    df       = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "alpha2_complexity.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # ── Group summary ─────────────────────────────────────────────────────────
    print("\n=== Group Summary (Alpha 2 Band) ===")
    group_labels = [("A", "Alzheimer"), ("F", "FTD"), ("C", "Healthy")]
    for g, label in group_labels:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        print(f"\n{label} (n={len(sub)}):")
        print(f"  Alpha2 Abs Power  : {sub['global_alpha2_abs_power'].mean():.4e}"
              f" ± {sub['global_alpha2_abs_power'].std():.4e}")
        print(f"  Alpha2 Rel Power  : {sub['global_alpha2_rel_power'].mean():.4f}"
              f" ± {sub['global_alpha2_rel_power'].std():.4f}")
        print(f"  Spectral Entropy  : {sub['global_alpha2_spec_entropy'].mean():.4f}"
              f" ± {sub['global_alpha2_spec_entropy'].std():.4f}")
        print(f"  Permutation Entrop: {sub['global_alpha2_perm_entropy'].mean():.4f}"
              f" ± {sub['global_alpha2_perm_entropy'].std():.4f}")


if __name__ == "__main__":
    main()
