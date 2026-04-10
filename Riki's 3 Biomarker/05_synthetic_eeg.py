"""
05_synthetic_eeg.py
────────────────────
Generate synthetic EEG data for EEGPT training using three methods.
All methods operate on FULL recordings to preserve temporal dynamics.
Each method targets ~TARGET_FILES total outputs, distributed evenly
across subjects: ceil(TARGET_FILES / n_subjects) files per subject.
With 88 subjects and TARGET_FILES=1000: ceil(1000/88) = 12 per subject
→ 1056 total per method.

Method A — Frequency-Domain Augmentation
  Randomly scales per-band amplitudes (±20%) while preserving phase.
  Full recording length preserved.
  Phase preservation keeps inter-channel coherence and synchronization.
  Use for: spectral diversity without destroying spatial relationships.

Method B — Noise Augmentation (1/f pink noise)
  Adds controlled pink (1/f) noise to full recordings at a target SNR.
  Pink noise matches EEG's natural spectral slope (~1/f).
  Full recording length preserved.
  Use for: amplitude robustness, preventing overfitting to artifacts.

Method C — Combined (Freq Aug + Noise)
  Applies both frequency augmentation and pink noise to full recordings.
  Each copy gets independent random spectral scaling + noise.
  Full recording length preserved.
  Use for: EEGPT primary training set — maximum diversity.

Output structure:
  synthetic/freq/raw/      sub-XXX_freq_NNNN.set
  synthetic/noise/raw/     sub-XXX_noise_NNNN.set
  synthetic/combined/raw/  sub-XXX_combined_NNNN.set

Requires: eeglabio  (pip install eeglabio)
"""

import math
import mne
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR           = "/mnt/lustre/koa/scratch/rikimacm/ds004504_eeg"
DATA_DIR           = BASE_DIR  # raw files are directly in BASE_DIR (no subfolder)
PARTICIPANTS_FILE   = os.path.join(BASE_DIR, "participants.tsv")
OUTPUT_DIR         = os.path.join(BASE_DIR, "synthetic")
DATASET_NAME       = "raw"

# Target total output files per method.
# Actual count = ceil(TARGET_FILES / n_subjects) × n_subjects
# 88 subjects → ceil(1000/88) = 12 per subject → 1056 total
TARGET_FILES       = 1000

# Frequency-Domain Augmentation  (Methods A & C)
FREQ_SCALE_RANGE   = 0.20  # ± amplitude scale per band  (0.20 = ±20%)
FREQ_BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Noise Augmentation  (Methods B & C)
NOISE_SNR_DB       = 20.0  # 30 dB = subtle, 20 dB = mild, 10 dB = strong

METHODS = ["freq", "noise", "combined"]  # remove any to skip

rng = np.random.default_rng(42)
# ──────────────────────────────────────────────────────────────────────────────


def save_set(raw: mne.io.BaseRaw, path: str):
    """Export RawArray to EEGLAB .set format (requires eeglabio)."""
    raw.export(path, fmt="eeglab", overwrite=True)


def make_raw(data: np.ndarray, info: mne.Info) -> mne.io.RawArray:
    return mne.io.RawArray(data, info, verbose=False)


def load_subject(sub_id: str):
    in_path = os.path.join(DATA_DIR, sub_id, "eeg",
                           f"{sub_id}_task-eyesclosed_eeg.set")
    if not os.path.exists(in_path):
        print(f"  [SKIP] {sub_id}: file not found")
        return None
    return mne.io.read_raw_eeglab(in_path, preload=True)


def files_per_subject(n_subjects: int) -> int:
    """Files to generate per subject to reach ~TARGET_FILES total."""
    return math.ceil(TARGET_FILES / n_subjects)


# ─── Method A: Frequency-Domain Augmentation ─────────────────────────────────

def freq_domain_augment(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Randomly scale per-band amplitudes while preserving phase.
    Operates on the full recording — length unchanged.
    Phase intact → inter-channel coherence is fully preserved.
    """
    n_times   = data.shape[1]
    freqs     = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    augmented = np.zeros_like(data)

    for ch in range(data.shape[0]):
        spectrum = np.fft.rfft(data[ch])
        for fmin, fmax in FREQ_BANDS.values():
            mask  = (freqs >= fmin) & (freqs < fmax)
            scale = 1.0 + rng.uniform(-FREQ_SCALE_RANGE, FREQ_SCALE_RANGE)
            spectrum[mask] *= scale
        augmented[ch] = np.fft.irfft(spectrum, n=n_times)

    return augmented


# ─── Method B: Noise Augmentation ────────────────────────────────────────────

def pink_noise(n_samples: int, sfreq: float) -> np.ndarray:
    """Generate unit-variance pink (1/f) noise."""
    n_fft     = n_samples // 2 + 1
    freqs     = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    white     = rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)
    scale     = np.ones(n_fft)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    noise     = np.fft.irfft(white * scale, n=n_samples)
    noise    /= (noise.std() + 1e-12)
    return noise


def add_noise(data: np.ndarray, sfreq: float, snr_db: float) -> np.ndarray:
    """
    Add pink noise per channel at a target SNR (dB).
    Operates on the full recording — length unchanged.
    """
    augmented = np.zeros_like(data)
    for ch in range(data.shape[0]):
        signal        = data[ch]
        sig_rms       = np.sqrt(np.mean(signal ** 2))
        noise_rms     = sig_rms / (10 ** (snr_db / 20.0))
        augmented[ch] = signal + pink_noise(len(signal), sfreq) * noise_rms
    return augmented


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")
    n_subjects   = len(participants)
    per_sub      = files_per_subject(n_subjects)
    expected     = per_sub * n_subjects

    print(f"Participants : {n_subjects} subjects")
    print(f"Target files : {TARGET_FILES} per method  "
          f"→ {per_sub} per subject → {expected} actual")
    print(f"Active methods: {METHODS}")
    print(f"All methods operate on FULL recordings (length unchanged)\n")

    # ── Method A: Frequency-Domain Augmentation ───────────────────────────────
    if "freq" in METHODS:
        out_dir = os.path.join(OUTPUT_DIR, "freq", DATASET_NAME)
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== Method A: Frequency-Domain Augmentation "
              f"(±{FREQ_SCALE_RANGE*100:.0f}% per band, "
              f"{per_sub} copies/subject) ===")

        total = 0
        for _, row in participants.iterrows():
            sub_id = row["participant_id"]
            raw    = load_subject(sub_id)
            if raw is None:
                continue

            data  = raw.get_data(picks="eeg")
            info  = raw.copy().pick("eeg").info
            sfreq = info["sfreq"]

            for i in range(per_sub):
                aug_data = freq_domain_augment(data, sfreq)
                out_path = os.path.join(out_dir, f"{sub_id}_freq_{i+1:04d}.set")
                save_set(make_raw(aug_data, info), out_path)

            total += per_sub
            print(f"  [DONE] {sub_id}: {per_sub} copies")

        print(f"  Total saved: {total}\n")

    # ── Method B: Noise Augmentation ─────────────────────────────────────────
    if "noise" in METHODS:
        out_dir = os.path.join(OUTPUT_DIR, "noise", DATASET_NAME)
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== Method B: Noise Augmentation "
              f"(SNR={NOISE_SNR_DB:.0f} dB, {per_sub} copies/subject) ===")

        total = 0
        for _, row in participants.iterrows():
            sub_id = row["participant_id"]
            raw    = load_subject(sub_id)
            if raw is None:
                continue

            data  = raw.get_data(picks="eeg")
            info  = raw.copy().pick("eeg").info
            sfreq = info["sfreq"]

            for i in range(per_sub):
                noisy_data = add_noise(data, sfreq, NOISE_SNR_DB)
                out_path   = os.path.join(out_dir,
                                          f"{sub_id}_noise_{i+1:04d}.set")
                save_set(make_raw(noisy_data, info), out_path)

            total += per_sub
            print(f"  [DONE] {sub_id}: {per_sub} copies")

        print(f"  Total saved: {total}\n")

    # ── Method C: Combined (Freq Aug + Noise on full recording) ──────────────
    if "combined" in METHODS:
        out_dir = os.path.join(OUTPUT_DIR, "combined", DATASET_NAME)
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== Method C: Combined "
              f"(freq aug ±{FREQ_SCALE_RANGE*100:.0f}% + "
              f"noise {NOISE_SNR_DB:.0f} dB, {per_sub} copies/subject) ===")

        total = 0
        for _, row in participants.iterrows():
            sub_id = row["participant_id"]
            raw    = load_subject(sub_id)
            if raw is None:
                continue

            data  = raw.get_data(picks="eeg")
            info  = raw.copy().pick("eeg").info
            sfreq = info["sfreq"]

            for i in range(per_sub):
                aug_data   = freq_domain_augment(data, sfreq)
                noisy_data = add_noise(aug_data, sfreq, NOISE_SNR_DB)
                out_path   = os.path.join(out_dir,
                                          f"{sub_id}_combined_{i+1:04d}.set")
                save_set(make_raw(noisy_data, info), out_path)

            total += per_sub
            print(f"  [DONE] {sub_id}: {per_sub} copies")

        print(f"  Total saved: {total}\n")

    print("=== Done ===")
    print(f"Output root: {OUTPUT_DIR}/")
    for m in METHODS:
        print(f"  {m:10s}→  {OUTPUT_DIR}/{m}/{DATASET_NAME}/")


if __name__ == "__main__":
    main()
