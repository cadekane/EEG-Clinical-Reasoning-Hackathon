"""
preprocess_and_extract_hpc.py
---------------------------------
HPC version of the EEG pipeline for KOA (SLURM) cluster.
Paths use /mnt/lustre/koa/scratch/$USER instead of local Windows paths.

Runs the full pipeline on raw BIDS EEG data:
  raw .set  ->  preprocess  ->  extract biomarkers  ->  save CSVs

Preprocessing (per dataset README):
  1. Rename old 10-20 channel names + set standard_1020 montage
  2. Butterworth bandpass 0.5-45 Hz  (4th-order, zero-phase)
  3. Artifact Subspace Reconstruction (ASR)
       ds004504: cutoff=17  |  ds006036: cutoff=15  |  ds005048: cutoff=15
  4. ICA -- extended Infomax (19 components)
  5. ICLabel -- reject Eye + Muscle components (p > 0.5)

Biomarkers extracted after cleaning:
  1. Spectral   -- band power (delta/theta/alpha1/alpha2/beta/gamma), relative power,
                   theta/alpha ratio, DTABR, slowing ratio
  2. Alpha-2    -- absolute & relative power, peak frequency, hemispheric asymmetry,
                   permutation entropy, spectral entropy
  3. Microstates -- ModK-Means (k=4, A-D): coverage, mean duration, occurrence, GEV,
                    inter-state transition probabilities

Datasets supported:
  - eyes_closed  (ds004504 -- eyes-closed resting state)
  - eyes_open    (ds006036 -- eyes-open / photic stimulation)
  - auditory     (ds005048 -- 40 Hz auditory entrainment)

Output ($SCRATCH/biomarker_results/):
  eyes_closed_biomarkers.csv      per-subject, ds004504
  eyes_open_biomarkers.csv        per-subject, ds006036
  combined_biomarkers.csv         per-subject avg of eyes_closed + eyes_open (88 subjects)
  auditory_biomarkers.csv         per-subject, ds005048 (standalone)
  *_summary.csv                   group-level mean +/- std
  key_metrics_summary.csv         human-readable key biomarkers, mean +/- std

Requirements:
  pip install mne mne-icalabel meegkit onnxruntime pycrostates antropy scipy h5py

Usage:
  # Process eyes_closed + eyes_open (default):
  python preprocess_and_extract_hpc.py

  # Process only auditory -- loads saved eyes CSVs for combined output:
  python preprocess_and_extract_hpc.py --dataset auditory

  # Process all three datasets:
  python preprocess_and_extract_hpc.py --dataset all

  # Single subject:
  python preprocess_and_extract_hpc.py --dataset auditory --subject sub-001

  # Single EEG file (no BIDS structure needed):
  python preprocess_and_extract_hpc.py --file /path/to/eeg.set

  # All .set files in a folder:
  python preprocess_and_extract_hpc.py --folder /path/to/eeg_folder

  # Also save preprocessed .set files:
  python preprocess_and_extract_hpc.py --dataset all --save_set
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import mne
import antropy
from pathlib import Path
from scipy.signal import welch, find_peaks
from scipy.optimize import linear_sum_assignment
from pycrostates.cluster import ModKMeans

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# -- Paths (HPC) ---------------------------------------------------------------
SCRATCH = Path(f"/mnt/lustre/koa/scratch/{os.environ.get('USER', 'user')}")
OUT_DIR = SCRATCH / "biomarker_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations -- raw BIDS directories on scratch storage
DATASETS = {
    "eyes_closed": {
        "raw_dir":     SCRATCH / "ds004504_eeg",
        "file_suffix": "task-eyesclosed_eeg",
        "asr_cutoff":  17,
        "meta_file":   SCRATCH / "ds004504_eeg" / "participants.tsv",
        "group_col":   "Group",
    },
    "eyes_open": {
        "raw_dir":     SCRATCH / "ds006036_eeg",
        "file_suffix": "task-photomark_eeg",
        "asr_cutoff":  15,
        "meta_file":   SCRATCH / "ds004504_eeg" / "participants.tsv",  # shared participants file
        "group_col":   "Group",
    },
    "auditory": {
        "raw_dir":     SCRATCH / "ds005048_eeg",
        "file_suffix": "40HzAuditoryEntrainment_eeg",
        "asr_cutoff":  15,
        "meta_file":   SCRATCH / "ds005048_eeg" / "participants.tsv",
        "group_col":   "Group",
        "group_map": {          # map ds005048 labels -> canonical labels; None = exclude
            "Normal":      "CN",
            "Mild AD":     "AD",
            "Moderate AD": "AD",
            "MCI":         None,   # excluded -- clinically distinct, only 6 subjects
            "-":           None,   # missing label
        },
    },
}

# -- Preprocessing parameters --------------------------------------------------
LOWCUT         = 0.5
HIGHCUT        = 45.0
ICA_N_COMP     = 19
ICA_SEED       = 42
ICLABEL_REJECT = {"eye blink", "muscle artifact"}
ICLABEL_THRESH = 0.5

# -- Biomarker parameters ------------------------------------------------------
BANDS = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha1": (8.0, 10.0),
    "alpha2": (10.0, 13.0),
    "alpha":  (8.0, 13.0),
    "beta":   (13.0, 30.0),
    "gamma":  (30.0, 45.0),
}

REGIONS = {
    "frontal":   ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"],
    "central":   ["C3", "CZ", "C4"],
    "temporal":  ["T7", "T8"],
    "parietal":  ["P7", "P3", "PZ", "P4", "P8"],
    "occipital": ["O1", "O2"],
    "posterior": ["O1", "O2", "P3", "PZ", "P4"],
    "left":      ["FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"],
    "right":     ["FP2", "F8", "F4", "T8", "C4", "P8", "P4", "O2"],
}

N_STATES       = 4
MS_LABELS      = ["A", "B", "C", "D"]
MS_RESAMPLE_HZ = 250
MS_MAX_S       = 5 * 60
WELCH_WIN_S    = 2.0

# Old 10-20 -> standard names
CH_RENAME = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "Fp1": "FP1", "Fp2": "FP2", "Fz": "FZ", "Cz": "CZ", "Pz": "PZ",
}

# Canonical microstate reference topographies (Koenig et al. 2002)
CANONICAL_MAPS = {
    "A": {"O1": +1, "P7": +1, "P3": +1, "FP2": -1, "F8": -1, "F4": -1},
    "B": {"O2": +1, "P8": +1, "P4": +1, "FP1": -1, "F7": -1, "F3": -1},
    "C": {"FP1": +1, "FP2": +1, "F7": +1, "F3": +1, "FZ": +1, "F4": +1, "F8": +1,
          "O1": -1, "O2": -1, "P7": -1, "P3": -1, "PZ": -1, "P4": -1, "P8": -1},
    "D": {"CZ": +1, "C3": +1, "C4": +1, "PZ": +1, "P3": +1, "P4": +1,
          "FP1": -1, "FP2": -1, "F3": -1, "F4": -1, "FZ": -1},
}


# ==============================================================================
#  PREPROCESSING STEPS
# ==============================================================================

def step_normalise_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Rename old 10-20 names, uppercase all channels, drop non-EEG, set montage."""
    rename = {ch: CH_RENAME[ch] for ch in raw.ch_names if ch in CH_RENAME}
    if rename:
        raw.rename_channels(rename)
    raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})

    NON_EEG = {"EOG", "EMG", "ECG", "EKG", "STI", "STIM", "TRIGGER",
               "REF", "A1", "A2", "M1", "M2", "STATUS", "MISC"}
    drop = [ch for ch in raw.ch_names if any(ch.startswith(k) for k in NON_EEG)]
    if drop:
        raw.drop_channels(drop)

    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose=False)

    known   = set(sum(REGIONS.values(), []))
    missing = known - set(raw.ch_names)
    if missing:
        print(f"    [channels] Missing standard 10-20: {sorted(missing)}")
    return raw


def step_filter(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Butterworth bandpass 0.5-45 Hz (4th-order, zero-phase)."""
    raw.filter(
        LOWCUT, HIGHCUT,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )
    return raw


def step_asr(raw: mne.io.BaseRaw, cutoff: int) -> mne.io.BaseRaw:
    """ASR via meegkit; falls back to MNE annotate_amplitude if unavailable."""
    try:
        from meegkit.asr import ASR  # type: ignore
        data        = raw.get_data().astype(np.float64)
        sfreq       = raw.info["sfreq"]
        cal_samples = int(60 * sfreq)

        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        asr = ASR(method="euclid", cutoff=cutoff)
        asr.fit(np.ascontiguousarray(data[:, :cal_samples]))
        result  = asr.transform(np.ascontiguousarray(data))
        cleaned = np.asarray(result[0] if isinstance(result, tuple) else result,
                             dtype=np.float64)

        if cleaned.shape != data.shape:
            if cleaned.T.shape == data.shape:
                cleaned = cleaned.T
            else:
                raise ValueError(f"unexpected ASR output shape {cleaned.shape}")

        ann = raw.annotations
        raw = mne.io.RawArray(cleaned, raw.info, verbose=False)
        raw.set_annotations(ann)
        print(f"    [ASR] cutoff={cutoff} applied")

    except ImportError:
        print(f"    [ASR] meegkit not found -- using MNE annotate_amplitude fallback")
        cal_std   = raw.copy().crop(tmax=min(60.0, raw.times[-1])).get_data().std(axis=1)
        threshold = float(cutoff * np.median(cal_std))
        annotations, _ = mne.preprocessing.annotate_amplitude(
            raw, peak=threshold, min_duration=0.1, picks="eeg", verbose=False)
        if len(annotations):
            raw.set_annotations(raw.annotations + annotations)

    return raw


def step_ica(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Extended Infomax ICA + ICLabel component rejection (eye + muscle)."""
    try:
        from mne_icalabel import label_components  # type: ignore
    except ImportError:
        print("    [ICA] mne-icalabel not installed -- skipping. pip install mne-icalabel onnxruntime")
        return raw

    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMP,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=ICA_SEED,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw, picks="eeg", reject_by_annotation=True, verbose=False)

    try:
        labels  = label_components(raw, ica, method="iclabel")
        probs   = labels["y_pred_proba"]
        names   = labels["labels"]
        exclude = [idx for idx, (lbl, pv) in enumerate(zip(names, probs))
                   if lbl in ICLABEL_REJECT and pv.max() >= ICLABEL_THRESH]

        # Always keep at least 2 components so downstream signal is non-trivial
        MIN_KEEP = 2
        if len(exclude) > ICA_N_COMP - MIN_KEEP:
            exclude = sorted(exclude,
                             key=lambda i: probs[i].max(), reverse=True)[: ICA_N_COMP - MIN_KEEP]
            print(f"    [ICA] capped rejection to {len(exclude)} components (keeping {MIN_KEEP})")

        print(f"    [ICA] {len(exclude)}/{ICA_N_COMP} components rejected "
              f"({[names[i] for i in exclude]})")
        ica.exclude = exclude
        ica.apply(raw, verbose=False)
    except Exception as e:
        print(f"    [ICA] ICLabel failed ({e}) -- no components rejected")

    return raw


def preprocess(raw: mne.io.BaseRaw, asr_cutoff: int) -> mne.io.BaseRaw:
    """Apply the full preprocessing pipeline to a raw recording."""
    raw = step_normalise_channels(raw)
    print(f"    Step 1 -- Butterworth bandpass {LOWCUT}-{HIGHCUT} Hz")
    raw = step_filter(raw)
    print(f"    Step 2 -- ASR (cutoff={asr_cutoff})")
    raw = step_asr(raw, asr_cutoff)
    print(f"    Step 3 -- ICA + ICLabel")
    raw = step_ica(raw)
    return raw


# ==============================================================================
#  BIOMARKER EXTRACTION
# ==============================================================================

def _region_indices(ch_names, region_key):
    wanted = set(REGIONS[region_key])
    return [i for i, ch in enumerate(ch_names) if ch in wanted]


def _compute_psd(data, sfreq):
    nperseg = int(sfreq * WELCH_WIN_S)
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd


def _band_power_array(psd, freqs, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.mean(psd[..., idx], axis=-1)


def _spectral_entropy(psd, freqs, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    p   = psd[..., idx]
    p   = p / (p.sum(axis=-1, keepdims=True) + 1e-20)
    return -np.sum(p * np.log2(p + 1e-20), axis=-1)


def compute_spectral(raw):
    data  = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch    = raw.ch_names

    freqs, psd = _compute_psd(data, sfreq)
    bp = {name: _band_power_array(psd, freqs, lo, hi) for name, (lo, hi) in BANDS.items()}

    total = bp["delta"] + bp["theta"] + bp["alpha"] + bp["beta"]
    total = np.where(total < 1e-20, 1e-20, total)

    out = {}
    for name in BANDS:
        out[f"{name}_abs"] = float(bp[name].mean())
        out[f"{name}_rel"] = float((bp[name] / total).mean())

    g = {name: float(arr.mean()) for name, arr in bp.items()}
    out["theta_alpha_ratio"]   = g["theta"]  / (g["alpha"]  + 1e-20)
    out["DTABR"]               = (g["delta"] + g["theta"]) / (g["alpha"] + g["beta"] + 1e-20)
    out["slowing_ratio"]       = (g["delta"] + g["theta"]) / (g["alpha"] + g["beta"] + g["gamma"] + 1e-20)
    out["alpha2_alpha1_ratio"] = g["alpha2"] / (g["alpha1"] + 1e-20)

    out["alpha2_spectral_entropy"] = float(_spectral_entropy(psd, freqs, 10.0, 13.0).mean())

    for region in ("posterior", "occipital", "frontal"):
        idx = _region_indices(ch, region)
        if idx:
            out[f"alpha2_{region}_abs"] = float(bp["alpha2"][idx].mean())
            out[f"alpha2_{region}_rel"] = float((bp["alpha2"][idx] / total[idx]).mean())

    l_idx = _region_indices(ch, "left")
    r_idx = _region_indices(ch, "right")
    if l_idx and r_idx:
        l_a2  = bp["alpha2"][l_idx].mean()
        r_a2  = bp["alpha2"][r_idx].mean()
        denom = r_a2 + l_a2
        out["alpha2_asymmetry"] = float((r_a2 - l_a2) / denom) if denom > 1e-20 else 0.0

    # Peak frequency in 10-13 Hz
    idx_a2   = (freqs >= 10.0) & (freqs <= 13.0)
    peak_bin = psd[:, idx_a2].mean(axis=0).argmax()
    out["alpha2_peak_freq"] = float(freqs[idx_a2][peak_bin])

    # Permutation entropy on spatially-averaged alpha-2 signal
    raw_a2         = raw.copy().filter(10.0, 13.0, method="fir", fir_window="hamming", verbose=False)
    mean_a2_signal = raw_a2.get_data().mean(axis=0)
    out["alpha2_perm_entropy"] = float(
        antropy.perm_entropy(mean_a2_signal, order=3, delay=1, normalize=True)
    )
    del raw_a2

    return out


def _canonical_order(model, ch_names):
    """Map cluster indices 0-3 to canonical A/B/C/D via absolute Pearson correlation + Hungarian."""
    try:
        centers = model.cluster_centers_
        if hasattr(centers, "get_data"):
            centers = centers.get_data()
        n_clust = centers.shape[0]

        ref_vecs = np.zeros((len(MS_LABELS), len(ch_names)))
        for j, letter in enumerate(MS_LABELS):
            for ch, polarity in CANONICAL_MAPS[letter].items():
                if ch in ch_names:
                    ref_vecs[j, ch_names.index(ch)] = polarity

        cost = np.zeros((n_clust, len(MS_LABELS)))
        for i in range(n_clust):
            for j in range(len(MS_LABELS)):
                if ref_vecs[j].any():
                    c = np.corrcoef(centers[i], ref_vecs[j])[0, 1]
                    cost[i, j] = abs(c) if np.isfinite(c) else 0.0

        row_ind, col_ind = linear_sum_assignment(-cost)
        return {int(r): MS_LABELS[c] for r, c in zip(row_ind, col_ind)}

    except Exception as e:
        print(f"    [microstates] canonical ordering failed ({e}), using sequential labels")
        return {k: MS_LABELS[k] for k in range(N_STATES)}


def compute_microstates(raw):
    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    if n_eeg < N_STATES:
        print(f"    [microstates] Only {n_eeg} EEG channels -- skipping.")
        return {f"ms_{l}_{s}": np.nan
                for l in MS_LABELS for s in ("coverage", "mean_dur_ms", "occurrence")}

    raw_ms = raw.copy()
    raw_ms.crop(tmin=0, tmax=min(MS_MAX_S, raw_ms.times[-1]))
    if raw_ms.info["sfreq"] > MS_RESAMPLE_HZ:
        raw_ms.resample(MS_RESAMPLE_HZ, verbose=False)
    raw_ms.set_eeg_reference("average", projection=False, verbose=False)

    model  = ModKMeans(n_clusters=N_STATES, random_state=42, n_init=10)
    model.fit(raw_ms, picks="eeg", verbose=False)
    seg    = model.predict(raw_ms, picks="eeg", verbose=False)
    labels = np.asarray(seg.labels)
    sfreq  = raw_ms.info["sfreq"]
    dt_ms  = 1000.0 / sfreq

    eeg_picks     = mne.pick_types(raw_ms.info, eeg=True)
    ch_names      = [raw_ms.ch_names[i] for i in eeg_picks]
    canon_map     = _canonical_order(model, ch_names)
    letter_to_idx = {letter: idx for idx, letter in enumerate(MS_LABELS)}
    remapped      = np.array([letter_to_idx[canon_map[k]] if k >= 0 else -1 for k in labels])

    n_tot = int((remapped >= 0).sum())
    if n_tot == 0:
        return {f"ms_{l}_{s}": np.nan
                for l in MS_LABELS for s in ("coverage", "mean_dur_ms", "occurrence")}

    out = {}
    for k, letter in enumerate(MS_LABELS):
        mask = remapped == k
        n_k  = int(mask.sum())
        out[f"ms_{letter}_coverage"] = float(n_k / n_tot)

        diff   = np.diff(mask.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [n_tot]])
        runs = ends - starts if len(starts) > 0 else np.array([0])
        out[f"ms_{letter}_mean_dur_ms"] = float(runs.mean() * dt_ms) if len(runs) else 0.0
        out[f"ms_{letter}_occurrence"]  = float(len(runs) / (n_tot / sfreq)) if n_tot else 0.0

    out["ms_GEV"] = float(getattr(model, "GEV_", np.nan))

    valid_pairs = (remapped[:-1] >= 0) & (remapped[1:] >= 0)
    for i, fi in enumerate(MS_LABELS):
        from_mask = (remapped[:-1] == i) & valid_pairs
        n_from    = from_mask.sum()
        for j, tj in enumerate(MS_LABELS):
            if i == j:
                continue
            prob = float((from_mask & (remapped[1:] == j)).sum() / n_from) if n_from else 0.0
            out[f"ms_trans_{fi}{tj}"] = prob

    del raw_ms
    return out


# ==============================================================================
#  PER-SUBJECT PIPELINE
# ==============================================================================

def _read_raw_set(set_path: Path) -> mne.io.BaseRaw:
    """
    Read an EEGLAB .set file. Tries MNE first; falls back to h5py for
    MATLAB v7.3 (HDF5) files that scipy cannot load.
    """
    try:
        return mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    except NotImplementedError:
        pass  # MATLAB v7.3 -- fall through to h5py reader

    # MATLAB v7.3 HDF5 format -- header in .set, raw data in companion .fdt file
    print(f"    [loader] MATLAB v7.3 detected -- using h5py")
    import h5py  # type: ignore

    with h5py.File(str(set_path), "r") as f:
        sfreq  = float(np.array(f["srate"]).flatten()[0])
        nbchan = int(np.array(f["nbchan"]).flatten()[0])
        pnts   = int(np.array(f["pnts"]).flatten()[0])

        # Channel names: each entry is an HDF5 object reference pointing to a char array
        labels_ds = f["chanlocs"]["labels"]
        ch_names  = []
        for i in range(min(int(labels_ds.shape[0]), nbchan)):
            ref   = labels_ds[i, 0]
            chars = f[ref][()].flatten()
            ch_names.append("".join(chr(int(c)) for c in chars))

        fdt_name = "".join(chr(int(c)) for c in np.array(f["datfile"]).flatten())

    # Read binary .fdt: raw float32, shape (n_channels, n_timepoints)
    fdt_path = set_path.parent / fdt_name
    data = (np.fromfile(str(fdt_path), dtype=np.float32)
              .reshape(nbchan, pnts)
              .astype(np.float64))

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def save_preprocessed_set(raw: mne.io.BaseRaw, out_path: Path):
    """Export cleaned Raw to EEGLAB .set format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.export(str(out_path), fmt="eeglab", overwrite=True, verbose=False)
    print(f"    Saved preprocessed .set -> {out_path.name}")


def process_subject(sub_id, set_path, meta_row, condition, asr_cutoff, save_set: bool = False):
    """Load raw EEG -> preprocess -> extract biomarkers -> return result row."""
    print(f"    Loading: {set_path.name}")
    raw = _read_raw_set(set_path)
    print(f"    Channels: {len(raw.ch_names)}  |  "
          f"Duration: {raw.times[-1]/60:.1f} min  |  sfreq: {raw.info['sfreq']} Hz")

    raw  = preprocess(raw, asr_cutoff)

    if save_set:
        set_out = OUT_DIR / "preprocessed" / condition / sub_id / set_path.name
        save_preprocessed_set(raw, set_out)

    spec = compute_spectral(raw)
    ms   = compute_microstates(raw)
    del raw

    row = {
        "participant_id": sub_id,
        "condition":      condition,
        "group":          meta_row.get("Group", "?"),
        "age":            meta_row.get("Age",   np.nan),
        "gender":         meta_row.get("Gender","?"),
        "mmse":           meta_row.get("MMSE",  np.nan),
    }
    row.update(spec)
    row.update(ms)
    return row


# ==============================================================================
#  DATASET-LEVEL PROCESSOR
# ==============================================================================

def _load_meta(tsv_path: Path, group_col: str = "Group",
               group_map: dict = None) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = df.columns.str.strip()
    if group_col != "Group" and group_col in df.columns:
        df = df.rename(columns={group_col: "Group"})
    if group_map:
        df["Group"] = df["Group"].map(group_map)   # unmapped values become NaN
        n_before = len(df)
        df = df[df["Group"].notna()]               # drop excluded groups (None -> NaN)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"  [meta] {n_dropped} subject(s) excluded by group_map (MCI / missing label)")
    return df.set_index("participant_id")


def process_dataset(condition: str, subject_filter: str = None, save_set: bool = False) -> pd.DataFrame:
    cfg        = DATASETS[condition]
    raw_dir    = Path(cfg["raw_dir"])
    suffix     = cfg["file_suffix"]
    asr_cutoff = cfg["asr_cutoff"]
    meta_file  = Path(cfg["meta_file"])
    group_col  = cfg.get("group_col", "Group")

    if not raw_dir.exists():
        print(f"\n  [{condition}] Raw directory not found: {raw_dir} -- skipped")
        return pd.DataFrame()
    if not meta_file.exists():
        print(f"\n  [{condition}] participants.tsv not found: {meta_file} -- skipped")
        return pd.DataFrame()

    group_map = cfg.get("group_map", None)
    meta = _load_meta(meta_file, group_col=group_col, group_map=group_map)

    subjects = sorted(d for d in os.listdir(raw_dir)
                      if d.startswith("sub-") and (raw_dir / d).is_dir()
                      and d in meta.index)   # skip subjects excluded by group_map
    if subject_filter:
        subjects = [s for s in subjects if s == subject_filter]
        if not subjects:
            print(f"  [WARNING] Subject '{subject_filter}' not found in {raw_dir}")
            return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  Condition : {condition}  |  Dataset: {raw_dir.name}")
    print(f"  Subjects  : {len(subjects)}  |  ASR cutoff: {asr_cutoff}")
    print(f"{'='*60}")

    rows, failed = [], []

    for i, sub in enumerate(subjects, 1):
        eeg_dir   = raw_dir / sub / "eeg"
        if not eeg_dir.exists():
            print(f"  [{i:02d}/{len(subjects):02d}] {sub} -- no eeg/ folder, skipped")
            failed.append(sub)
            continue

        set_files = [f for f in os.listdir(eeg_dir)
                     if f.endswith(".set") and suffix in f]
        if not set_files:
            print(f"  [{i:02d}/{len(subjects):02d}] {sub} -- no matching .set, skipped")
            failed.append(sub)
            continue

        set_path = eeg_dir / set_files[0]
        meta_row = meta.loc[sub].to_dict() if sub in meta.index else {}

        print(f"\n  [{i:02d}/{len(subjects):02d}] {sub}  (group={meta_row.get('Group','?')})")
        try:
            row = process_subject(sub, set_path, meta_row, condition, asr_cutoff, save_set=save_set)
            rows.append(row)

            dtab = row.get("DTABR", float("nan"))
            ta   = row.get("theta_alpha_ratio", float("nan"))
            a2   = row.get("alpha2_abs", float("nan"))
            msC  = row.get("ms_C_mean_dur_ms", float("nan"))
            print(f"    -> theta/alpha={ta:.3f}  DTABR={dtab:.3f}  alpha2={a2:.2e}  msC={msC:.1f}ms")

        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            failed.append(sub)

    print(f"\n  Done -- {len(rows)} succeeded, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")

    return pd.DataFrame(rows)


# ==============================================================================
#  SUMMARY BUILDERS
# ==============================================================================

GROUP_LABEL = {"A": "AD", "C": "CN", "F": "FTD"}
GROUP_ORDER = ["AD", "FTD", "CN"]


def subject_average(df_closed: pd.DataFrame, df_open: pd.DataFrame) -> pd.DataFrame:
    """
    Average biomarkers per subject across eyes-closed and eyes-open conditions,
    so each of the 88 subjects contributes exactly one row to the combined output.
    Subjects that appear in only one condition are still included (no data dropped).
    """
    frames = [f for f in [df_closed, df_open] if not f.empty]
    if not frames:
        return pd.DataFrame()

    combined  = pd.concat(frames, ignore_index=True)
    meta_cols = ["participant_id", "group", "age", "gender", "mmse"]
    bio_cols  = [c for c in combined.select_dtypes(include=np.number).columns
                 if c not in {"age", "mmse"}]

    averaged = combined.groupby("participant_id")[bio_cols].mean().reset_index()

    meta = (combined[meta_cols]
            .drop_duplicates(subset="participant_id")
            .reset_index(drop=True))

    result = meta.merge(averaged, on="participant_id")
    result["condition"] = "combined"
    return result


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["group_label"] = df["group"].map(GROUP_LABEL).fillna(df["group"])
    group_cols = ["group_label", "condition"] if "condition" in df.columns else ["group_label"]
    bio_cols   = [c for c in df.select_dtypes(include=np.number).columns
                  if c not in {"age", "mmse"}]
    agg = df.groupby(group_cols)[bio_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg["group_label"] = pd.Categorical(agg["group_label"], categories=GROUP_ORDER, ordered=True)
    return agg.sort_values("group_label").reset_index(drop=True)


def make_key_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["group_label"] = df["group"].map(GROUP_LABEL).fillna(df["group"])

    KEY = {
        "Theta/Alpha Ratio"     : "theta_alpha_ratio",
        "DTABR"                 : "DTABR",
        "Alpha2 Abs Power (V2)" : "alpha2_abs",
        "Alpha2 Rel Power"      : "alpha2_rel",
        "Alpha2 Spec Entropy"   : "alpha2_spectral_entropy",
        "MS-A Duration (ms)"    : "ms_A_mean_dur_ms",
        "MS-A Coverage"         : "ms_A_coverage",
        "MS-B Duration (ms)"    : "ms_B_mean_dur_ms",
        "MS-B Coverage"         : "ms_B_coverage",
        "MS-C Duration (ms)"    : "ms_C_mean_dur_ms",
        "MS-C Coverage"         : "ms_C_coverage",
        "MS-D Duration (ms)"    : "ms_D_mean_dur_ms",
        "MS-D Coverage"         : "ms_D_coverage",
    }

    rows = []
    for grp in GROUP_ORDER:
        sub = df[df["group_label"] == grp]
        if sub.empty:
            continue
        row = {"Group": grp, "N": len(sub), "Condition": label}
        for display, col in KEY.items():
            if col in sub.columns:
                m, s = sub[col].mean(), sub[col].std()
                row[display] = f"{m:.4e} +/- {s:.4e}" if abs(m) < 1e-6 else f"{m:.4f} +/- {s:.4f}"
            else:
                row[display] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def print_group_means(df: pd.DataFrame, label: str):
    KEY_COLS = ["DTABR", "theta_alpha_ratio", "alpha2_abs", "alpha2_rel",
                "alpha2_spectral_entropy",
                "ms_A_coverage", "ms_B_coverage", "ms_C_coverage", "ms_D_coverage"]
    avail = [c for c in KEY_COLS if c in df.columns]
    if not avail:
        return
    df2 = df.copy()
    df2["group_label"] = df2["group"].map(GROUP_LABEL).fillna(df2["group"])
    counts  = df2.groupby("group_label", dropna=False).size().to_dict()
    summary = df2.groupby("group_label")[avail].mean().reindex(GROUP_ORDER)
    print(f"\n-- {label} group means  (n={counts}) --")
    print(summary.to_string(float_format="{:.4e}".format, na_rep="n/a"))


# ==============================================================================
#  MAIN
# ==============================================================================

def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    """Load a saved biomarkers CSV if it exists, otherwise return empty DataFrame."""
    if path.exists():
        print(f"  [load] Using saved results: {path.name}")
        return pd.read_csv(path)
    return pd.DataFrame()


def process_single_file(set_path: Path, asr_cutoff: int = 15, save_set: bool = False) -> pd.DataFrame:
    """Run the full pipeline on a single .set file with no BIDS structure needed."""
    print(f"\n{'='*60}")
    print(f"  Single-file mode: {set_path.name}")
    print(f"  ASR cutoff: {asr_cutoff}")
    print(f"{'='*60}")

    raw  = _read_raw_set(set_path)
    print(f"    Channels: {len(raw.ch_names)}  |  "
          f"Duration: {raw.times[-1]/60:.1f} min  |  sfreq: {raw.info['sfreq']} Hz")

    raw  = preprocess(raw, asr_cutoff)

    if save_set:
        set_out = OUT_DIR / "preprocessed" / "single" / f"{set_path.stem}_preprocessed.set"
        save_preprocessed_set(raw, set_out)

    spec = compute_spectral(raw)
    ms   = compute_microstates(raw)
    del raw

    row = {
        "participant_id": set_path.stem,
        "condition":      "unknown",
        "group":          "unknown",
        "age":            float("nan"),
        "gender":         "unknown",
        "mmse":           float("nan"),
    }
    row.update(spec)
    row.update(ms)

    df = pd.DataFrame([row])

    out_path = OUT_DIR / f"{set_path.stem}_biomarkers.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    # Print key biomarkers to console
    dtab = row.get("DTABR", float("nan"))
    ta   = row.get("theta_alpha_ratio", float("nan"))
    a2   = row.get("alpha2_abs", float("nan"))
    msC  = row.get("ms_C_mean_dur_ms", float("nan"))
    print(f"\n  Key biomarkers:")
    print(f"    theta/alpha ratio : {ta:.4f}")
    print(f"    DTABR             : {dtab:.4f}")
    print(f"    alpha2 abs power  : {a2:.4e}")
    print(f"    MS-C duration(ms) : {msC:.2f}")
    print(f"    MS-A coverage     : {row.get('ms_A_coverage', float('nan')):.4f}")
    print(f"    MS-B coverage     : {row.get('ms_B_coverage', float('nan')):.4f}")
    print(f"    MS-C coverage     : {row.get('ms_C_coverage', float('nan')):.4f}")
    print(f"    MS-D coverage     : {row.get('ms_D_coverage', float('nan')):.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw EEG and extract biomarkers in one pass (HPC version)."
    )
    parser.add_argument(
        "--file", default=None,
        help="Path to a single raw .set file. Use this for unknown/hackathon EEG data."
    )
    parser.add_argument(
        "--folder", default=None,
        help="Path to a folder containing .set files. Processes all .set files found "
             "and saves a separate CSV per file named <filename>_biomarkers.csv."
    )
    parser.add_argument(
        "--dataset",
        choices=["eyes_closed", "eyes_open", "auditory", "both", "all"],
        default="both",
        help=(
            "Dataset(s) to process fresh. "
            "'both'=eyes_closed+eyes_open, "
            "'all'=all three. "
            "'auditory' processes only the auditory dataset and loads saved "
            "eyes_closed/eyes_open CSVs for the combined output. "
            "(default: both)"
        )
    )
    parser.add_argument(
        "--subject", default=None,
        help="Process only this subject (e.g. sub-001). Omit for all subjects."
    )
    parser.add_argument(
        "--save_set", action="store_true", default=False,
        help="Save preprocessed .set files in addition to CSVs. "
             "Saved to biomarker_results/preprocessed/<condition>/<subject>/."
    )
    args = parser.parse_args()

    # -- Single-file mode ------------------------------------------------------
    if args.file:
        set_path = Path(args.file)
        if not set_path.exists():
            print(f"ERROR: File not found: {set_path}")
            return
        process_single_file(set_path, save_set=args.save_set)
        return

    # -- Folder mode -----------------------------------------------------------
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"ERROR: Folder not found: {folder}")
            return
        set_files = sorted(folder.glob("*.set"))
        if not set_files:
            print(f"ERROR: No .set files found in {folder}")
            return
        print(f"\n  Folder mode: {folder}")
        print(f"  Found {len(set_files)} .set file(s)")
        for i, set_path in enumerate(set_files, 1):
            print(f"\n  [{i:02d}/{len(set_files):02d}] {set_path.name}")
            try:
                process_single_file(set_path, save_set=args.save_set)
            except Exception as exc:
                import traceback
                print(f"    ERROR: {exc}")
                traceback.print_exc()
        return

    if args.dataset == "both":
        to_process = ["eyes_closed", "eyes_open"]
    elif args.dataset == "all":
        to_process = ["eyes_closed", "eyes_open", "auditory"]
    else:
        to_process = [args.dataset]

    # -- Run preprocessing + extraction for selected datasets ------------------
    results = {}
    for cond in to_process:
        results[cond] = process_dataset(cond, subject_filter=args.subject, save_set=args.save_set)

    # -- Assemble DataFrames -- load saved CSVs when not freshly processed -----
    def _get(cond, csv_name):
        if cond in results:
            return results[cond]
        return _load_csv_if_exists(OUT_DIR / csv_name)

    df_closed = _get("eyes_closed", "eyes_closed_biomarkers.csv")
    df_open   = _get("eyes_open",   "eyes_open_biomarkers.csv")
    df_aud    = _get("auditory",    "auditory_biomarkers.csv")

    # combined = per-subject mean across eyes_closed + eyes_open
    df_combined = subject_average(df_closed, df_open)

    # -- Save per-condition CSVs (only for freshly processed datasets) ---------
    if "eyes_closed" in results and not df_closed.empty:
        df_closed.to_csv(OUT_DIR / "eyes_closed_biomarkers.csv", index=False)
        make_summary(df_closed).to_csv(OUT_DIR / "eyes_closed_summary.csv", index=False)

    if "eyes_open" in results and not df_open.empty:
        df_open.to_csv(OUT_DIR / "eyes_open_biomarkers.csv", index=False)
        make_summary(df_open).to_csv(OUT_DIR / "eyes_open_summary.csv", index=False)

    if "auditory" in results and not df_aud.empty:
        df_aud.to_csv(OUT_DIR / "auditory_biomarkers.csv", index=False)
        make_summary(df_aud).to_csv(OUT_DIR / "auditory_summary.csv", index=False)

    # -- Save combined CSV -- always rebuilt (subject-averaged, unbiased) ------
    if not df_combined.empty:
        df_combined.to_csv(OUT_DIR / "combined_biomarkers.csv", index=False)
        make_summary(df_combined).to_csv(OUT_DIR / "combined_summary.csv", index=False)

    # -- Key metrics (human-readable) ------------------------------------------
    km_parts = []
    if not df_closed.empty:
        km_parts.append(make_key_metrics(df_closed,  "eyes_closed"))
    if not df_open.empty:
        km_parts.append(make_key_metrics(df_open,    "eyes_open"))
    if not df_combined.empty:
        km_parts.append(make_key_metrics(df_combined, "combined"))
    if not df_aud.empty:
        km_parts.append(make_key_metrics(df_aud,     "auditory"))
    if km_parts:
        pd.concat(km_parts, ignore_index=True).to_csv(
            OUT_DIR / "key_metrics_summary.csv", index=False)

    # -- Console summary -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}")
    for cond, df in results.items():
        print(f"  {cond:15s} -- {len(df)} subjects processed fresh")
    if not df_combined.empty:
        print(f"  combined        -- {len(df_combined)} subjects (per-subject avg of eyes_closed + eyes_open)")

    for label, df in [
        ("Eyes-Closed",            df_closed),
        ("Eyes-Open",              df_open),
        ("Combined (subject avg)", df_combined),
        ("Auditory (ds005048)",    df_aud),
    ]:
        if not df.empty:
            print_group_means(df, label)


if __name__ == "__main__":
    main()
