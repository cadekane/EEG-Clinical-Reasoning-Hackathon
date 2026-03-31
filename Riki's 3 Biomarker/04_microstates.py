"""
04_microstates.py
──────────────────
Biomarker 3 – EEG Microstate Analysis (Class C & D Duration)

Pipeline:
  1. Band-pass filter 1–40 Hz
  2. Compute Global Field Power (GFP)
  3. Extract EEG topographies at GFP local maxima
  4. Polarity-invariant modified K-means clustering (k=4)
  5. Back-fit microstate labels to every time-point
  6. Match clusters to canonical A/B/C/D maps
  7. Compute statistics: mean duration, coverage, occurrence rate

Expected in AD:  Class C and D mean duration INCREASE

Canonical map references (19 ch, 10-20 system):
  A – left temporal / right frontal polarity gradient
  B – right temporal / left frontal polarity gradient
  C – anterior positive / posterior negative
  D – central-parietal positive / frontal negative

NOTE: Canonical map values below are approximations.
      Validate against published templates (e.g. Murray et al. 2008,
      Koenig et al. 2002) for your specific channel layout.

Input:  {BASE_DIR}/derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
Output: {BASE_DIR}/results/microstates.csv
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
DATA_DIR         = os.path.join(BASE_DIR, "derivatives")
PARTICIPANTS_FILE = os.path.join(BASE_DIR, "participants.tsv")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_STATES    = 4     # number of microstate classes
N_INITS     = 10    # random initialisations for K-means
MAX_ITER    = 500   # iterations per run
FILTER_LO   = 1.0   # Hz  (re-filter for microstate analysis)
FILTER_HI   = 40.0  # Hz

# Standard 19-channel order used for canonical map matching
CH_ORDER_19 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
               "O1",  "O2",  "F7", "F8", "T3", "T4", "T5", "T6",
               "Fz",  "Cz", "Pz"]

# Approximate canonical microstate maps (unit-normalised, same channel order)
# Row 0 = A, Row 1 = B, Row 2 = C, Row 3 = D
# Values derived from Michel & Koenig (2018) NIMG review, Fig 1.
_CMAP_RAW = np.array([
    # A: left-posterior / right-anterior
    [-0.10,  0.30, -0.30,  0.50, -0.20,  0.40, -0.60,  0.70,
     -0.30,  0.20, -0.50,  0.60, -0.30,  0.40, -0.80,  0.90,
      0.10,  0.10,  0.00],
    # B: right-posterior / left-anterior
    [ 0.30, -0.10,  0.50, -0.30,  0.40, -0.20,  0.70, -0.60,
      0.20, -0.30,  0.60, -0.50,  0.40, -0.30,  0.90, -0.80,
     -0.10,  0.10,  0.00],
    # C: frontal positive / occipital negative
    [ 0.80,  0.80,  0.50,  0.50,  0.10,  0.10, -0.30, -0.30,
     -0.80, -0.80,  0.60,  0.60, -0.10, -0.10, -0.50, -0.50,
      0.70,  0.20, -0.20],
    # D: central-parietal positive / frontal negative
    [-0.40, -0.40,  0.10,  0.10,  0.70,  0.70,  0.60,  0.60,
      0.20,  0.20, -0.20, -0.20,  0.40,  0.40,  0.40,  0.40,
     -0.10,  0.80,  0.70],
], dtype=np.float64)

CANONICAL_MAPS = _CMAP_RAW / np.linalg.norm(_CMAP_RAW, axis=1, keepdims=True)
# ──────────────────────────────────────────────────────────────────────────────


# ─── Modified K-means (polarity-invariant) ───────────────────────────────────

def _normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def modified_kmeans(data: np.ndarray, n_states: int = 4,
                    n_inits: int = 10, max_iter: int = 500):
    """
    Polarity-invariant modified K-means for EEG microstates.

    Parameters
    ----------
    data : (n_samples, n_channels)  – topographies at GFP peaks
    Returns
    -------
    maps : (n_states, n_channels)   – cluster centres (unit-normalised)
    gev  : float                    – global explained variance
    """
    data_n  = _normalize(data)
    gfp_sq  = np.sum(data ** 2, axis=1)
    n_samp  = len(data_n)

    best_gev  = -np.inf
    best_maps = None

    rng = np.random.default_rng(42)

    for _ in range(n_inits):
        idx  = rng.choice(n_samp, n_states, replace=False)
        maps = data_n[idx].copy()

        prev_labels = None
        for _ in range(max_iter):
            # Assignment: polarity-invariant (max |correlation|)
            corr   = np.abs(data_n @ maps.T)       # (n_samp, n_states)
            labels = np.argmax(corr, axis=1)

            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels.copy()

            # Update centres
            for k in range(n_states):
                mask = labels == k
                if mask.sum() == 0:
                    maps[k] = data_n[rng.integers(n_samp)]
                    continue
                subset  = data_n[mask]
                # Flip signs so all point in the same hemisphere as current map
                signs   = np.sign(subset @ maps[k])
                signs[signs == 0] = 1
                aligned = subset * signs[:, None]
                m       = aligned.mean(axis=0)
                maps[k] = m / (np.linalg.norm(m) + 1e-12)

        # GEV for this run
        corr     = data_n @ maps.T
        labels   = np.argmax(np.abs(corr), axis=1)
        corr_sq  = corr[np.arange(n_samp), labels] ** 2
        gev      = float(np.sum(gfp_sq * corr_sq) / (np.sum(gfp_sq) + 1e-12))

        if gev > best_gev:
            best_gev  = gev
            best_maps = maps.copy()

    return best_maps, best_gev


def backfit(data: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Assign every time-point to the nearest microstate (polarity-invariant)."""
    data_n = _normalize(data)
    corr   = np.abs(data_n @ maps.T)
    return np.argmax(corr, axis=1)


# ─── Canonical label matching ────────────────────────────────────────────────

def match_canonical(maps: np.ndarray, ch_names: list) -> dict:
    """
    Match k computed maps to canonical A/B/C/D using polarity-invariant
    correlation.  Returns {cluster_idx: 'A'/'B'/'C'/'D'}.

    Falls back to sorted labelling (by GEV fraction) when channel overlap
    with the canonical layout is < 10.
    """
    available = [ch for ch in CH_ORDER_19 if ch in ch_names]
    if len(available) < 10:
        print(f"  [WARN] Only {len(available)} channels match canonical layout; "
              "using index-based labels 1-4 instead.")
        return {k: str(k + 1) for k in range(N_STATES)}

    can_idx = [CH_ORDER_19.index(ch) for ch in available]
    cmp_idx = [ch_names.index(ch)    for ch in available]

    can_sub = CANONICAL_MAPS[:, can_idx]
    cmp_sub = maps[:, cmp_idx]

    can_n = can_sub / (np.linalg.norm(can_sub, axis=1, keepdims=True) + 1e-12)
    cmp_n = cmp_sub / (np.linalg.norm(cmp_sub, axis=1, keepdims=True) + 1e-12)

    sim = np.abs(cmp_n @ can_n.T)   # (n_states, 4)

    assignment   = {}
    used_canon   = set()
    label_names  = ["A", "B", "C", "D"]

    # Greedy assignment by descending similarity
    flat_order = np.argsort(sim.ravel())[::-1]
    for flat_idx in flat_order:
        cmp_i, can_j = divmod(int(flat_idx), N_STATES)
        if cmp_i not in assignment and can_j not in used_canon:
            assignment[cmp_i] = label_names[can_j]
            used_canon.add(can_j)
        if len(assignment) == N_STATES:
            break

    return assignment


# ─── Statistics ──────────────────────────────────────────────────────────────

def microstate_statistics(labels: np.ndarray, sfreq: float, n_states: int = 4) -> dict:
    """
    Per-class statistics:
      coverage         – fraction of total recording time
      mean_duration_ms – mean segment length in ms
      occurrence_per_s – number of segments per second
    """
    n_total = len(labels)
    stats   = {}

    for k in range(n_states):
        mask      = labels == k
        coverage  = float(mask.sum() / n_total)

        # Contiguous run lengths
        durations_ms = []
        run_len      = 0
        in_run       = False
        for m in mask:
            if m:
                in_run   = True
                run_len += 1
            elif in_run:
                durations_ms.append(run_len / sfreq * 1000)
                run_len = 0
                in_run  = False
        if in_run:
            durations_ms.append(run_len / sfreq * 1000)

        total_sec = n_total / sfreq
        stats[k]  = {
            "coverage":           coverage,
            "mean_duration_ms":   float(np.mean(durations_ms))   if durations_ms else 0.0,
            "median_duration_ms": float(np.median(durations_ms)) if durations_ms else 0.0,
            "occurrence_per_s":   len(durations_ms) / total_sec  if total_sec > 0 else 0.0,
        }

    return stats


# ─── Per-subject pipeline ────────────────────────────────────────────────────

def compute_gfp(data: np.ndarray) -> np.ndarray:
    """GFP = std across channels at each sample. data: (n_ch, n_times)."""
    return np.std(data, axis=0)   # (n_times,)


def gfp_peaks(gfp: np.ndarray) -> np.ndarray:
    """Indices of local maxima in GFP trace."""
    g = gfp
    peaks = np.where((g[1:-1] > g[:-2]) & (g[1:-1] > g[2:]))[0] + 1
    return peaks


def process_subject(raw: mne.io.BaseRaw) -> dict:
    sfreq    = raw.info["sfreq"]
    ch_names = raw.copy().pick("eeg").ch_names
    data     = raw.get_data(picks="eeg")       # (n_ch, n_times)

    gfp   = compute_gfp(data)
    peaks = gfp_peaks(gfp)

    if len(peaks) < N_STATES * 10:
        raise ValueError(f"Too few GFP peaks ({len(peaks)}); recording may be too short.")

    peak_topo = data[:, peaks].T               # (n_peaks, n_ch)

    maps, gev = modified_kmeans(peak_topo, n_states=N_STATES,
                                n_inits=N_INITS, max_iter=MAX_ITER)

    label_map = match_canonical(maps, ch_names)

    raw_labels = backfit(data.T, maps)         # (n_times,)
    stats      = microstate_statistics(raw_labels, sfreq, n_states=N_STATES)

    results = {"gev": gev, "n_gfp_peaks": len(peaks)}
    for k in range(N_STATES):
        canon = label_map[k]
        for metric, value in stats[k].items():
            results[f"ms{canon}_{metric}"] = value

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

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
            raw = mne.io.read_raw_eeglab(in_path, preload=True)

            # Band-pass for microstate analysis
            raw.filter(
                l_freq=FILTER_LO, h_freq=FILTER_HI,
                method="iir",
                iir_params=dict(order=4, ftype="butter"),
                picks="eeg",
            )

            features = process_subject(raw)

            entry = {"subject": sub_id, "group": group, "age": age, "mmse": mmse}
            entry.update(features)
            rows.append(entry)

            c_dur = features.get("msC_mean_duration_ms", float("nan"))
            d_dur = features.get("msD_mean_duration_ms", float("nan"))
            print(f"  GEV={features['gev']:.3f}  "
                  f"C_dur={c_dur:.1f} ms  D_dur={d_dur:.1f} ms")

        except Exception as exc:
            print(f"[ERROR] {sub_id}: {exc}")

    if not rows:
        print("No results collected. Check DATA_DIR path.")
        return

    df       = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "microstates.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # ── Group summary ─────────────────────────────────────────────────────────
    print("\n=== Group Summary (Class C & D Duration — AD biomarker) ===")
    group_labels = [("A", "Alzheimer"), ("F", "FTD"), ("C", "Healthy")]
    for g, label in group_labels:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        print(f"\n{label} (n={len(sub)}):")
        for cls in ["A", "B", "C", "D"]:
            col = f"ms{cls}_mean_duration_ms"
            if col in df.columns:
                print(f"  Class {cls} duration: {sub[col].mean():.2f} ± {sub[col].std():.2f} ms  "
                      f"| coverage: {sub[f'ms{cls}_coverage'].mean():.3f}")


if __name__ == "__main__":
    main()
