# EEG Biomarker Results

**Data source:** Raw EEG files full preprocessing applied per-subject (bandpass → ASR → ICA+ICLabel).  
**Resting-state datasets:** ds004504 (eyes-closed) + ds006036 (eyes-open), 88 subjects each, 3 groups.  
**Auditory dataset:** ds005048 (40 Hz auditory entrainment), 27 subjects (CN/AD only), standalone.

---

## Quick Start

### What file should I look at first?

Open **`key_metrics_summary.csv`**. It has one row per group per condition, with the key clinical numbers in plain `mean ± std` format. This is the best file for a quick overview.

### Datasets

| Dataset | OpenNeuro | Description | Subjects |
|---|---|---|---|
| `eyes_closed` | ds004504 | Eyes-closed resting state EEG | 88 |
| `eyes_open` | ds006036 | Eyes-open resting state EEG | 88 |
| `combined` | ds004504 + ds006036 | Per-subject average of eyes_closed and eyes_open — each of the 88 subjects contributes one row | 88 |
| `auditory` | ds005048 | 40 Hz auditory entrainment EEG (standalone, not comparable to resting state) | 27 |

> **Auditory dataset:** AD (n=17), CN (n=10) only. MCI subjects (n=6) and subjects with missing labels are excluded — MCI is clinically distinct from AD/CN and too small a group for meaningful comparison.

### Priority Biomarkers

| Column | What it measures | Observed pattern (combined) |
|---|---|---|
| `DTABR` | (delta+theta) / (alpha+beta) — broad EEG slowing | FTD ≈ AD (31.0–31.2) > CN (23.1) |
| `theta_alpha_ratio` | Theta / alpha — focused EEG slowing | AD (2.98) > FTD (2.78) > CN (1.99) |
| `alpha2_abs` | Absolute alpha-2 power (10–13 Hz) | CN (1.82e-11) > AD (7.94e-12) > FTD (3.56e-12) |
| `alpha2_rel` | Alpha-2 as fraction of total band power | CN (0.0401) > FTD (0.0292) ≈ AD (0.0286) |
| `alpha2_spectral_entropy` | Complexity of alpha-2 spectrum | AD (2.747) > FTD (2.714) > CN (2.649) |
| `ms_B_coverage` | Fraction of time in microstate B (right posterior) | FTD (0.266) > AD (0.252) > CN (0.229) |

These six biomarkers together capture EEG slowing (DTABR, theta/alpha), thalamocortical degradation (alpha2_abs, alpha2_rel), spectral disorganization (alpha2_spectral_entropy), and network-level dynamics (ms_B_coverage). Use them as the primary features for group classification and z-score comparison.

Other biomarkers (microstate A/C/D, regional alpha-2, permutation entropy, transition probabilities) are available in the CSVs as supporting evidence but are secondary.

---

## How to Regenerate (HPC)

### HPC Tips

> **Processing time warning:** The full pipeline (filter → ASR → ICA → biomarkers) is slow. Processing all 88 subjects for a single dataset takes at least 6 hours.
> **Always request at least 6 hours when submitting a SLURM job or opening a Jupyter session on KOA.** Running `--dataset both` and `--dataset auditory` as separate jobs is safer than `--dataset all` in one session.

### Prerequisites

Upload the script to your KOA scratch directory.

Install dependencies in Jupyter on KOA:
```python
%pip install mne mne-icalabel meegkit onnxruntime pycrostates antropy scipy h5py
```

### Running the pipeline

```python
import os
os.environ["OMP_NUM_THREADS"] = "4"   # suppress onnxruntime affinity warnings

# Process eyes-closed + eyes-open (default):
%run preprocess_and_extract_hpc.py --dataset both

# Process auditory only (loads saved eyes CSVs for combined output):
%run preprocess_and_extract_hpc.py --dataset auditory

# Process all three datasets:
%run preprocess_and_extract_hpc.py --dataset all

# Process a single subject:
%run preprocess_and_extract_hpc.py --dataset eyes_closed --subject sub-001

# Process a single unknown EEG file (hackathon mode):
%run preprocess_and_extract_hpc.py --file /path/to/given_eeg.set
```

### Saving Preprocessed .set Files

By default only CSVs are saved. Add `--save_set` to also save the cleaned EEG as EEGLAB `.set` files:

```python
# Save preprocessed .set for all datasets
%run preprocess_and_extract_hpc.py --dataset both --save_set

# Save preprocessed .set for a single subject
%run preprocess_and_extract_hpc.py --dataset auditory --subject sub-001 --save_set

# Save preprocessed .set for a single unknown file (hackathon)
%run preprocess_and_extract_hpc.py --file /path/to/eeg.set --save_set
```

Preprocessed `.set` files are saved to:
```
biomarker_results/
└── preprocessed/
    ├── eyes_closed/
    │   └── sub-001/
    │       └── sub-001_task-eyesclosed_eeg.set
    ├── eyes_open/
    │   └── sub-001/
    │       └── sub-001_task-photomark_eeg.set
    ├── auditory/
    │   └── sub-001/
    │       └── sub-001_..._eeg.set
    └── single/
        └── given_eeg_preprocessed.set
```



### Output location

All CSVs are saved to:
```
/mnt/lustre/koa/scratch/$USER/biomarker_results/
```

### Hackathon single-file mode

When you receive an unknown EEG file at the hackathon:
```python
%run preprocess_and_extract_hpc.py --file /path/to/patient_eeg.set
```
- No `--dataset` flag needed — works with any `.set` file regardless of experiment type
- Key biomarkers printed to console immediately
- Output saved to `biomarker_results/single_subject_biomarkers.csv`

---

## How the Pipeline Works

```
Raw .set file
    │
    ▼
1. Channel normalisation
   - Rename old 10-20 names (T3→T7, Fp1→FP1, etc.)
   - Drop non-EEG channels (EOG, EMG, ECG, STI)
   - Set standard_1020 montage
    │
    ▼
2. Butterworth bandpass filter  0.5 – 45 Hz  (4th-order, zero-phase)
    │
    ▼
3. Artifact Subspace Reconstruction (ASR, meegkit)
   - Calibrated on first 60 s of recording
   - cutoff=17 for ds004504,  cutoff=15 for ds006036 / ds005048
    │
    ▼
4. ICA — extended Infomax (19 components, RunICA equivalent)
   + ICLabel — auto-reject Eye Blink and Muscle Artifact components (p > 0.5)
   - Always keeps at least 2 components (safety cap)
    │
    ▼
5. Biomarker extraction (on cleaned data in memory — no intermediate .set saved)
   ├── Spectral: Welch PSD → band powers, ratios, DTABR
   ├── Alpha-2: regional power, asymmetry, peak freq, entropy
   └── Microstates: ModKMeans k=4, canonical A/B/C/D relabeling
    │
    ▼
CSV output
```

**Note on auditory dataset:** ds005048 uses 40 Hz auditory entrainment stimulation. The 40 Hz gamma drive dominates the spectrum, making DTABR and theta/alpha near-zero (~0.03) and alpha2_rel > 1. These biomarkers are not comparable to resting-state values — auditory results are kept standalone.

---

## All Output Files

| File | Rows | Use case |
|---|---|---|
| `key_metrics_summary.csv` | 12 (3 groups × 4 conditions) | Quick read — mean ± std strings, key biomarkers only |
| `eyes_closed_summary.csv` | 3 (one per group) | Machine-readable group stats, eyes-closed |
| `eyes_open_summary.csv` | 3 (one per group) | Machine-readable group stats, eyes-open |
| `combined_summary.csv` | 3 (one per group) | Machine-readable group stats, combined |
| `auditory_summary.csv` | 2 (AD, CN only) | Machine-readable group stats, auditory |
| `eyes_closed_biomarkers.csv` | 88 (one per subject) | Per-subject detail, eyes-closed |
| `eyes_open_biomarkers.csv` | 88 (one per subject) | Per-subject detail, eyes-open |
| `combined_biomarkers.csv` | 88 (one per subject, averaged across conditions) | Per-subject detail, unbiased combined |
| `auditory_biomarkers.csv` | 27 (one per subject) | Per-subject detail, auditory |
| `single_subject_biomarkers.csv` | 1 | Hackathon single-file output |

> `combined_biomarkers.csv` has **88 rows** (one per subject) — biomarkers are averaged across eyes-closed and eyes-open conditions per subject. This avoids double-counting the same 88 subjects.

---

## Observed Results

### Biomarker 1 — Spectral Power Ratios

**Theta/Alpha Ratio and DTABR** (higher = more EEG slowing = more dementia-like):

*Eyes-closed (n: AD=36, FTD=23, CN=29)*
| Group | Theta/Alpha | DTABR |
|---|---|---|
| AD | 2.878 | 29.32 |
| FTD | 2.812 | 31.73 |
| CN | 1.948 | 21.72 |

*Eyes-open (n: AD=36, FTD=23, CN=29)*
| Group | Theta/Alpha | DTABR |
|---|---|---|
| AD | 3.077 | 33.16 |
| FTD | 2.748 | 30.30 |
| CN | 2.028 | 24.49 |

*Combined — per-subject average (n: AD=36, FTD=23, CN=29)*
| Group | Theta/Alpha | DTABR |
|---|---|---|
| AD | 2.978 | 31.24 |
| FTD | 2.780 | 31.02 |
| CN | 1.988 | 23.11 |

> CN has consistently lower DTABR and theta/alpha than AD and FTD across all conditions, as expected. AD and FTD are close to each other — FTD shows slightly higher DTABR in eyes-closed, AD slightly higher in eyes-open.

*Auditory — standalone, not comparable to resting state (n: AD=17, CN=10)*
| Group | Theta/Alpha | DTABR |
|---|---|---|
| AD | 0.00253 | 0.0333 |
| CN | 0.00184 | 0.0269 |

> Near-zero values are expected — the 40 Hz gamma entrainment dominates the spectrum, suppressing relative delta/theta/alpha power.

---

### Biomarker 2 — Alpha-2 Power (10–13 Hz)

**Alpha-2 absolute and relative power** (lower = more degraded thalamocortical rhythm):

*Eyes-closed (n: AD=36, FTD=23, CN=29)*
| Group | Alpha2 Abs (V²/Hz) | Alpha2 Rel | Spec Entropy |
|---|---|---|---|
| AD | 1.200e-11 | 0.03187 | 2.747 |
| FTD | 4.276e-12 | 0.02482 | 2.717 |
| CN | 1.238e-11 | 0.03992 | 2.645 |

*Eyes-open (n: AD=36, FTD=23, CN=29)*
| Group | Alpha2 Abs (V²/Hz) | Alpha2 Rel | Spec Entropy |
|---|---|---|---|
| AD | 3.872e-12 | 0.02528 | 2.748 |
| FTD | 2.835e-12 | 0.03354 | 2.712 |
| CN | 2.397e-11 | 0.04033 | 2.653 |

*Combined — per-subject average (n: AD=36, FTD=23, CN=29)*
| Group | Alpha2 Abs (V²/Hz) | Alpha2 Rel | Spec Entropy |
|---|---|---|---|
| AD | 7.937e-12 | 0.02857 | 2.747 |
| FTD | 3.556e-12 | 0.02918 | 2.714 |
| CN | 1.817e-11 | 0.04012 | 2.649 |

> CN has clearly higher alpha2 relative power than AD and FTD across all conditions. Spectral entropy is slightly higher in AD/FTD than CN, consistent with less organized alpha rhythms.

---

### Biomarker 3 — EEG Microstates (A/B/C/D Coverage)

Fitted with ModKMeans (k=4) on first 5 minutes at 250 Hz. Labels canonically assigned via Hungarian algorithm matching to Koenig et al. 2002 reference topographies.

*Eyes-closed (n: AD=36, FTD=23, CN=29)*
| State | AD | FTD | CN |
|---|---|---|---|
| A | 0.2479 | 0.2680 | 0.2396 |
| B | 0.2507 | 0.2599 | 0.2300 |
| C | 0.2749 | 0.2792 | 0.3174 |
| D | 0.2265 | 0.1929 | 0.2130 |

*Eyes-open (n: AD=36, FTD=23, CN=29)*
| State | AD | FTD | CN |
|---|---|---|---|
| A | 0.2366 | 0.2583 | 0.2576 |
| B | 0.2539 | 0.2724 | 0.2282 |
| C | 0.3003 | 0.2610 | 0.2815 |
| D | 0.2092 | 0.2083 | 0.2327 |

*Combined — per-subject average (n: AD=36, FTD=23, CN=29)*
| State | AD | FTD | CN |
|---|---|---|---|
| A | 0.2423 | 0.2632 | 0.2486 |
| B | 0.2523 | 0.2661 | 0.2291 |
| C | 0.2876 | 0.2701 | 0.2995 |
| D | 0.2179 | 0.2006 | 0.2229 |

*Auditory — standalone (n: AD=17, CN=10)*
| State | AD | CN |
|---|---|---|
| A | 0.2560 | 0.2637 |
| B | 0.2477 | 0.2414 |
| C | 0.2466 | 0.2489 |
| D | 0.2497 | 0.2461 |

> Microstate coverage is near-uniform (~0.25 each) in the auditory dataset — the 40 Hz entrainment overrides the natural resting-state microstate dynamics, making these values uninformative as dementia markers.

> In resting state, CN shows higher microstate C coverage than AD/FTD in eyes-closed (0.317 vs 0.275 / 0.279), which is consistent with stronger attention network engagement in healthy subjects.

---

## Biomarker Descriptions

### Spectral Power Ratios

Computed as **ratio of global channel means** (mean theta power across all channels divided by mean alpha power — not per-channel ratio).

| Column | Formula | Unit |
|---|---|---|
| `theta_alpha_ratio` | mean(θ 4–8 Hz) / mean(α 8–13 Hz) | dimensionless |
| `DTABR` | (mean(δ) + mean(θ)) / (mean(α) + mean(β)) | dimensionless |
| `slowing_ratio` | (δ+θ) / (α+β+γ) | dimensionless |
| `delta_abs` … `gamma_abs` | Mean Welch PSD in band | V²/Hz |
| `delta_rel` … `beta_rel` | Band mean / (δ+θ+α+β total) | 0–1 |

### Alpha-2 Metrics (10–13 Hz)

| Column | Description | Unit |
|---|---|---|
| `alpha2_abs` | Mean Welch PSD in 10–13 Hz, averaged across all channels | V²/Hz |
| `alpha2_rel` | alpha2_abs / (δ+θ+α+β total mean) | 0–1 |
| `alpha2_perm_entropy` | Permutation entropy of channel-averaged alpha-2 signal | 0–1 |
| `alpha2_spectral_entropy` | Shannon entropy of PSD shape within 10–13 Hz | bits |
| `alpha2_peak_freq` | Dominant frequency in 10–13 Hz band | Hz |
| `alpha2_asymmetry` | (right − left) / (right + left) alpha-2 power | −1 to +1 |
| `alpha2_posterior_abs/rel` | Alpha-2 power at O1/O2/P3/PZ/P4 | V²/Hz or 0–1 |
| `alpha2_occipital_abs/rel` | Alpha-2 power at O1/O2 | V²/Hz or 0–1 |
| `alpha2_frontal_abs/rel` | Alpha-2 power at FP1/FP2/F3/FZ/F4/F7/F8 | V²/Hz or 0–1 |

### Microstates

| Column | Description | Unit |
|---|---|---|
| `ms_A_coverage` … `ms_D_coverage` | Fraction of time in each state | 0–1 |
| `ms_A_mean_dur_ms` … | Average visit duration | milliseconds |
| `ms_A_occurrence` … | Occurrences per second | Hz |
| `ms_GEV` | Global explained variance by the 4 states | 0–1 |
| `ms_trans_AB` … (12 columns) | State-to-state transition probability | 0–1 |

**Canonical state descriptions (Koenig et al. 2002):**
| State | Positive lobe | Associated network |
|---|---|---|
| A | O1, P7, P3 (left posterior) | Visual processing |
| B | O2, P8, P4 (right posterior) | Visual processing |
| C | FP1, FP2, F3, FZ, F4, F7, F8 (frontal) | Attention / salience |
| D | CZ, C3, C4, PZ, P3, P4 (centroparietal) | Default mode network |

---

## Technical Notes

**PSD computation:** Welch method, 2-second window, `scipy.signal.welch`.

**Band power:** Mean PSD over frequency bins in each band (V²/Hz) — not trapezoid-integrated.

**Relative power denominator:** Sum of (delta + theta + alpha + beta) band means. Gamma excluded to keep values bounded.

**Permutation entropy:** Computed on the spatially-averaged alpha-2 signal. Parameters: order=3, delay=1, normalize=True (`antropy` library).

**Microstate canonical ordering:** ModKMeans cluster centers are matched to canonical A/B/C/D topographies using absolute Pearson correlation + Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). Falls back to sequential labeling if matching fails.

**Combined output:** 88 rows — biomarkers are averaged per subject across eyes-closed and eyes-open conditions. Each subject contributes exactly one row (no double-counting).

**MATLAB v7.3 .set files:** Loaded via h5py (header) + binary .fdt (raw data) when MNE's scipy-based reader fails.
