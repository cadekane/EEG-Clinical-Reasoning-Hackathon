# Biomarker 3: EEG Microstates — Source 1

## Citation
Yang, Xiaoli, et al. "Resting-State EEG Microstate Features for Alzheimer's Disease Classification." PLOS ONE, vol. 19, no. 12, 2024, article e0311958.
https://pmc.ncbi.nlm.nih.gov/articles/PMC11637251/

## Biomarker Category
EEG Microstates — Coverage, Duration, Occurrence, GEV, Transition Probabilities

## Key Findings
- Classification accuracy of 99.22% using microstate features with SVM in the alpha band (8-13 Hz).
- Average accuracy of 98.61% across four machine learning classifiers using microstate features, substantially outperforming conventional EEG features (91.19%).
- Microstates B and C showed the most pronounced differences between AD patients and healthy controls in the alpha band.
- Altered transitions between microstates distinguished patient groups, particularly increased transitions from microstate A to B in AD patients.
- Higher GEV values indicate more accurate microstate segmentation.
- Duration, coverage, occurrence, transition probability, and average correlation coefficients were extracted as microstate features.

## Clinical Context
- EEG microstates: short (60-120 ms), recurring spatial patterns representing discrete brain network activations.
- Four canonical microstate classes (A, B, C, D) are used as reference topographies.
- AD is associated with global microstate disorganization.
- Microstate features capture large-scale brain-state dynamics that spectral power features alone cannot measure.
- Microstate B is associated with the default mode network and visual processing.

## Relevance to This Project
- Supports use of microstate coverage, duration, occurrence, GEV, and transition probabilities as AD biomarkers.
- Microstate B coverage was selected as the primary microstate biomarker based on its reported association with dementia-related changes.
- Observed pattern in combined resting-state dataset: microstate B coverage AD (0.257) ≈ FTD (0.256) > CN (0.236).
- Increased microstate B coverage in dementia groups is consistent with altered default mode network dynamics in AD.
