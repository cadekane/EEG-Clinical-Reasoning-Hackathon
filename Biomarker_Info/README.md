# Biomarker_Info

RAG reference documents for EEG biomarkers used in the Alzheimer's disease / FTD classification pipeline. Each file contains key findings, clinical context, and observed patterns from this project, extracted from peer-reviewed publications.

## Files

| File | Biomarker Category | Paper |
|---|---|---|
| [bio1_source1.md](bio1_source1.md) | Spectral Power Ratios | Moretti 2015 — Theta/alpha interplay in MCI |
| [bio1_source2.md](bio1_source2.md) | Spectral Power Ratios | Papaliagkas 2025 — Role of qEEG in AD diagnosis |
| [bio2_source1.md](bio2_source1.md) | Alpha-2 Power and Complexity | Patchitt 2022 — Alpha3/alpha2 ratio and cognitive performance |
| [bio2_source2.md](bio2_source2.md) | Alpha-2 Power and Complexity | Moretti 2015 — Alpha3/alpha2 ratio and MCI-to-AD conversion |
| [bio3_source1.md](bio3_source1.md) | EEG Microstates | Yang 2024 — Microstate features for AD classification |
| [bio3_source2.md](bio3_source2.md) | EEG Microstates | Chang 2025 — Microstate differences in AD, FTD, and controls |

## Biomarker Summary

| Biomarker | Observed Pattern | Interpretation |
|---|---|---|
| DTABR | AD ≈ FTD > CN | AD and FTD showed stronger broad EEG slowing than CN |
| Theta/Alpha Ratio | AD > FTD > CN | AD showed the strongest focused slowing relative to alpha |
| Alpha2 Absolute Power | CN > AD > FTD | CN preserved stronger alpha-2 activity |
| Alpha2 Relative Power | CN > AD ≈ FTD | CN had more stable alpha-2 rhythm after normalization |
| Microstate B Coverage | AD ≈ FTD > CN | Dementia groups spent more time in microstate B |

Observed patterns are based on the combined resting-state dataset (ds004504 + ds006036, 88 subjects: 36 AD, 23 FTD, 29 CN), averaged per subject across eyes-closed and eyes-open conditions.
