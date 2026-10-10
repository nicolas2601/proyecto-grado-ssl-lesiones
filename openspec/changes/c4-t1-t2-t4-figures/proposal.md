# Proposal: Per-class and ITA figures

## Why
The thesis needs traceable class-level classification metrics and clear ITA explanations, without mixing mismatched validation partitions or presenting estimated ITA groups as clinical Fitzpatrick diagnoses.

## What Changes
- Add C4 precision, sensitivity, and F1 by diagnosis from matched B71/A71 confusion matrices.
- Add a conceptual ITA geometry diagram (T1), a LaTeX table of verified threshold intervals (T2), and a perilesional ITA violin figure (T4).
- Keep existing figures and `beautiful.py` unchanged; export through the thesis style module with source traceability.

## Capabilities
- `per-class-classification-metrics`: C4 figures from verified count matrices.
- `ita-explanatory-figures`: T1/T2/T4 artifacts from current formula, thresholds, and E5 ROI values.

## Impact
- Add scripts and focused tests under `scripts/figures/` and `tests/`.
- Add C4 and T4 SVG/PDF/PNG exports, T2 LaTeX, captions, notes, and traceability under `resultados/figuras/thesis_c1_c2_20261009_final/`.
- No new experiments, external publication, or thesis-document edits.
