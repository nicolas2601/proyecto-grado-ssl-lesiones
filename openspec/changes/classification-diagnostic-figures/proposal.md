# Proposal: Classification diagnostic figures C5–C7

## Why
The thesis needs reproducible evidence for melanoma discrimination, confidence calibration, and the combined F4 ablation without reconstructing observations from aggregate metrics or overstating causal attribution.

## What Changes
- Add C5 precision–recall and ROC figures from validated per-image probability archives only.
- Add C6 reliability diagrams with explicitly defined predicted-class confidence, fixed bins, bin support, and ECE validation against source JSON.
- Add separate C7 figures for A71 versus B71 and B71 versus B71+F4, using stored paired contrasts and seed robustness artifacts.
- Preserve unavailable comparisons as Pending; do not infer individual component effects for F4.

## Capabilities
- `classification-diagnostic-figures`: traceable C5–C7 figures and exports.

## Impact
- Add figure scripts and focused tests under `scripts/figures/` and `tests/`.
- Update `docs/FIGURE_INVENTORY.md` and add SVG/PDF/PNG plus Spanish sidecars under `resultados/figuras/`.
- Do not modify the thesis document, run new experiments, or replace existing figures.
