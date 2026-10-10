# Proposal: Reproducible C1/C2 thesis figures

## Why
The classification results need publication-ready figures with explicit traceability to the committed result artifacts. The existing `beautiful.py` style does not match the thesis brief and is used by previously delivered figures, so changing it would create unnecessary regressions.

## What Changes
- Add a Spanish inventory of thesis figures, source artifacts, evaluation units, evidence status, and missing inputs.
- Add an isolated `thesis_style.py` module for the brief's palette, Nimbus Mono PS typography, 17.3 cm width, editable SVG, embedded-font PDF, and 300 dpi PNG exports.
- Add C1 macro-F1 horizontal-bar figures for HAM10000, ISIC 2019, and ISIC 2020, using only existing frozen-probe results and labeling each dataset's actual protocol and support.
- Add C2 paired macro-F1 forest figures only where paired source artifacts exist: HAM10000 B71-TTA against eight frozen baselines, plus canonical ISIC B71-original versus the matched public B71 control.
- Add tests that verify source-value equality, pair selection, resampling metadata, output dimensions, and export formats.

## Capabilities
- `thesis-classification-figures`: traceable C1/C2 figures and shared export style.

## Impact
- New scripts: `scripts/figures/thesis_style.py`, `fig_c1_macro_f1.py`, and `fig_c2_paired_delta_f1.py`.
- New tests for style, C1, and C2.
- New inventory and generated C1/C2 exports under `docs/` and `resultados/figuras/`.
- Existing `beautiful.py`, prior figure scripts, datasets, and experiment artifacts remain unchanged. No training or evaluation reruns.
