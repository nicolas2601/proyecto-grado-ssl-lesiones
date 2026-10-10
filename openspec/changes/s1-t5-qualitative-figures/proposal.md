# Proposal: Segmentation and tone-pair figures

## Why
S1 and T5 require qualitative figures tied to pod-produced inputs, but those inputs are not yet versioned in this repository. Reproducible loaders and fail-closed figure builders can be prepared before transfer without creating sample results.

## What Changes
- Add S1 comparison sheets with Original, Reference, student_r3, S4-A2, and Contours columns across high, medium, and low Dice-percentile cases.
- Add T5 before/after tone-pair sheets with the same image crop, lesion mask, shift level, and derived L* shift visible.
- Read only manifests under `resultados/insumos_pod/{S1,T5}/`; stop without exports when manifests or referenced files are absent or invalid.
- Keep synthetic image fixtures inside pytest `tmp_path`; never write synthetic figure artifacts into repository results.

## Capabilities
- `s1-segmentation-case-comparisons`: verified paired segmentation examples across three quality strata.
- `t5-tone-transformation-pairs`: verified original/transformed image pairs with mask and shift provenance.

## Impact
- Add focused scripts and tests under `scripts/figures/` and `tests/`.
- Update the figure inventory with pending input status and source/output paths.
- Do not run model inference, create pod data, modify existing figures, or publish anything externally.
