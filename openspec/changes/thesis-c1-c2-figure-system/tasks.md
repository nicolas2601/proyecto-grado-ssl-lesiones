## 1. Inventory and source verification
- [x] 1.1 Build Spanish inventory for all figures in the brief with objective, placement, data/artifact paths, commit/revision, evaluation/resampling unit, evidence state, and missing inputs.
- [x] 1.2 Verify ITA thresholds against `src/dermssl/fitzpatrick.py` and docs; document that Groh thresholds are `(40, 23, 12, 0, -25)` and distinguish them from the superseded Chardon values.
- [x] 1.3 Verify CIRCLe discretization/sign against `src/dermssl/circle.py` and `scripts/generate_tone_transforms.py`; document that 8 L* per bucket × 2 buckets = 16 L*, with positive shift darkening.
- [x] 1.4 Validate E3b JSON digest against its environment sidecar and record the C1/C2 source revisions and input hashes.

## 2. Shared style module
- [x] 2.1 Add tests for approved palette, Nimbus/Courier resolution, physical width, font sizes, line widths, and export metadata.
- [x] 2.2 Implement `scripts/figures/thesis_style.py` without changing `beautiful.py`.
- [x] 2.3 Confirm SVG text is editable, PDF fonts are embedded, PNG width matches 17.3 cm at 300 dpi, and `bbox_inches='tight'` is not used.

## 3. C1 macro-F1 figures
- [x] 3.1 Add tests comparing HAM10000, ISIC 2019, and ISIC 2020 C1 values/supports to committed source JSON.
- [x] 3.2 Implement one C1 horizontal-bar figure per dataset, preserving fixed model order and excluding fine-tuning rows.
- [x] 3.3 Export three formats and Spanish caption/note plus traceability for each dataset.

## 4. C2 paired macro-F1 figures
- [x] 4.1 Add tests for eight HAM10000 E3b paired contrasts and canonical ISIC B71-original/public-B71 contrasts.
- [x] 4.2 Implement C2 forest figures from stored contrast JSON only; show zero, exact CIs, units, resamples, and no multiplicity correction.
- [x] 4.3 Export three formats and Spanish caption/note plus traceability for each contrast family.

## 5. Verification
- [x] 5.1 Run RED/GREEN tests for style, C1, and C2.
- [x] 5.2 Run related existing figure tests and verify all source hashes and output dimensions.
- [x] 5.3 Inspect generated SVG/PDF/PNG visually; stop on overlap, clipping, wrong font, or source mismatch.
- [ ] 5.4 Report verified paths and exact verification commands to Paula; do not send to Pau or modify her repository/document in this change.
