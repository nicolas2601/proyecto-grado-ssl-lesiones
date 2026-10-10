## 1. C5 melanoma discrimination
- [x] 1.1 Add strict loaders for E3b, canonical ISIC E1, and fine-tuning archives; verify hashes, IDs, labels, and class order.
- [x] 1.2 Compute binary PR/AP and ROC/AUC from image-level predictions; test tied scores and published HAM melanoma AUC.
- [x] 1.3 Export HAM10000, ISIC 2019, ISIC 2020, and verified ISIC 2020 fine-tuning figures with Spanish sidecars.
- [x] 1.4 Update inventory, run focused tests, and visually inspect PNG previews.
- [x] 1.5 Verify final SVG/PDF/PNG dimensions, editable SVG text, embedded PDF fonts, and source hashes.

## 2. C6 confidence calibration
- [ ] 2.1 Verify probability NPZ inputs and published ECE records for each candidate arm.
- [ ] 2.2 Add tests for predicted-class confidence, bin edges/support, reliability accuracy, and ECE consistency gates.
- [ ] 2.3 Build only reliability panels supported by verified aligned probabilities; document any ECE mismatch or pending input.
- [ ] 2.4 Export figures and sidecars; inspect all PNGs and validate vector/font/dimension metadata.

## 3. C7 ablations
- [ ] 3.1 Locate verified A71/B71 protocol comparison and stored paired B71/B71+F4 contrasts.
- [ ] 3.2 Add tests for source values, intervals, and seed coverage.
- [ ] 3.3 Build two separate figures; explicitly bound claims to bundled A71/B71 and combined F4 effects.
- [ ] 3.4 Export figures and sidecars; inspect all PNGs and validate vector/font/dimension metadata.

## 4. Verification and delivery
- [ ] 4.1 Run focused C5–C7 tests and relevant figure tests; record pre-existing failures separately.
- [ ] 4.2 Validate source hashes, SVG/PDF/PNG metadata, and inspect every final PNG.
- [ ] 4.3 Commit each script/test set and each export batch atomically; report hashes and verification to Paula.
