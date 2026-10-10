## 1. C4 per-diagnosis metrics
- [x] 1.1 Add RED tests for class metrics, matrix/split consistency, and source traceability.
- [x] 1.2 Implement validated B71/A71 loaders and precision/sensitivity/F1 derivation.
- [x] 1.3 Export one three-panel figure per dataset with Spanish sidecars; inspect PNGs.

## 2. T1/T2 ITA explanation
- [ ] 2.1 Add RED tests for formula geometry, threshold inclusivity, and abstention wording.
- [ ] 2.2 Generate conceptual T1 diagram and T2 LaTeX table from source constants/functions.
- [ ] 2.3 Validate OpenSpec and inspect rendered T1/T2 artifacts.

## 3. T4 ITA violins
- [ ] 3.1 Add RED tests for E5 row filtering, valid/abstention counts, and threshold consistency.
- [ ] 3.2 Build and export perilesional ITA violins with source hashes and Spanish sidecars.
- [ ] 3.3 Inspect all PNGs and verify SVG/PDF dimensions and font metadata.

## 4. Verification and delivery
- [ ] 4.1 Run focused and related tests; report skips/warnings exactly.
- [ ] 4.2 Validate `openspec validate c4-t1-t2-t4-figures` and review the diff.
- [ ] 4.3 Commit C4 and each ITA artifact batch atomically; report hashes to Paula.
