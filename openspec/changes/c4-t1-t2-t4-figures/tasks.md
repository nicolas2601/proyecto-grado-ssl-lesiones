## 1. C4 per-diagnosis metrics
- [x] 1.1 Add RED tests for class metrics, matrix/split consistency, and source traceability.
- [x] 1.2 Implement validated B71/A71 loaders and precision/sensitivity/F1 derivation.
- [x] 1.3 Export one three-panel figure per dataset with Spanish sidecars; inspect PNGs.

## 2. T1 ITA geometry
- [x] 2.1 Add RED tests for formula, sign, and the b* reporting guard.
- [x] 2.2 Generate conceptual T1 diagram and traceability sidecars.
- [x] 2.3 Verify T1 against the domain source; inspect PNG and PDF/SVG output.

## 3. T2 ITA intervals
- [ ] 3.1 Add RED tests for threshold intervals, boundary assignment, and LaTeX output.
- [ ] 3.2 Generate a LaTeX table from `FST_THRESHOLDS` and `fst_bucket` with provenance.
- [ ] 3.3 Validate the table and inspect the rendered artifact.

## 4. T4 ITA violins
- [ ] 4.1 Add RED tests for E5 row filtering, valid/abstention counts, and threshold consistency.
- [ ] 4.2 Build perilesional ITA violins with source hashes and Spanish sidecars.
- [ ] 4.3 Inspect all PNGs and verify SVG/PDF dimensions and font metadata.

## 5. Verification and delivery
- [ ] 5.1 Run focused and related tests; report skips/warnings exactly.
- [ ] 5.2 Validate `openspec validate c4-t1-t2-t4-figures` and review the diff.
- [ ] 5.3 Commit each figure/table batch atomically; report hashes to Paula.
