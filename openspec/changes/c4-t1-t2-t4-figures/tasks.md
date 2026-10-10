## 1. C4 per-diagnosis metrics
- [x] 1.1 Add RED tests for class metrics, matrix/split consistency, and source traceability.
- [x] 1.2 Implement validated B71/A71 loaders and precision/sensitivity/F1 derivation.
- [x] 1.3 Export one three-panel figure per dataset with Spanish sidecars; inspect PNGs.

## 2. T1 ITA geometry
- [x] 2.1 Add RED tests for formula, sign, and the b* reporting guard.
- [x] 2.2 Generate conceptual T1 diagram and traceability sidecars.
- [x] 2.3 Verify T1 against the domain source; inspect PNG and PDF/SVG output.

## 3. T2 ITA intervals
- [x] 3.1 Add RED tests for threshold intervals, boundary assignment, and LaTeX output.
- [x] 3.2 Generate a LaTeX table from `FST_THRESHOLDS` and `fst_bucket` with provenance.
- [x] 3.3 Validate OpenSpec, source hashes, and LaTeX fragment; no local TeX compiler is available for a rendered preview.

## 4. T4 ITA violins
- [x] 4.1 Add RED tests for E5 row filtering, valid/abstention counts, and threshold consistency.
- [x] 4.2 Build perilesional ITA violins with source hashes and Spanish sidecars.
- [x] 4.3 Inspect all PNGs and verify SVG/PDF dimensions and font metadata.

## 5. Verification and delivery
- [x] 5.1 Run focused tests and all figure suites; 167 passed with strict dependency checks.
- [x] 5.2 Validate `openspec validate c4-t1-t2-t4-figures`; review source/hash and output metadata.
- [x] 5.3 Commit each figure/table batch atomically; report hashes to Paula.
