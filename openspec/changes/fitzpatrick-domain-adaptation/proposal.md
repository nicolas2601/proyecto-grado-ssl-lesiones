# Proposal: Fitzpatrick Domain Adaptation (CIRCLe)

**Base branch**: `origin/feature/v6-gpu-ssl-pretraining` (where `src/dermssl/`, `tests/`, docs 15-19 live). Work happens on `feature/fitzpatrick-domain-adaptation` rebased onto it — NOT on `main`.

## Intent

The repo declares skin-tone bias as a known limit (`docs/07`, `docs/12`: HAM10000 is Fitzpatrick I-III) but never measures it: the fairness gap is a documented goal, not a number. This change makes the bias measurable, then mitigates it by replicating CIRCLe (Pakzad et al., arXiv:2208.13528; evaluated on skin-lesion classifiers by Corbin & Marques, IEEE Access 2023). Thesis contribution = relative fairness improvement over baseline.

## Scope

### In Scope — Stage 1 (measure)

- FST estimation via ITA in LAB (`ITA = arctan((L*-50)/b*) x 180/pi`), Groh et al. empirical thresholds, computed on skin pixels only.
- Lesion masks: small U-Net trained on ~2600 ISIC 2018 Task 1 masks (own held-out split), inferred over HAM10000/ISIC2019.
- New `metrics.stratified_summary()` (per-FST-bucket), consumed from `scripts/notebook_code_v5.py`.
- New fairness gate in `gates.py` following the G0-G2 pattern.
- BOSQUE (arXiv 2507.14176) as an optional research task: validate ITA against annotated Fitzpatrick if licensing lands in time; otherwise deferred, non-blocking.

### In Scope — Stage 2 (mitigate, gated on Stage 1 showing measurable bias)

- CIRCLe replication on the **trainable probe only** (backbone stays frozen, as in v5): deterministic LAB color transformer preserving the lesion via mask, joint MSE regularization, 50/50 synthetic blend with max 2-level FST shift.
- Comparison: baseline vs. regularization vs. synthetic blend.

### Out of Scope

- Hooking the color transformer into v6 pretraining (`pretrain/color_constancy.py`, `pretrain/torch_data.py`) — blocked on DDP fixes and cluster access; also must respect `assert_label_free` in `pretrain/config.py`. Recorded as future work only.
- Retraining or unfreezing any backbone.

## Capabilities

### New Capabilities
- `fitzpatrick-estimation`: lesion masking + ITA-based FST bucket assignment.
- `fairness-evaluation`: FST-stratified metrics and the new acceptance gate.
- `circle-adaptation`: color transformer, joint regularization, synthetic blend at probe level.

### Modified Capabilities
- None (`openspec/specs/` is empty; nothing at spec level changes).

## Approach

Isolated, dependency-light modules under `src/dermssl/`, CPU-testable, wired into the existing gate/metric flow rather than into `probes.py`. `scikit-image` becomes a new `requirements-dev.txt` dependency for LAB conversion. U-Net training is GPU work (Kaggle), separated from CPU-testable logic per `rules.tasks`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/dermssl/metrics.py` | Modified | add `stratified_summary()` |
| `src/dermssl/gates.py` | Modified | new fairness gate, skip-with-warning on thin buckets |
| `src/dermssl/` (new module) | New | ITA/FST estimation + CIRCLe transforms |
| `scripts/notebook_code_v5.py` | Modified | consume stratified summary |
| `tests/test_notebook_build.py` | Modified | extend `REQUIRED_API` |
| `requirements-dev.txt` | Modified | add `scikit-image` |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| ISIC 2018 Task 1 Kaggle handle unconfirmed | Med | Confirm before mask work; fall back to official ISIC archive download |
| BOSQUE license/access unverified | High | Optional dataset by design; measurement proceeds on HAM/ISIC2019 |
| `scikit-image` is a new dependency (repo has neither it nor opencv) | Low | Dev-only, pinned, isolated to one module |
| Sparse FST buckets (esp. FST 5-6) | High | Gate emits SKIP + warning (never FAIL), mirroring the G2 smoke-skip pattern; documented as a known limitation, as the original paper did for FST 6 |
| ITA heuristic != true Fitzpatrick | Med | Reported as an estimator; BOSQUE validation pending |

## Rollback Plan

New modules are additive and import-isolated. Revert = drop the new module(s), remove the gate entry from `evaluate_gates`, revert `metrics.py` / `notebook_code_v5.py` / `REQUIRED_API` hunks, drop `scikit-image`. No data migration, no backbone or checkpoint changes, v5 behavior unchanged when the gate is absent.

## Dependencies

- `scikit-image` (new, `requirements-dev.txt`).
- ISIC 2018 Task 1 Lesion Boundary Segmentation masks (~2600).
- GPU session (Kaggle) for U-Net training and mask inference.
- Stage 2 depends on Stage 1 reporting a measurable gap.

## Success Criteria

- [ ] **Primary**: measurable relative fairness improvement (per-FST gap) of Stage 2 over the Stage 1 baseline, reported with CIs.
- [ ] **Secondary / stretch**: absolute gap <= 0.15 (the threshold documented in `docs/04`, `docs/07`, `docs/11`). Used as the mechanical gate threshold, but exceeding it is a plus, NOT the acceptance condition of the contribution.
- [ ] Every HAM10000/ISIC2019 image carries an FST bucket or an explicit "unavailable" marker.
- [ ] Fairness gate reports pass/fail/skip and skips (not fails) on thin buckets.
- [ ] Full suite green: `CI_REQUIRE_FULL_DEPS=1 pytest -q -rs --strict-markers`; `ruff check` clean.
