# Tasks: Fitzpatrick Domain Adaptation (CIRCLe)

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~1400-1800 across 4 units |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 -> PR 2 -> PR 3 -> PR 4 |
| Delivery strategy | auto-chain |
| Chain strategy | feature-branch-chain |

Decision needed before apply: Resolved - feature-branch-chain
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | FST core (`fitzpatrick.py`) | PR 1 | Independent |
| 2 | Fairness eval + notebook wiring | PR 2 | Needs PR 1 |
| 3 | GPU mask pipeline (static-tested) | PR 3 | Needs PR 1 |
| 4 | CIRCLe Stage 2 + comparison report | PR 4 | Gated on Stage 1 bias; needs PR 1-2 |

## Phase 0: Preconditions

- [x] 0.1 Confirm ISIC 2018 Task 1 Kaggle handle. **Resolved**: `tschandl/isic2018-challenge-task1-data-segmentation` (mirror by the HAM10000 author, 2594 JPEG + PNG mask pairs — matches the proposal's ~2600 estimate). Fallback = official ISIC archive if the mirror is pulled. Unblocks Phase 3.
- [x] 0.2 Flag `min_bucket_n=30` as placeholder; revisit after first `fst_manifest.csv` run. Already flagged in `design.md` ("Open Questions" + the `fairness_gate` contract's default value comment); no code uses it yet in PR1 (it belongs to Phase 2's `fairness_gate`). Revisit trigger stands for Phase 2.

## Phase 1: FST core (`src/dermssl/fitzpatrick.py`) - PR 1

- [x] 1.1 RED `tests/test_fitzpatrick.py`: `srgb_to_lab` vs `skimage.color.rgb2lab`, atol=1e-4.
- [x] 1.2 GREEN: implement `srgb_to_lab` (numpy, D65/CIE Lab).
- [x] 1.3 RED: `ita_degrees`/`fst_bucket` at Groh thresholds (55,41,28,19,10) and boundaries.
- [x] 1.4 GREEN: implement `FST_THRESHOLDS`, `ita_degrees`, `fst_bucket`.
- [x] 1.5 RED: `estimate_fst` returns `FSTEstimate(bucket=None, status=...)` for `no_skin` and `degenerate_mask`.
- [x] 1.6 GREEN: implement `estimate_fst`, composing `pretrain/lesion_guided.lesion_mask` Otsu fallback + skin selection + ITA/bucket.
- [x] 1.7 Add `scikit-image>=0.22` to `requirements-dev.txt`; add `"skimage"` to `REQUIRED_FOR_FULL_RUN` in `test_environment_complete.py`.
- [x] 1.8 REFACTOR: docstrings; assert no runtime `skimage` import.

**PR1 status**: Phase 1 complete, all 8 tasks done, 22/22 tests green
(`tests/test_fitzpatrick.py`). Local branch
`feature/fitzpatrick-domain-adaptation-pr1-fst-core`, off tracker
`feature/fitzpatrick-domain-adaptation`. Not pushed, no PR opened -
awaiting review per the apply-agent's instructions. Also found and
fixed a real bug during TDD (see apply-progress: `ita_degrees` was
using `arctan2` instead of the literature's plain `arctan`, which
would have wrapped negative-`b*` ITA readings outside the documented
[-90, 90] invariant).

## Phase 2: Fairness evaluation - PR 2 (needs PR 1)

- [x] 2.1 RED `tests/test_metrics.py`: `stratified_summary` table-driven (populated, sparse-SKIPPED, empty-SKIPPED, `"unavailable"` key).
- [x] 2.2 GREEN: implement `metrics.stratified_summary(y_true, proba, buckets, n_classes, min_n=30)`.
- [x] 2.3 RED `tests/test_gates.py`: `fairness_gate` table-driven (pass/fail/skip-thin/skip-empty).
- [x] 2.4 GREEN: implement `gates.fairness_gate(summary, *, max_gap=0.15, min_bucket_n=30)`, label `"G3 equidad por fototipo"`, standalone per D1.
- [x] 2.5 `test_notebook_build.py`: `REQUIRED_API += ["stratified_summary", "fairness_gate", "estimate_fst"]`.
- [x] 2.6 `__init__.py`: `MODULE_ORDER` insert `"fitzpatrick"` before `"gates"`.
- [x] 2.7 `notebook_code_v5.py`: load manifest if present, call `stratified_summary`, append `fairness_gate(strat)`; no-manifest path byte-identical.

**PR2 status**: Phase 2 complete, all 7 tasks done. New CPU-testable
functions: `metrics.stratified_summary` (8 tests), `gates.fairness_gate`
(9 tests, standalone per D1, composes via `gates = evaluate_gates(...) +
[fairness_gate(strat)]`). Notebook wiring: `REPORT_CODE` in
`notebook_code_v5.py` now loads `fst_manifest.csv` if present and appends
the fairness gate; when the manifest is absent, `gate_list` is never
touched by the new code at all (structurally enforced by 2 new AST-based
tests in `test_notebook_build.py`), so the no-manifest run is byte-identical
to pre-PR2 v5 output — this is a deliberate, stricter reading of "v5
behavior unchanged" than design.md's flow diagram literally draws (which
shows the manifest join always running); see apply-progress for the full
rationale. `src/dermssl/__init__.py::MODULE_ORDER` now inlines
`fitzpatrick` (before `gates`) into the notebook core, wiring `estimate_fst`
into `REQUIRED_API` for the first time. Notebook regenerated
(`python scripts/build_notebook_v5.py`). Local branch
`feature/fitzpatrick-domain-adaptation-pr2-fairness-eval`, off
`feature/fitzpatrick-domain-adaptation-pr1-fst-core` (needs PR1's
`fitzpatrick.py`). Not pushed, no PR opened - awaiting review. Full suite:
346 passed, 5 pre-existing unrelated failures (`test_readme_tables_fresh.py`,
Windows cp1252 `UnicodeDecodeError` reading `README.md`, same as documented
in PR1's apply-progress). `ruff check .`: all checks passed.

## Phase 3: GPU mask pipeline (static-tested) - PR 3 (needs PR 1)

- [x] 3.1 RED `tests/test_torch_unet_static.py`: py_compile + ast checks on `torch_unet.py`, modeled on `test_pretrain_torch_layer.py`.
- [x] 3.2 GREEN: implement `torch_unet.py` (U-Net, ISIC 2018 masks) + Kaggle script emitting `fst_manifest.csv` with mask provenance, confidence, fallback, artifact, and explicit unavailable fields.
- [x] 3.3 Manifest schema doc written (`docs/20-esquema-fst-manifest.md`) - schema, example, generation command, downstream consumption. **The actual Kaggle run is still a MANUAL step, NOT done here**: this apply session has no GPU, no Kaggle session, and no access to run `scripts/kaggle_fst_manifest.py`. See "PR3 status" below for the exact TODO.

**PR3 status**: Phase 3 complete (static-tested layer only), all 3 tasks
done. New: `src/dermssl/torch_unet.py` (U-Net model, `ISIC2018SegmentationDataset`,
`train_unet`, `save_checkpoint`/`load_checkpoint`, `infer_mask` - all
GPU-only, py_compile/ast-tested via `tests/test_torch_unet_static.py`, 14
tests, NOT wired into `MODULE_ORDER`, confirmed to never import torch
transitively through `import dermssl`); `scripts/kaggle_fst_manifest.py`
(trains the U-Net once on ISIC 2018 Task 1, runs inference on the FST target
pool, writes `fst_manifest.csv` via `fitzpatrick.estimate_fst`, mirrors
`scripts/pretrain.py`'s validate-before-import-torch order);
`docs/20-esquema-fst-manifest.md` (schema doc, no raw data). Local branch
`feature/fitzpatrick-domain-adaptation-pr3-gpu-mask-pipeline`, off
`feature/fitzpatrick-domain-adaptation-pr1-fst-core` (needs only PR1's
`fitzpatrick.py`, NOT PR2 - sibling branch to PR2, per the work-units
table). Not pushed, no PR opened - awaiting review. Full suite: 341 passed,
5 pre-existing unrelated failures (`test_readme_tables_fresh.py`, Windows
cp1252 `UnicodeDecodeError` reading `README.md`, same as PR1/PR2). `ruff
check .`: all checks passed.

**Manual Kaggle TODO for a human with a real GPU session** (not done, not
doable from this environment):
1. Push/run `scripts/kaggle_fst_manifest.py` on Kaggle with
   `--unet-train-images`/`--unet-train-masks` pointed at the
   `tschandl/isic2018-challenge-task1-data-segmentation` dataset (task 0.1's
   resolved handle) and `--target-image-root` pointed at the FST target pool
   (e.g. the HAM10000 image folder already used by v5).
2. Start with `--smoke` to validate wiring (1 epoch, <=8 target images), then
   a real run (`--epochs 20` default, tune as needed).
3. Output: `fst_manifest.csv` (schema in `docs/20-esquema-fst-manifest.md`) -
   download it from the Kaggle session and drop it where
   `scripts/notebook_code_v5.py`'s `FST_MANIFEST_PATH` (`WORK /
   "fst_manifest.csv"`) expects it before running v5's report cell, so the
   fairness gate actually evaluates instead of SKIPping for "no manifest".

## Phase 4: CIRCLe transformer - PR 4a (needs PR 1, gated on Stage 1 bias)

- [x] 4.1 RED `tests/test_circle.py`: determinism, lesion unchanged under mask, invalid mask raises/no-op, output LAB/RGB in range.
- [x] 4.2 GREEN: implement `circle.py` (transformer, FST-shift sampling, blend policy, clamp at 2 levels).
- [x] 4.3 `__init__.py`: `MODULE_ORDER` append `"circle"` after `"gates"`.

**PR4a status**: Phase 4 complete, all 3 tasks done, 26/26 tests green
(`tests/test_circle.py`). Pure numpy, real TDD (not static-only): `shift_skin_tone`
(lesion-preserving LAB color shift), `clamp_fst_shift`/`target_fst_bucket` (bounded
2-level shift), `synthetic_blend_mask` (blend policy), `lab_to_srgb` (inverse of
`fitzpatrick.srgb_to_lab`), `should_run_stage2` (stage-gate dependency, spec.md req 1
- not itemized in the original 4.1 bullet but added as extra coverage since it is an
explicit spec.md requirement assigned to this module's domain).

## Phase 5: Joint regularization (static-tested) - PR 4b

- [x] 5.1 RED: extend static test suite for `finetune/circle_torch.py` (py_compile + ast).
- [x] 5.2 GREEN: implement `circle_torch.py` MSE regularization, probe-only trainable, backbone `requires_grad=False` asserted.
- [x] 5.3 Manual Kaggle checkpoint documented (`docs/21-checkpoint-manual-circle-stage2.md`) - baseline vs regularization vs synthetic-blend runs, schema/steps/precondition written. **The actual Kaggle GPU run is explicitly NOT done**: this apply environment has no GPU, no Kaggle session, no real `fst_manifest.csv` (PR3's own Kaggle step is also still pending). See "Manual Kaggle TODO" below.

**PR4b status**: Phase 5 complete, all 3 tasks done, 11/11 static tests green
(`tests/test_circle_torch_static.py`). `finetune/circle_torch.py`: `freeze_backbone`,
`ProbeHead`, `build_probe_optimizer`, `joint_regularization_loss`, `stage2_train_step`
- backbone `requires_grad=False` asserted before every optimizer step, backbone
features computed under `torch.no_grad()` (structurally unreachable by
`backward()`), optimizer built from `probe.parameters()` only.

## Phase 6: Comparison reporting - PR 4c

- [x] 6.1 RED: test report structure (3 variants, per-variant gap, relative improvement vs baseline).
- [x] 6.2 GREEN: implement comparison report consuming 3 `stratified_summary()` runs.

**PR4c status**: Phase 6 complete, both tasks done, 7/7 new tests green
(`tests/test_metrics.py`). `metrics.comparison_report(baseline, regularization,
synthetic_blend, *, metric="macro_f1", min_bucket_n=30)` - per-variant gap (max-min
`metric` among qualifying buckets) + each mitigation variant's `relative_improvement`
over the baseline gap; `None` (never a fabricated number) when the baseline is
undefined, zero, or the variant itself is undefined. Deviation from design.md: File
Changes table does not name a file for this function - placed in `metrics.py` by
analogy with `stratified_summary`'s placement (see apply-progress for full
rationale). Tested only against synthetic, hand-built `stratified_summary()`-shaped
fixtures - no real data exists yet.

**Manual Kaggle TODO for a human with a real GPU session (Phase 5/6, not done, not
doable from this environment)**: see `docs/21-checkpoint-manual-circle-stage2.md`
for the full checklist. Summary: (1) confirm `fst_manifest.csv` exists and Stage 1's
fairness gate reports `FAIL` (`circle.should_run_stage2`); (2) create the required
real frozen-backbone feature bundle; (3) run `scripts/kaggle_circle_stage2.py
--smoke`; (4) run the full three-variant protocol; (5) retain the paired-bootstrap
confidence intervals and relative improvement report. No real result has been
produced in this environment.

## Phase 7: Full-suite verification

- [x] 7.1 `CI_REQUIRE_FULL_DEPS=1 py -m pytest -q -rs --strict-markers`: 432 passed; `py -m ruff check .`: all checks passed; `git diff --check`: clean (line-ending conversion warnings only).
- [x] 7.2 Confirm v5 output unchanged when `fst_manifest.csv` absent (gate SKIP path), covered by the notebook AST/integration tests in the strict suite.

## Phase 8: Integration remediation after audit

- [x] 8.1 Integrate PR3 into the PR4 line on `feature/fitzpatrick-domain-adaptation-integration`, yielding one local review line with PR1+PR2+PR3+PR4.
- [x] 8.2 Read README freshness fixtures as UTF-8, removing the five Windows cp1252 failures.
- [x] 8.3 Split U-Net data deterministically into held-out train/validation sets; report validation BCE/Dice/IoU; restore and save the best validation-Dice checkpoint.
- [x] 8.4 Complete the FST contract and manifest with `mask_confidence`, `fallback_used`, reduced `confidence`, `artifact_detected`, and explicit `unavailable` output.
- [x] 8.5 Preserve unavailable/missing FST rows in Stage 1 counts and use the balanced linear probe for the fairness path.
- [x] 8.6 Add paired-bootstrap 95% confidence intervals for per-variant gaps and relative improvements over baseline.
- [x] 8.7 Add `scripts/kaggle_circle_stage2.py`, preserving Stage 1 balanced-probe predictions as the primary baseline and training regularization and synthetic-blend probes from real supplied feature bundles; smoke output is explicitly non-final.
- [x] 8.8 Add integration/static tests for FST-to-fairness flow, U-Net validation, Stage 2 gate-schema compatibility, and the executable Stage 2 runner.

## Phase 9: Confirmed review remediation

- [x] 9.1 Read the actual `Gate.as_dict()` schema (`estado`), normalize state case, and reject unknown Stage 1 gate states.
- [x] 9.2 Exclude `unavailable` rows from G3 gaps and paired-bootstrap strata while retaining their reported count.
- [x] 9.3 Derive the LAB lightness delta from source ITA and the clamped target bucket so transformed samples stay within the requested two-level bound.
- [x] 9.4 Canonicalize string numeric FST values at the NPZ boundary and validate their domain and total validation count.
- [x] 9.5 Use preserved Stage 1 `logreg_balanced` probabilities as the primary Stage 2 baseline.
- [x] 9.6 Reject lesion masks whose shape does not exactly match the input image spatial shape.

## External execution still pending

- [ ] Run the U-Net/FST manifest pipeline on real ISIC/HAM10000 data with a GPU and retain best validation Dice/IoU plus the checkpoint.
- [ ] Run Stage 1 on the real manifest and record the G3 decision and unavailable count.
- [ ] If G3 is `FAIL`, run Stage 2 smoke and full experiments on real frozen-backbone features; report point estimates and paired-bootstrap confidence intervals. These runs require external data, Kaggle credentials, and GPU compute and were not available during apply.
