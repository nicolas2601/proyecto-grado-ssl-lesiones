## Exploration: fst-domain-adaptation-circle

### Current State

The repository already contains an FST estimator, a JSON-first diagnostic, a label-free GPU SSL pretraining stack, and a CIRCLe Stage 2 probe runner. The components are not yet an end-to-end domain-adaptation pipeline.

`scripts/diagnostico_fst_comparativa.py` executes both a whole-image ITA baseline and `estimate_fst(img)` over each image, emits a per-image CSV audit manifest and a JSON summary. The latter is the appropriate primary artifact. However, it supplies no lesion mask, so `estimate_fst` uses its Otsu fallback for all rows; its `mask_source` therefore represents the fallback rather than verified U-Net masks. The committed/working-tree full-pool summaries cover 68,472 images: HAM10000 10,015 (18.2% bucket changes), ISIC2019 25,331 (27.5%; 19 degenerate fallback masks), and ISIC2020 33,126 (16.9%; 88 degenerate fallback masks).

`src/dermssl/pretrain` is Nico's GPU-oriented, label-free SSL pipeline. It supports SimSiam, SimSiam-DINOv2, continual DINO, and DINO-from-scratch; deterministic crop seeds, DDP, AMP, checkpoint fractions, MLflow parameters, backbone export, and a hard label-leak check are already present. Its input contract is a single recursive image root, not a versioned multi-dataset pool manifest.

The remote `origin/feature/v6-gpu-ssl-pretraining` is available but has diverged substantially: current HEAD is 24 commits ahead and the remote is 41 commits ahead from merge base `032bdd1`. The branch diff is 71 files / 7,163 additions, spanning the SSL system, finetuning, cluster handoff, results and tests. It must be integrated selectively or through an explicit reconciliation, not assumed to be a drop-in dependency.

`src/dermssl/circle.py` applies bounded, deterministic Lab skin-tone shifts outside a lesion mask; `src/dermssl/finetune/circle_torch.py` freezes the backbone and trains a linear probe with task plus invariance regularization. `scripts/kaggle_circle_stage2.py` only runs after a Stage 1 fairness gate is `FAIL`, then compares baseline, shifted-only, and synthetic-blend probes from a precomputed feature bundle. This is a gated mitigation experiment, not domain adaptation during SSL pretraining and not a complete-model integration.

### Affected Areas

- `src/dermssl/fitzpatrick.py` — estimator status, confidence, artifact flags, whole-image baseline, and mask requirements.
- `scripts/diagnostico_fst_comparativa.py` — full-pool manifest/audit protocol; currently fallback-only rather than U-Net-mask validation.
- `scripts/build_kaggle_diagnostico_fst.py` and `scripts/kaggle_fst_manifest.py` — GPU staging and any reproducible mask-generation handoff.
- `src/dermssl/data_resolver.py` — canonical HAM10000/ISIC2019/ISIC2020 discovery, but no immutable pooled-input manifest.
- `src/dermssl/pretrain/{config.py,torch_data.py,torch_loop.py}` and `scripts/pretrain.py` — SSL ingestion, reproducibility, checkpoints, and provenance.
- `origin/feature/v6-gpu-ssl-pretraining` — Nico's remote implementation must be compared file-by-file against local extensions before adoption.
- `src/dermssl/{circle.py,finetune/circle_torch.py}` and `scripts/kaggle_circle_stage2.py` — bounded CIRCLe mitigation and Stage 1 gate contract.
- `src/dermssl/gates.py`, `src/dermssl/metrics.py`, `src/dermssl/finetune/protocol.py` — fairness strata, evaluation outcomes, anti-leak splits, and existing result naming.
- `scripts/build_results_tables.py` and `scripts/build_notebook_v5.py` — generated notebook/report protocol; tables currently consume model-result JSONs and deliberately exclude smoke results.
- `tests/test_fitzpatrick_pipeline_integration.py`, `tests/test_fitzpatrick.py`, `tests/test_circle.py`, `tests/test_circle_torch_static.py`, `tests/test_stage2_runner_static.py`, `tests/test_pretrain_*.py` — existing seams suited for contract and provenance tests.

### Constraints and Contradictions

- The professor's first milestone says estimate/validate tone over the agreed pool. The current summaries estimate all images but do not validate against a ground-truth skin-tone reference, nor validate use of real segmentation masks. “Complete pool” must distinguish image coverage from estimator validity.
- CIRCLe needs a lesion mask to preserve lesion pixels, while the FST diagnostic's masked path presently has no supplied lesion mask. A common versioned mask/provenance contract is a prerequisite for both claims.
- FST bucket labels are estimated proxies, not demographic ground truth. They can stratify fairness analysis only with explicit uncertainty, availability, and minimum-sample rules.
- SSL pretraining must remain label-free. FST values, diagnoses, patient IDs, and fairness-gate outcomes must not enter the pretraining dataset, sampler, loss, or augmentations; they belong in a separated audit/evaluation join.
- The current CIRCLe runner trains frozen linear probes on feature bundles after a FAIL gate. It cannot substantiate “complete model” mitigation without a decision on whether CIRCLe is an inference/test-time augmentation, finetuning regularizer, or a separate post-hoc probe study.
- Local working-tree changes include FST scripts/results and the v5 notebook. This change must neither overwrite them nor treat untracked experiment output as committed evidence without recording provenance.
- The remote branch is not merge-safe by default because it has broad overlapping ownership across SSL, finetune, results, notebooks, and tests.

### Approaches

| Approach | Pros | Cons | Effort |
|---|---|---|---|
| 1. Extend the current fallback diagnostic directly into CIRCLe and SSL | Fastest path; reuses existing scripts and summaries | Conflates Otsu fallback with U-Net validation, lacks canonical pool provenance, and risks label leakage or unsupported fairness claims | Medium, high scientific risk |
| 2. Stage-gated evidence pipeline: canonical pool and mask validation, then SSL adaptation, then CIRCLe probe study, then full-model decision | Separates scientific claims; makes JSON audit artifacts and stop/go gates explicit; preserves label-free SSL; supports interactive review after each milestone | More artifacts and requires resolving mask source plus Nico-branch reconciliation | High, recommended |
| 3. Merge Nico's branch first, then add FST/CIRCLe work | Consolidates a broad pretraining baseline early | 71-file divergence increases merge and regression risk; makes causal attribution and review harder | High, not recommended |

### Recommendation

Adopt Approach 2 as a single SDD change with four dependency-ordered, independently reviewable slices:

1. **FST evidence foundation** — define one JSON pool manifest for the three agreed resized sources, deterministic image identity/source/resolution provenance, a JSON validation report, and CSV only as a per-image audit export. Run whole-image and mask-guided estimates separately; mark unavailable rows explicitly. Do not call the mask-guided result validated until the exact mask producer/version and a reference-validation or expert-audit rule are decided.
2. **Nico SSL reconciliation and domain-adaptation experiment** — map the remote branch's selected pretraining interfaces into the local branch without a blind merge. Build a label-free pooled image input whose run manifest records source composition but is inaccessible to training loss/sampling. Compare no-adaptation and domain-adapted backbones under the existing anti-leak evaluation protocol.
3. **CIRCLe mitigation study** — retain the existing FAIL-only Stage 2 gate, but feed it FST strata with estimator availability/confidence and minimum-bucket eligibility. Emit one JSON report containing baseline, shifted-only, and synthetic-blend outcomes plus per-stratum intervals; CSV is optional for per-sample predictions/audit only.
4. **Complete-model integration decision** — only after the study identifies a credible benefit and an approved operational form, specify a bounded integration (likely finetune regularization or controlled test-time augmentation) and evaluate it against the same backbone/checkpoint/split baseline.

This shape prevents the invalid shortcut of describing fallback FST buckets or a frozen-probe CIRCLe run as validated end-to-end fairness mitigation.

### Missing Decisions for Proposal Questions

1. What is the authoritative mask source/version for each of HAM10000, ISIC2019, and ISIC2020, and what coverage/quality threshold makes an image eligible?
2. What validates FST: external annotated reference data, expert audit sample, agreement with an approved estimator, or only operational QA? The acceptance wording must not overclaim proxy buckets as ground truth.
3. What exactly is the SSL domain-adaptation objective: pooled label-free pretraining, continued pretraining from a specified checkpoint, or a source-to-target setup? Which datasets are sources and targets?
4. What result is sufficient to invoke CIRCLe: a specified FST-stratified gate with confidence intervals and minimum `n`, or any aggregate fairness failure?
5. If CIRCLe succeeds, is the intended complete-model intervention training-time regularization, evaluation-time robustness augmentation, or a separate research comparison?

### Risks

- No confirmed real lesion-mask artifact/producer was found for the current full-pool diagnostic; this blocks a mask-guided FST validation claim.
- Existing FST output is untracked and should be treated as provisional until its command, input checksum, code revision, and mask provenance are captured in JSON.
- FST bucket imbalance is severe (e.g., HAM10000 has 73 masked bucket-5 rows); fairness comparisons can be unstable without predeclared minimum counts and confidence intervals.
- The remote pretraining branch diverges by 71 files, so importing it without a compatibility matrix risks overwriting current FST/CIRCLe work and notebook/result protocol.
- GPU runs are external (Kaggle/cluster) and cannot be fully reproduced locally; local verification must be contract/static tests, with real-run evidence separately recorded.
- This work likely exceeds the 400-line review budget and should be planned as chained, independently verifiable slices if implementation begins.

### Ready for Proposal

No — the technical direction is ready, but the proposal needs the five product/scientific decisions above, particularly authoritative masks, the meaning of FST validation, and the intended definition of domain adaptation. In interactive mode, ask that question round before creating the proposal.
