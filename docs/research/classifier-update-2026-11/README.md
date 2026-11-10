# Classifier update (November 2026): SSL encoder improvements and class-imbalance study

This directory documents the experimental work of November 9-10, 2026 on the self-supervised (SSL) skin-lesion classifier: objective, protocol, results including negative results, a process audit, and a statistical review. All numbers are development-set results unless stated otherwise.

## Objective

Improve the SSL classifier while keeping the encoder self-supervised (MoCo continued on DINOv2 ViT-B/14, no diagnosis labels in representation learning), with two goals:

1. Reduce class-imbalance errors: melanoma classified as nevus (HAM10000) and squamous cell carcinoma classified as basal cell carcinoma (PAD-UFES-20).
2. Verify that gains are not caused by overfitting or by evaluation artefacts.

Two tracks are reported side by side:

- **Track A (SSL):** SSL encoder plus a minimal linear readout. The readout uses labels by necessity and is stated as such.
- **Track B (comparison only):** adds supervision (clinical metadata fusion, supervised fine-tuning). Reported as a gap against Track A, never as an SSL result.

## Protocol

- **Cohorts (development only):** HAM `ham_dev` (1,528 images, 1,152 lesion groups, 7 classes) and PAD `pad_adapt` (1,134 images, 682 patient groups, 6 classes).
- **Sealed sets:** `ham_holdout` and PAD `target_test` were not opened. Confirmation requires a single frozen, predeclared model evaluated once on these sets.
- **Readout:** class-balanced logistic regression, row-wise L2 normalisation, nested grouped cross-validation (outer 5 folds, inner 3), three CV seeds (20261025, 20261026, 20261027), regularisation grid 0.3 to 100.
- **Encoder seeds:** 42, 43, 44.
- **Same-environment comparisons:** every candidate is compared with a baseline refitted in the same environment. Stored historical out-of-fold results are not used as baselines (solver numerics and thread count shift results by about 0.001).
- **Endpoints:** macro-F1 per dataset; HAM melanoma recall and melanoma-to-nevus rate; PAD melanoma recall, SCC recall, and SCC-to-BCC rate.
- **Reporting:** seed-wise paired deltas with group-level bootstrap intervals (conditional on fixed out-of-fold predictions). No p-values from uncentred bootstraps.

## Phase A (November 9): results

| Model | HAM macro-F1 | PAD macro-F1 | HAM mel to NV | PAD SCC to BCC | Status |
|---|---|---|---|---|---|
| V10, 224 px, CLS4 | 0.621 | 0.675 | 24.4% | 47.8% | Reference |
| V10 + 448 px + D4 TTA | 0.614 | 0.678 | 19.7% | 44.8% | Candidate A (post hoc on dev) |
| Queue 4096 + 448 px + D4 TTA | 0.631 | 0.677 | 19.6% | 47.5% | Candidate B; passes the literal predeclared rule, not confirmed by the statistical review |
| Track B: + clinical metadata | 0.619 | 0.737 | 17.5% | 38.7% | Supervised, comparison only |

Findings:

- **Melanoma to nevus:** 24.4% to 19.7% with 448 px and D4 TTA. The reduction is the same sign in 3 of 3 encoder seeds (per-seed changes of -4.3, -4.1, -5.7 pp); two of three intervals exclude zero.
- **Macro-F1:** the queue-4096 gain on HAM is +0.017, +0.020, +0.014 across seeds. Its interval in seed 44 includes zero, and the melanoma endpoint does not move.
- **SCC to BCC:** worse for queue 4096 in seed 44 (+7.0 pp, interval [0.9, 14.9]).
- **Track B:** metadata fusion raises PAD macro-F1 by about +0.06 (interval [0.034, 0.090]) but lowers HAM macro-F1 at 224 px (-0.024 to -0.040). Supervised fine-tuning gives no clinical gain: PAD melanoma recall falls by 16.1 pp.

### Negative results (Phase A)

None of the following improved the clinical endpoints without a macro-F1 cost:

- Label-free sampling: k-means (flat, PAD-uniform, hierarchical), cluster-balanced sampling (224 px HAM macro-F1 -0.037, interval [-0.061, -0.010]).
- SSL objectives and augmentations: SDCLR, colour in training, D4-only views, EMA teacher CLS4, cosine prototypes, 224 px patch pooling.
- Reader-side imbalance handling (seven methods, including logit adjustment, class weights, over- and undersampling): every gain in macro-F1 worsened melanoma-to-nevus or SCC-to-BCC. This is a trade-off frontier, not a free gain.

Overfitting check: the linear readout memorises the training folds (train-OOF gap 0.24 to 0.34 in every configuration). SSL checkpoints at 2,000, 4,000 and 6,000 steps show no peak-then-decline pattern, except for one partial hierarchical-sampling control.

## Phase B (November 10): experimental log

Predeclaration: frozen before any Phase B readout (SHA-256 prefix `618fecbb`). Gate, seed 42, checkpoint 6,000, 448 px CLS4 + D4, against queue 4096 with the same seed: HAM macro-F1 gain at least +0.005, melanoma-to-nevus change at most -1 pp, and PAD macro-F1 change at least -0.01.

| Arm | Delta HAM macro-F1 | Delta HAM mel to NV (pp) | Delta PAD macro-F1 | Gate |
|---|---|---|---|---|
| Debiased InfoNCE (tau+ 0.1) | -0.050 | +0.99 | -0.028 | Fail |
| Queue 16384 | +0.002 | +0.59 | +0.004 | Fail |
| False-negative elimination (threshold 0.8) | +0.021 | 0.00 | -0.007 | Fail (melanoma endpoint does not move) |
| Full-resolution ISIC 2019 source (sensitivity only) | -0.006 | +1.97 | +0.002 | Fail |

Status at the time of writing:

- No Phase B arm passes the gate. Seeds 43 and 44 were not launched.
- MoCHi and cosine temperature schedule: smoke tests only, not trained.
- The DINO objective without iBOT was stopped as superseded by an earlier three-seed DINO+iBOT run, which had already been evaluated.
- Full-resolution data is a source-sensitivity analysis. It changes the image source as well as the file set, so it cannot be attributed to any method without a source-matched baseline.

## Preprocessing diagnostics (November 10)

Label-free preprocessing treatments were tested on HAM and PAD with the same confirmatory rule (seeds 43 and 44 only; seed 42 was used to generate hypotheses): HAM macro-F1 gain positive in both seeds, mean gain at least +0.010, and melanoma-to-nevus change at most +1.0 pp.

| Treatment | Encoder | Delta HAM macro-F1 (mean of seeds 43, 44) | Bootstrap 95% CI | Delta HAM mel to NV (pp) | Rule |
|---|---|---|---|---|---|
| T1: DullRazor hair removal (HAM only) | V10 | +0.044 | [+0.007, +0.085] | -0.39 | Pass |
| T1: DullRazor hair removal (HAM only) | Queue 4096 | +0.013 | [-0.027, +0.060] | -1.78 | Pass |
| T3: DullRazor, then crop (HAM only) | V10 | +0.040 | [-0.005, +0.088] | -1.97 | Pass |
| T3: DullRazor, then crop (HAM only) | Queue 4096 | +0.019 | [-0.026, +0.064] | -3.16 | Pass |

Notes:

- Six confirmatory tests (three treatments by two encoders). Bonferroni-adjusted p-value for T1 with V10 is 0.11; for T3 with V10 it is 0.29.
- Identity controls: the pipeline without treatment reproduces the baseline macro-F1 and out-of-fold probabilities exactly (maximum difference 0.0).
- The melanoma-to-nevus interval for T3 is about +/-7 pp, so the clinical endpoint is not resolved. Adding a crop has no demonstrated gain of its own (crop alone, +0.012, interval includes zero).
- Shades-of-Gray colour constancy was negative for both encoders (HAM -0.014 and -0.017). Geometric padding was neutral.
- On seed 42, DullRazor lowered PAD macro-F1 by about 0.03 for both encoders, so the treatment is applied to HAM only.
- Effects are on HAM dermoscopy only; there is no validation on clinical photographs.

## Outcome

Decision of the project owner (November 10): the model is kept as V10 and released as v10.0.0. No candidate change is adopted, and no candidate is evaluated on the sealed sets.

- No SSL change improved the clinical error endpoints (melanoma to nevus, SCC to BCC) without a macro-F1 cost.
- The only treatment that passed the confirmatory rule is DullRazor hair removal on dermoscopy images (HAM only). It is not adopted, for compatibility with the segmentation pipeline. It is recorded as future work.
- Limits of that result: the hypothesis was generated post hoc from seed-42 readouts that had already been inspected; the confirmatory test uses two encoder seeds; the broader exploration (about 100 or more comparisons) is not covered by the six-test adjustment; the clinical endpoint did not move; there is no clinical-photograph validation.

## Method comparison: re-reading earlier encoders with the current readout

Earlier campaigns used a 768-dimensional CLS readout, so their numbers were not comparable with V10's CLS4 (3,072 dimensions). They were re-read under the current protocol (seed 42 only):

| Encoder (448 px, CLS4 + D4) | HAM macro-F1 | PAD macro-F1 | Delta HAM vs V10 |
|---|---|---|---|
| V10 | 0.614 | 0.676 | reference |
| MoCo v3, no anchor | 0.621 | 0.671 | +0.007 (interval [-0.013, 0.029]; noise) |
| VICReg | 0.577 | 0.653 | -0.037 |
| Barlow Twins | 0.567 | 0.652 | -0.046 |
| Barlow + VICReg | 0.566 | 0.657 | -0.047 |
| DINO + iBOT | 0.573 | 0.642 | -0.041 |
| SimSiam | 0.569 | 0.635 | -0.044 |

No earlier encoder exceeds V10 at 448 px + D4 in seed 42. The MoCo v3 (no anchor) melanoma-to-nevus rate is 3.2 pp worse than V10 in this seed (interval [0.5, 6.1]). The MoCo v3 versus VICReg gap persists in seed 42 (0.044 at 448 px), but the MoCo v3 run has its anchor disabled, so the comparison is confounded. The earlier tables also mixed pooling and readouts, so the gap cannot be attributed to the training recipe from these numbers alone.

## Process audit

An audit of the SSL, extraction, readout and data steps, without training, found:

- **Source resolution:** the source pool is 224 to 256 px, although full-resolution ISIC 2019 images (1,022 x 767) exist. Evaluation at 448 px upsamples the source by 1.75x. Expected impact on melanoma and SCC errors is high but not yet quantified.
- **Augmentations:** the two SSL views differ only by crop and flip. Trivial pixel statistics separate the domains with 97.6% (ISIC pool versus PAD) and 95.0% (HAM versus PAD) balanced accuracy, so the representation may learn acquisition signatures.
- **Transduction:** all 1,134 PAD adaptation images were seen by the SSL objective without labels. PAD figures are therefore transductive and must be declared as such.
- **Sampling:** PAD images were drawn about 21 times each versus 2.1 times for source images. Cluster-uniform sampling over-samples images in small clusters (about 5.9 versus 0.9 draws per image).
- **Readout:** the regularisation parameter sits at the grid edge (C = 100) in 41% of folds (53% for V10 variants). Extending the grid changes macro-F1 by +0.001 (HAM) and +0.006 (PAD) in one seed.
- **Feature normalisation:** L2 normalisation of concatenated CLS blocks gives layer 8 about 42% of the squared norm.
- **Geometry:** extraction squashes images to a square (1.33x distortion for HAM images), whereas training uses random resized crops.
- **Checks that passed:** no identity group crosses train and test folds; out-of-fold predictions are exact; hyperparameters match the recorded configuration.

Items marked as inference in the audit have not been tested. Each has a cheap test defined.

## Statistical review

- **Noise floor:** standard deviation of macro-F1 across CV seeds is about 0.008 to 0.010. Thresholds of about 0.01 are roughly one standard deviation.
- **Multiplicity:** about 100 or more development comparisons were made without correction. Development winners are optimistic.
- **Sign-consistency rule:** with three encoder seeds, a 3/3 positive pattern has probability 1/8 under the null per dataset and feature. The rule is weak and any success is reported as exploratory unless it survives a stricter test.
- **Predeclaration timing:** the timestamp of the Phase B predeclaration was not captured at save time. The corrected value is the file modification time (12:47 UTC), and the predeclaration was written before any Phase B readout.

## Limitations

- All results are on development cohorts. Sealed-set confirmation is pending.
- PAD results are transductive (see the process audit).
- Overlap between the SSL pool and `ham_holdout` is excluded by documented policy and by perceptual-hash filtering only. It has not been verified independently by content hash.
- The SSCD copy-detection audit of the pool has not been run.
- Track B results use the labelled metadata and are not SSL results.
- Melanoma and SCC endpoints rest on small counts (for example, 169 melanoma images in HAM dev).

## Open items

1. Release v10.0.0 (V10 model): three packages (PyTorch weights, ONNX model, Docker image), with complete release notes that include all negative results and the multiplicity count.
2. Future work: DullRazor for dermoscopy, if the segmentation pipeline can accept it; validation on clinical photographs.
3. Independent verification of the holdout exclusion by content hash.
4. SSCD overlap audit of the source pool.
5. Source-matched baseline for the full-resolution analysis.
6. Licence verification for PanDerm before any use as a reference encoder.
## Files in this directory

| File | Content |
|---|---|
| `method-comparison.json` | Phase A candidate comparison, paired deltas, overfitting checks, Track B |
| `method-comparison-nov10.json` | Phase B comparison against the queue-4096 baseline |
| `experimental-log-nov10.json` | Phase B predeclaration, arm status, checkpoint curves |
| `gate-seed42.json` | Phase B seed-42 gate evaluation |
| `diagnostics-results.json`, `diagnostics-preregistration.json`, `diagnostics-multiplicity.json`, `diagnostics-treatment-v2.json` | Preprocessing diagnostics (DullRazor, colour constancy, crop): predeclaration, confirmatory results, multiplicity |
| `process-audit.json` | Process audit findings and checks |
| `reread-cls4.json` | Re-reading of earlier encoders with the current readout |
| `inventory.md` / `inventory.json` | Inventory of experiments and conflicts found in earlier tables |
| `reader-imbalance.json`, `reader-gap.json` | Reader-side imbalance arms and train-OOF gaps |
| `cluster-sampling.json`, `cluster-sampling-hier.json` | Cluster-based sampling arms |
| `supervised-finetune.json` | Supervised fine-tuning pilot (Track B) |
| `ssl-arms.json`, `sdclr-prep.json` | SSL arm definitions and preparation notes |
| `literature-*.json` | Literature searches and method shortlists |

The per-image out-of-fold prediction files are not included. Their summaries appear in the JSON files above.

## Related repositories

- PERMA (application): https://github.com/nicolas2601/perma
