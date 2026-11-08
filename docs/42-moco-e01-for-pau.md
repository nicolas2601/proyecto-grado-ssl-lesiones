# 42. E01 final proposed MoCo recipe and recovered historical implementation

## Identity and recovery

The final E01 classifier recipe is **MoCo-v3-style on DINOv2 ViT-B/14, quota12+4 and normalized last-four CLS concatenation (3072 dimensions)**. The historical uniform-sampler MoCo implementation was recovered and its three M0 students were reproduced exactly; it is distinct from the newly fixed quota4/CLS4 recipe. The old checkpoint basename `simsiam_dinov2_backbone_final.pt` is a legacy filename, not evidence that the training method was SimSiam. Byte-identical, versioned source copies are now at `scripts/training_moco/moco_v3_20261029_runtime.py` (SHA-256 `37fb14ac0883316d632d53ce0be9c3b8bacdbdb153f823df1bec986b248e1cf9`) and `scripts/training_moco/moco_v3_20261029_helpers.py` (SHA-256 `adee5b1583b11b251156a57694761a58c5ca1b3c98ea2e2452371ccd374083ee`).

The seed-42 raw student checkpoint SHA-256 is `fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e`; the recorded seed-43 and seed-44 prefixes are `b48bd…` and `878d…`. The archived source-recovery note records a three-check bitwise parity against M0 v4. Separately, the v5 run summary reports all 16 jobs complete and six of six student parity replays passing. This is reproduction/parity evidence, not independent clinical validation.

The recoverable run folders use names such as `runs/moco-v4-quota4-s42-full-20261107/`, despite the legacy checkpoint filename. Versioned runtime/helper copies are `scripts/training_moco/moco_v4_20261107_runtime.py` (SHA-256 `389221fa30e7bef2f35049ee1c133182320ffcd3a92db2289941576c6bad711b`) and `scripts/training_moco/moco_v4_20261107_helpers.py` (SHA-256 `5351f5d4cb473a07ddc83ab486e53b85897248aa81024709611c224181f4a3c7`). The exact quota4 seed-42 template is `scripts/training_moco/quota4-seed42.template.json` (SHA-256 `f79dbccba2b3f6955edd4d6202dc129e3c9bfbb0278e00ca47fb52adb83f2afc`); its source manifest SHA-256 is `afcfab263229f8bf08ffb412344e49a70d650f880b518d88e1078a8d1201eabb`. Its `${MIXED_FIT_JSONL}`, `${PUBLIC_WEIGHTS}`, and `${NEW_OUTPUT_DIR}` placeholders require manual expansion. Reproducing against another host requires validating that host's manifest and recording a new review if its digest differs; these files do not promise universal bitwise parity. E01's quota4 training protocol is not identical to the historical uniform-sampler M0 sweep; do not relabel all M0 results as quota4 or infer tonal-treatment parity from classification parity. No new SSL training was run for this release.

## Effective final quota4 training recipe

- **Backbone:** DINOv2 ViT-B/14 initialized from its public self-supervised DINOv2 checkpoint, not supervised ImageNet initialization; the source checkpoint SHA-256 is `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
- **Batch:** 16 images, with a quota of 12 source images plus 4 PAD-UFES-20 images. The recorded unlabeled pool contains 33,620 source images plus all 1,134 PAD development images (682 patient/component groups).
- **Views and transforms:** two independent random-resized crops at 224×224 with scale 0.4–1.0, horizontal flip, and ImageNet normalization. The training open-RGB path did not apply EXIF orientation correction; this differs from the evaluation/inference preprocessing below.
- **Objective:** symmetric in-batch InfoNCE at temperature 0.2, without a memory queue. The EMA teacher uses decay 0.99 for backbone/projector parameters; the predictor is student-only.
- **Optimization:** AdamW, backbone LR 1e-5, projector/predictor LR 1e-4, weight decay 0.05, with no extra scheduler or warmup recorded.
- **Not part of this recipe:** color-constancy or hue transforms, local crops, SimSiam anchors, L2-SP, extra consistency losses, or integrated tonal-treatment hooks.

## Readout/evaluation recipe

For evaluation, images were EXIF-transposed, converted to RGB, directly resized to 224×224 with bicubic interpolation, and ImageNet-normalized. The frozen feature is the concatenation of normalized CLS tokens from blocks 8, 9, 10, and 11 (3,072 values), followed by per-image L2 normalization. Blocks 0–7 and patch/position/CLS tokens stay frozen; blocks 8–11 and the final normalization are trainable in SSL. The reader is a balanced logistic regression.

The readout uses nested grouped five-outer/three-inner-fold OOF evaluation with three reader-CV seeds (`20261025`, `20261026`, `20261027`); the C grid is 0.3, 1, 3, 10, 30, and 100, with a smaller-C tie rule. PAD6 and HAM7 remain separate. Their development supports are 1,134 images / 682 groups and 1,528 images / 1,152 groups, respectively. PAD is transductive and HAM development results were exposed to selection. The sealed HAM holdout and PAD `target_test` are excluded.

The seed-42 quota4 CLS4 OOF readout is reported as 0.673685 macro-F1 on PAD and 0.634424 on HAM. These are **one encoder's OOF readout scores**, not the three-seed mean in [the V10 tables](../resultados/clasificacion/v10/README.md), not an ensemble, and not a score for a globally fitted deployment head.

## Final development readers (fit complete; no independent scores)

The final PAD6 and HAM7 readers were CPU-fit from BF16 CLS4 features of the fixed seed-42 encoder. Grouped three-fold CV over seeds `20261025`, `20261026`, and `20261027` selected C from 0.3, 1, 3, 10, 30, and 100 using macro-F1 with the smaller-C tie break. Both heads selected C = 30 and were refit on all authorized development data. HAM7 used 1,528 images / 1,152 groups and 143 solver iterations; PAD6 used 1,134 images / 682 groups and 144 iterations. The fitted heads have no independent evaluation score in this evidence; do not present fit metrics as validation.

The complete model was built from source commit `7d6622eebfff388d39f6adf98c6f2542cb3db4c2`. Complete `classifier.pt` SHA-256: `1eae148f602f18675ea4950a21eeea0841a3c52c8049a31be9d876719ac4a08e`; dual-output `model.onnx`: `eb41e67f472a1491a6970246e59f6db8c0276c79fdd190c68fb25e3b3f80dd7a`; bundle manifest: `7e54de3b0aac476ca7e55c029f99547c55c59f2c49fe53391b390d8213fb8664`. Both output heads share one encoder; there is no ensemble.

Native Hub FP32 → portable PyTorch/checkpoint reload → ONNX passed the fixed `1e-4` probability gate on synthetic fixtures and batch sizes1/2/3, with maximum difference about`1.26e-6`. A clean PyTorch CPU environment passed12runtime checks; Docker passed12checks with network disabled and read-only runtime. These are software parity checks, not diagnostic validation. BF16 research→FP32 serving drift was measured separately on eight fixed allowed samples per cohort: max probability difference `.01138` HAM and `.002368` PAD, with zero argmax changes in those samples only. No full-cohort equivalence is implied. API publication/deployment evidence is recorded separately when complete.

## Scope boundary

The recovered recipe does not complete tonal-treatment integration E06–E08 and does not execute T0, M1, or M1−T0. Future tonal M0 must use the newly frozen quota4/CLS4 recipe without tone treatment, not the historical uniform-sampler sweep M0. Classification release parity is not tonal parity. No clinical-performance or fairness claim follows from these development readouts.

Quota4 training receipts measured `0.4490`, `0.5702`, and `0.4638` loop wall-hours for seeds42/43/44, with peak allocated GPU memory `2.164GiB`. This includes I/O and is not kernel-only utilization or the future integrated tonal cost. The cluster has two RTXPRO4000 24GB GPUs; the campaign used independent one-GPU jobs concurrently when free. The source/PAD batch quota is not a GPU reservation. Check `nvidia-smi` and coordinate the window before Pau's runs; no future capacity or durable reservation is guaranteed.
