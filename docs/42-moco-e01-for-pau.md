# 42. E01 historical MoCo run: recovered recipe and evidence

## Identity and recovery

E01 is a historical **MoCo-v3-style** run on DINOv2 ViT-B/14. Its old checkpoint basename `simsiam_dinov2_backbone_final.pt` is a legacy filename, not evidence that the training method was SimSiam. The recovered source paths were `scripts/arms/moco_v3_20261029_runtime.py` and `scripts/arms/moco_v3_20261029_helpers.py`; the recorded SHA-256 prefixes are `37fb14ac…` and `adee5b…`, respectively. Do not expand these prefixes into full hashes without the source manifest.

The seed-42 raw student checkpoint SHA-256 is `fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e`; the recorded seed-43 and seed-44 prefixes are `b48bd…` and `878d…`. The archived source-recovery note records a three-check bitwise parity against M0 v4. Separately, the v5 run summary reports all 16 jobs complete and six of six student parity replays passing. This is reproduction/parity evidence, not independent clinical validation.

The recoverable run folders use names such as `runs/moco-v4-quota4-s42-full-20261107/`, despite the legacy checkpoint filename. E01's quota4 training protocol is not identical to the historical uniform-sampler M0 sweep; do not relabel all M0 results as quota4 or infer tonal-treatment parity from classification parity.

## Historical training recipe

- **Backbone:** DINOv2 ViT-B/14 initialized from its public ImageNet checkpoint; the source checkpoint SHA-256 is `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
- **Batch:** 16 images, with a quota of 12 source images plus 4 PAD-UFES-20 images. The recorded unlabeled pool contains 33,620 source images plus all 1,134 PAD development images (682 patient/component groups).
- **Views and transforms:** two independent random-resized crops at 224×224 with scale 0.4–1.0, horizontal flip, and ImageNet normalization. The training open-RGB path did not apply EXIF orientation correction; this differs from the evaluation/inference preprocessing below.
- **Objective:** symmetric in-batch InfoNCE at temperature 0.2, without a memory queue. The EMA teacher uses decay 0.99 for backbone/projector parameters; the predictor is student-only.
- **Optimization:** AdamW, backbone LR 1e-5, projector/predictor LR 1e-4, weight decay 0.05, with no extra scheduler or warmup recorded.
- **Not part of this recipe:** color-constancy or hue transforms, local crops, SimSiam anchors, L2-SP, extra consistency losses, or integrated tonal-treatment hooks.

## Readout/evaluation recipe

For evaluation, images were EXIF-transposed, converted to RGB, directly resized to 224×224 with bicubic interpolation, and ImageNet-normalized. The frozen feature is the concatenation of normalized CLS tokens from blocks 8, 9, 10, and 11 (3,072 values), followed by per-image L2 normalization. Blocks 0–7 and patch/position/CLS tokens stay frozen; blocks 8–11 and the final normalization are trainable in SSL. The reader is a balanced logistic regression.

The readout uses grouped five-fold OOF evaluation with three reader-CV seeds (`20261025`, `20261026`, `20261027`); the C grid is 0.3, 1, 3, 10, 30, and 100, with a smaller-C tie rule. PAD6 and HAM7 remain separate. Their development supports are 1,134 images / 682 groups and 1,528 images / 1,152 groups, respectively. PAD is transductive and HAM development results were exposed to selection. The sealed HAM holdout and PAD `target_test` are excluded.

The seed-42 quota4 CLS4 OOF readout is reported as 0.673685 macro-F1 on PAD and 0.634424 on HAM. These are **one encoder's OOF readout scores**, not the three-seed mean in [the V10 tables](../resultados/clasificacion/v10/README.md), not an ensemble, and not a score for a globally fitted deployment head.

## Scope boundary

The recovered E01 run does not complete tonal-treatment experiments E06–E08 and does not execute T0, M1, or M1−T0. Classification release parity is not tonal parity. No clinical-performance, fairness, or deployment claim follows from these development readouts. GPU availability is an observation, not a reservation; this OOF/summary evidence does not establish a numeric GPU wall-time or guarantee future capacity.
