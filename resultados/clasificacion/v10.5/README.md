# v10.5.0: single-view classifier (8× less compute)

## Problem

v10.4.0 encoded every photo under the 8 dihedral views (D4) and averaged the features: 8 ViT-B/14
passes at 448 px. On the 2-vCPU production host one classification took about 17 s, almost all of
the `/api/analyze` time.

## Change

- Encoder, preprocessing, PAD6 head and segmenter: unchanged (same frozen MoCo V10 s42 encoder).
- Recipe: one 448 px view (`tta: none`, `views: 1`), bundle schema `dermssl.moco-bundle.v3`,
  checkpoint schema `dermssl.moco-classifier.v3` (`src/dermssl/release/moco_single.py`).
- HAM7 reader: the v10.4.0 multi-domain recipe (`cell:2:3`) refit on single-view CLS4 features
  (`scripts/research/nevus_fix/views_refit.py`). Labels still enter only this linear reader.
- Melanoma-priority offset re-selected on milk_dev under a rule declared before any value was
  computed (`screening/RULE.md`): 0.25, picked by all five nested folds.

## Decision

Tables in `RESULTS.md` (generated from `evidence/`).

- On FP32 features, as served, the single view loses no accuracy that the data can detect: the
  milk_dev balanced accuracy difference is −0.007 [−0.024, +0.006] and no test set shows a
  significant change. Melanoma recall rises on PAD-UFES-20, BOSQUE and DDI. The 8-view recipe keeps
  a slightly higher milk_dev score (0.978 vs 0.965); the change is justified by speed, not by a
  measured accuracy gain, and the non-inferiority reading was not declared in advance.
- Verified on the production host before the cutover: 31/31 checks, `/api/analyze` p50 2.35 s
  (17.7 s for v10.4.0 `/api/predict` on the same host).
- Production check after the cutover: the 76 reviewed web photos through the public API
  (`evidence/internet-api-e2e-v10.{4,5}.jsonl`, same images and script as v10.4.0).
- Rejected: INT8 dynamic quantisation (2.1× faster on the host, but HAM7 argmax agreement with
  the reference fell to 25–45%); adding a 70% centre crop to the single view (two passes; better on
  web photos, significantly worse on MILK10k test, tie on milk_dev).

## Limits

- DDI stays weak (balanced accuracy 0.32 on the mapped subset); see `../v10.4/README.md`.
- The study features were extracted in bf16 and the reader was fit on them; the served graph runs
  in FP32 (single-image probability drift up to 0.042, argmax agreement 0.96–0.99).
- Research prototype, not a medical device.
