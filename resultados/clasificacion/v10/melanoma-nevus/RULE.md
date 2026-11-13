# Melanoma vs nevus separation: candidates and decision rule

Declared on 2026-11-12, before any candidate below is computed. This file is committed and pushed
before the analysis runs; the commit timestamp is the record.

## Problem

The shipped reader (v10.1.0: MoCo V10 encoder, 448 px, CLS4 with D4 TTA, balanced logistic
reader, C = 100) confuses melanoma with nevus: on development out-of-fold data, 22% of melanomas
are predicted as nevus and the one-vs-rest melanoma AUC is about 0.856. The melanoma-priority
reading (v10.3.0) only moves the operating point. This study looks for better separation.

## Constraints

- Encoder frozen. Labels enter only the reader. No retraining of the encoder.
- Selection uses development rows only (`ham_dev`, n = 1528). The sealed sets are not read.
- Metadata, when used, is limited to age, sex and anatomical site. `dx_type` is excluded (it is a
  proxy of the label).

## Endpoints (development, out of fold)

Grouped `StratifiedGroupKFold(5, shuffle=True)` over identity groups, CV seeds 20261025, 20261026,
20261027; encoder seeds 42, 43, 44 (except where a candidate pools encoders).

- Primary: melanoma-vs-nevus AUC on rows whose label is `mel` or `nv`, with the score
  `p(mel) / (p(mel) + p(nv))`.
- Secondary: melanoma recall at nevus recall 0.85 on those rows; seven-class macro-F1 at argmax;
  melanoma-to-nevus rate at argmax.

## Candidates (fixed list)

- C0, shipped: balanced logistic reader on CLS4-D4, C = 100.
- C1, melanoma-nevus specialist: C0 plus a binary balanced logistic reader trained only on `mel`
  and `nv` rows (C chosen by inner grouped CV from {1, 10, 100}); the mass `p(mel) + p(nv)` of C0
  is split by the specialist's posterior.
- C2, nevus undersampling: ten C0 readers, each fit on all non-nevus rows plus a random subset of
  nevus rows the size of the melanoma rows; probabilities averaged.
- C3, patch tokens: C0 recipe on CLS4 plus mean patch tokens (448 px, single view).
- C4, encoder ensemble: mean of the C0 probabilities of encoders 42, 43 and 44.
- C5, metadata: C0 features concatenated with standardized age, sex and anatomical site
  (one-hot), scaled by a weight chosen by inner grouped CV from {0.1, 0.3, 1.0}; missing age is
  imputed with the training-fold median plus a missing indicator.
- C6, lesion view: C0 features concatenated with CLS4-D4 features of a crop around the lesion
  (bounding box of the v10.2.0 segmenter mask, 20% margin, resized to 448 px). Run only if the
  crop features can be extracted with the shipped encoders.

Combinations: if two or more candidates pass, their combination is tested under the same rule.

## Decision rule

A candidate passes if, against C0 on the same encoder seed (pooled seeds for C4):

1. mean melanoma-vs-nevus AUC improves by at least 0.010 and improves in every encoder seed, and
2. seven-class macro-F1 drops by no more than 0.010.

## External confirmation

Passing candidates and C0 are evaluated once on an external dermoscopic set never used for
selection or training of any reader: ISIC 2020 (all `melanoma` images and a seed-42 random sample
of three `nevus` images per melanoma). Reported: melanoma-vs-nevus AUC with a 2000-sample
bootstrap 95% interval over patients. A candidate is confirmed if its external AUC is higher than
C0's and the paired bootstrap interval of the difference excludes zero.
