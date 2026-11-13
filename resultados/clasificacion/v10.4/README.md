# v10.4.0: multi-domain HAM7 reader against nevus collapse

## Problem

The v10.1.0 HAM7 head (the app default) was fit on 1,528 dermoscopic `ham_dev` rows only. On
clinical photos (phone or web) it predicts nevus for most lesions: on DDI it predicts nevus for 90%
of non-nevus lesions and 95% of melanomas; on the reviewed Wikimedia set the production API
returned nevus for 55 of 76 images (only 15 are nevi). The encoder still separates malignant from
benign (DDI malignant AUC 0.77), so the failure is in the single-domain reader, not in the
representation.

## Change

- Encoder: unchanged (MoCo V10 s42, `fe646105…`), 448 px, CLS4 with D4 mean. No supervised
  fine-tuning; labels enter only the linear reader.
- HAM7 reader: balanced logistic regression on pooled CLS4-D4 features of `ham_fit` + `ham_dev`
  (HAM10000), ISIC 2019 non-HAM rows, PAD-UFES-20 `pad_adapt` and the MILK10k train split (60% of
  lesions by SHA256 of `lesion_id`). Sample weights equalise every (modality, class) cell; clinical
  cells count twice. C = 3. Label maps: ACK/SCC → akiec, SEK/BKL → bkl.
- Selection: fixed grid, rule `max milk_dev clinical bacc + 0.5 × milk_dev dermoscopic bacc`, on
  the MILK10k dev split only. Every other set below is report-only.
- PAD6 head: unchanged.

## External results (generated: `EXTERNAL.md`, from `study-final.json`)

See `EXTERNAL.md`. Offline numbers use BF16-extracted features; the served FP32 graph differs by at
most 0.018 in probability on the checked images, with the same argmax.

## Negative results

- Continued SSL with a larger clinical bucket (PAD + MILK10k clinical train/dev, quota 4 or 8,
  6,000 steps, same recipe as the shipped encoder): no consistent gain over the shipped encoder
  (`SSL-ARMS.md`).
- Per-modality feature centering, even with the true modality: no gain (`scripts/research/nevus_fix/center_test.py`).

## Limits

- DDI stays weak (bacc 0.35): many DDI diagnoses fall outside the seven classes and skin tones V-VI
  are rare in the training rows.
- The screening (melanoma-priority) policy was selected for the v10.1.0 head and is reported as
  unavailable for v10.4.0 until it is re-selected.
- HAM holdout and PAD test were already opened for v10.1.0; they are reported, not used for
  selection. Research prototype, not a medical device.
