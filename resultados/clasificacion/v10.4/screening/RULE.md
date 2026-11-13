# Melanoma-priority operating point for the v10.4.0 HAM7 head: selection rule

Declared on 2026-11-13 before computing any value. Same mechanism and rule as
`resultados/clasificacion/v10/screening/RULE.md`; only the data source changes, because the v10.4.0
reader is fit on pooled rows and has no out-of-fold development predictions.

- Head: v10.4.0 HAM7 multi-domain reader (`heads/HAM-reader.json` 6765f307…), target `mel`.
- Data: `milk_dev` (MILK10k lesions with SHA256(lesion_id) mod 10 in {6, 7}), clinical and
  dermoscopic rows together. These rows were never used to fit the reader; they were used once to
  pick the reader spec.
- Mechanism: `screening pred = argmax_k (log p_k + delta * [k == mel])`; `pred` is unchanged.
- Grid: delta in {0.00, 0.25, ..., 3.00}. Selected delta = largest value with macro-F1 (seven HAM7
  classes) >= macro-F1 at delta = 0 minus 0.01.
- Unbiased estimate: 5 folds grouped by lesion inside `milk_dev` (seed 20261113); delta is selected
  on four folds and applied to the fifth.
- Test sets (`milk_test`, `pad_test`, `ham_holdout`, `ddi`, `fitz`, `bosque`, `internet`) are
  report-only at the selected delta.
- PAD6 head: unchanged, so its v10.1.0 offset (0.5) is kept.
