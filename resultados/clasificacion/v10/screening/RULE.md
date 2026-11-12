# Melanoma-priority operating point: selection rule

Declared on 2026-11-12, before computing any value for this rule. This file is committed before the
analysis script runs; the commit timestamp is the record.

## Scope

- Classifier: MoCo V10, 448 px, CLS4 with D4 test-time augmentation (v10.1.0 bytes, unchanged).
- Heads: HAM7 (target class `mel`) and PAD6 (target class `MEL`).
- Data: development rows only (`ham_dev`, `pad_adapt`). The sealed sets (`ham_holdout`, `pad_test`)
  were consumed by the v10.1.0 evaluation and are not read.
- No retraining. The encoder and the shipped reader weights stay byte-identical.

## Mechanism

The shipped prediction (`pred`, argmax of the reader probabilities) is not changed. A second
operating point is reported next to it:

    screening score_k = log p_k + delta * [k == target]
    screening pred    = argmax_k screening score_k

`delta >= 0` is a single scalar per head. With `delta = 0` the screening prediction equals `pred`.

## Out-of-fold probabilities

For each encoder seed (42, 43, 44) and each CV seed (20261025, 20261026, 20261027):
`StratifiedGroupKFold(5, shuffle=True, random_state=cv_seed)` over identity groups; a balanced
logistic reader with the shipped recipe (per-image L2, C = 100, lbfgs, max_iter 2000) is fit on
four folds and predicts the fifth.

## Selection rule

- Grid: `delta` in {0.00, 0.25, ..., 3.00}.
- For a set of OOF runs, `F(delta)` is the mean macro-F1 over the runs and `F(0)` the mean macro-F1
  of the shipped argmax.
- Selected `delta` = the largest grid value with `F(delta) >= F(0) - 0.01`.
- The shipped value is selected on the three OOF runs of encoder seed 42 (the shipped encoder).
  Encoder seeds 43 and 44 are reported as a robustness check, not used for selection.

## Unbiased estimate

The selection is repeated inside a nested split: for every run, the OOF rows are split with
`StratifiedGroupKFold(5, shuffle=True, random_state=cv_seed + 1000)`; `delta` is selected on four
folds by the same rule and applied to the fifth. The reported screening metrics (target recall,
target-to-nevus rate, nevus recall, target precision, macro-F1, alert rate) are these nested
held-out values, averaged over runs.

## What this does not claim

- No sealed-test number exists for this operating point. The sealed v10.1.0 metrics apply only to
  `pred`.
- The scores are not calibrated clinical probabilities.
