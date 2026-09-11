# Fairness Evaluation Specification

## Purpose

FST-stratified performance reporting (`metrics.stratified_summary()`) and a
new fairness acceptance gate in `gates.py`, consistent with the existing
G0-G2 pattern, that never fails on statistically thin buckets.

## Requirements

### Requirement: `stratified_summary()` Contract

The system MUST expose `metrics.stratified_summary()` accepting per-example
predictions, labels, and FST bucket assignments (from
fitzpatrick-estimation), and MUST return per-bucket performance metrics
(the same metrics already reported in the non-stratified summary) keyed by
bucket id 1-6 plus a separate `"unavailable"` count.

#### Scenario: Well-populated buckets

- GIVEN predictions/labels/buckets for a dataset with samples across all
  FST buckets
- WHEN `stratified_summary()` runs
- THEN it returns one metrics entry per bucket with sample counts included

### Requirement: Insufficient-Data Handling per Bucket

The system MUST mark any bucket with fewer samples than a configurable
`min_n` as `SKIPPED` in the summary output, MUST NOT compute point metrics
for skipped buckets, and MUST include the skip reason and observed count.

#### Scenario: Sparse FST 5-6 bucket

- GIVEN FST bucket 5 has fewer samples than `min_n`
- WHEN `stratified_summary()` runs
- THEN bucket 5's entry is `SKIPPED` with the sample count and threshold
  reported
- AND other buckets are computed normally

#### Scenario: Bucket is empty

- GIVEN FST bucket 6 has zero samples
- WHEN `stratified_summary()` runs
- THEN bucket 6 is `SKIPPED` with count 0, not omitted from the output
  structure

### Requirement: Fairness Gate SKIP vs FAIL Semantics

The system MUST add a fairness gate to `gates.py` following the existing
G0-G2 reporting pattern. The gate MUST compute a fairness metric (gap
across non-skipped buckets) and compare it to a configurable threshold
(default 0.15, per docs/04, docs/07, docs/11) to report `PASS` or `FAIL`.
The gate MUST report `SKIP` (never `FAIL`) when fewer than 2 buckets have
sufficient data to compute the gap.

#### Scenario: Enough buckets, gap within threshold

- GIVEN at least 2 buckets with sufficient data and gap <= threshold
- WHEN the fairness gate runs
- THEN it reports `PASS` with the observed gap and per-bucket breakdown

#### Scenario: Enough buckets, gap exceeds threshold

- GIVEN at least 2 buckets with sufficient data and gap > threshold
- WHEN the fairness gate runs
- THEN it reports `FAIL` with the observed gap, threshold used, and
  per-bucket breakdown

#### Scenario: Insufficient valid buckets overall

- GIVEN fewer than 2 buckets have `min_n` samples (e.g., only FST 1-3 are
  populated)
- WHEN the fairness gate runs
- THEN it reports `SKIP` with a warning naming the skipped buckets and
  reasons
- AND the gate MUST NOT report `FAIL` in this case

### Requirement: Configurable Threshold

The fairness gate threshold MUST be configurable (not hardcoded), since the
0.15 absolute-gap value is a secondary/reference criterion, not the primary
thesis acceptance criterion (relative improvement of Stage 2 over Stage 1).

#### Scenario: Threshold override

- GIVEN a caller supplies a custom threshold value
- WHEN the fairness gate runs
- THEN it uses the supplied threshold instead of the 0.15 default
