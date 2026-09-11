# CIRCLe Adaptation Specification

## Purpose

Stage 2 mitigation: a CIRCLe-style (Pakzad et al., arXiv:2208.13528) LAB
color transformer, joint regularization, and synthetic blend applied at the
trainable-probe level only, gated on Stage 1 confirming measurable bias.

## Requirements

### Requirement: Stage-Gate Dependency on Stage 1

Stage 2 execution MUST depend on Stage 1's fairness gate reporting
measurable bias (`FAIL`, or a gap above a configurable "worth mitigating"
threshold). Stage 2 MUST be skippable (not required to run) when Stage 1
reports `PASS` or `SKIP`.

#### Scenario: Stage 1 shows no measurable bias

- GIVEN Stage 1's fairness gate reports `PASS` or `SKIP`
- WHEN Stage 2 is invoked
- THEN Stage 2 execution is skippable and documented as such, not forced

### Requirement: Deterministic LAB Color Transformer with Lesion Preservation

The color transformer MUST take an image, a target FST shift, and the
lesion mask (from fitzpatrick-estimation), MUST apply the color shift only
to skin pixels (mask-protected lesion region unchanged), and MUST produce a
valid image (no clipped/out-of-range LAB values).

#### Scenario: Valid transform with mask

- GIVEN an image, a valid lesion mask, and a target FST shift
- WHEN the transformer runs
- THEN the lesion region pixels are unchanged
- AND skin pixels reflect the target FST color shift
- AND output pixel values are within valid LAB/RGB range

#### Scenario: Missing or invalid mask input

- GIVEN the lesion mask is missing or invalid for the input image
- WHEN the transformer is invoked
- THEN it MUST raise an explicit error or return a flagged no-op result
- AND MUST NOT silently apply an unprotected color shift over the lesion

### Requirement: Joint Regularization at the Probe Level

Training MUST add an MSE regularization term between the probe's
prediction/feature on the original image and its FST-shifted counterpart,
applied only to the trainable probe head. The backbone MUST remain frozen
(no weight updates) throughout Stage 2 training.

#### Scenario: Backbone frozen invariant holds

- GIVEN a Stage 2 training step with joint regularization enabled
- WHEN the step completes
- THEN backbone parameters have `requires_grad=False` and their weights are
  unchanged before/after the step
- AND only probe-head parameters receive updates

### Requirement: Synthetic Blend with Bounded FST Shift

Synthetic-blend augmentation MUST mix real and FST-shifted synthetic images
at a configurable target ratio (default ~50/50) and MUST NOT allow a
synthetic shift of more than 2 FST levels from the source image's estimated
bucket.

#### Scenario: Requested shift within bound

- GIVEN a source image at FST bucket 2 and a requested shift of 2 levels
- WHEN synthetic blend generates the sample
- THEN the resulting synthetic target bucket is FST 4

#### Scenario: Requested shift exceeds bound

- GIVEN a source image at FST bucket 2 and a requested shift of 3 levels
- WHEN synthetic blend generates the sample
- THEN the system MUST clamp the shift to the maximum allowed (2 levels) or
  reject the request, and MUST NOT silently apply the larger shift

### Requirement: Baseline vs. Mitigation Comparison Reporting

Stage 2 evaluation MUST report Stage 1 stratified fairness metrics for
three variants — baseline, regularization-only, and synthetic-blend — so
the relative fairness improvement (primary thesis acceptance criterion) is
directly computable between variants.

#### Scenario: Comparison report produced

- GIVEN baseline, regularization-only, and synthetic-blend evaluation runs
  have completed using `metrics.stratified_summary()`
- WHEN the comparison report is generated
- THEN it lists the fairness gap for each variant
- AND states the relative improvement of each mitigation variant over the
  baseline
