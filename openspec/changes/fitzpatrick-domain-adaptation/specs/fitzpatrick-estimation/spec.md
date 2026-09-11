# Fitzpatrick Estimation Specification

## Purpose

Mask-guided, ITA-based estimation of Fitzpatrick Skin Type (FST) on skin
pixels only, so every HAM10000/ISIC2019 image carries an FST bucket (or an
explicit "unavailable" marker) usable by fairness evaluation (Stage 1).

## Requirements

### Requirement: Lesion Mask Generation

The system MUST generate a binary lesion mask per input image using a U-Net
trained on ISIC 2018 Task 1 boundary masks (own held-out split).

#### Scenario: Valid image produces a usable mask

- GIVEN a dermoscopic image from HAM10000 or ISIC2019
- WHEN the U-Net runs inference
- THEN it returns a binary mask covering the lesion region
- AND the mask is passed to skin-pixel selection

### Requirement: Skin Pixel Selection

The system MUST derive the skin-pixel region by excluding the lesion mask
(and image border/vignette) from the full image before any ITA computation.

#### Scenario: Lesion excluded from skin sampling

- GIVEN a valid lesion mask
- WHEN skin pixels are selected
- THEN pixels inside the lesion mask are excluded
- AND only remaining skin-labeled pixels feed ITA computation

### Requirement: ITA Computation and FST Bucketing

The system MUST convert selected skin pixels to LAB colorspace, compute
`ITA = arctan((L*-50)/b*) * 180/pi` as an aggregate (median) over skin
pixels, and MUST map the aggregate ITA to an FST bucket (1-6) using Groh et
al. empirical thresholds.

#### Scenario: Clean image yields a bucket

- GIVEN skin pixels with a computable median ITA
- WHEN the ITA-to-FST mapping is applied
- THEN the system returns exactly one FST bucket in {1..6}

### Requirement: Explicit Bucket-or-Unavailable Output

The system MUST NOT silently drop an image. Every processed image MUST
receive either an FST bucket (1-6) or an explicit `"unavailable"` marker
with a reason code.

#### Scenario: Mask generation fails or confidence is too low

- GIVEN the U-Net produces an empty mask or below-threshold confidence
- WHEN estimation runs
- THEN the system applies a documented fallback skin-region heuristic
- AND marks the result with a `fallback_used` flag and lower confidence
- AND still returns a bucket or `"unavailable"`, never a silent skip

#### Scenario: Dermoscopy artifacts confuse estimation

- GIVEN the image contains hair, ruler markings, or ink borders known to
  bias ITA (per Pakzad et al.)
- WHEN estimation runs
- THEN the system flags the result with an `artifact_detected` indicator
- AND still returns a bucket estimate with reduced confidence, not a hard
  failure

### Requirement: Optional BOSQUE Validation

The system MAY compare ITA-estimated FST buckets against BOSQUE-annotated
ground truth when the dataset is available, reporting agreement metrics
without blocking the estimation pipeline if BOSQUE is unavailable.

#### Scenario: BOSQUE unavailable

- GIVEN BOSQUE licensing/access has not landed
- WHEN the estimation pipeline runs
- THEN estimation proceeds and reports buckets normally
- AND no validation step blocks or fails the run
