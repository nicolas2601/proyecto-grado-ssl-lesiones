"""Provenance and pairing guards for the C1 cells.

Split out of :mod:`dermssl.mask_conditions` after review: the geometry half is
pure image maths, this half is the part that decides whether a result is
admissible at all. Keeping them apart makes it obvious which rules are
arithmetic and which are policy.

Two hazards live here. The first is the vgg19 class of bug -- a probe scoring
features it was not fitted on returns a plausible number with no error. The
second is subtler: C1 is only self-supervised if its masks are, so a mask has
to arrive with the manifest of the step-0 artifact and a supervised producer
cannot be constructed at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from dermssl.c1_mask_export import is_sha256
from dermssl.mask_conditions import CONDITIONS
from dermssl.probes import logreg_predict_proba

SPLITS = ("train", "val")

# A tolerance looser than this is not a gate. The demo's cross-device drift was
# 7.85e-05, so 1e-3 already leaves an order of magnitude of headroom; anything
# above it would wave through the very defects this check exists to catch.
MAX_TOLERANCE = 1e-3

@dataclass
class FallbackTally:
    """Counts CLS fallbacks in condition (c).

    Reported per split x backbone x condition. A high rate means the mask
    selected no patch for many images, so (c) quietly degenerated into (0) and
    the row is no longer measuring what its name says.
    """

    WARN_ABOVE = 0.01

    n_cls_fallback: int = 0
    n_total: int = 0

    def record(self, used_fallback: bool) -> None:
        self.n_total += 1
        self.n_cls_fallback += int(used_fallback)

    @property
    def rate(self) -> float:
        return self.n_cls_fallback / self.n_total if self.n_total else 0.0

    def summary(self) -> dict[str, float | int]:
        return {
            "n_cls_fallback": self.n_cls_fallback,
            "n_total": self.n_total,
            "cls_fallback_rate": self.rate,
            "warning": bool(self.rate > self.WARN_ABOVE),
        }


@dataclass(frozen=True)
class ConditionedFeatures:
    """Features bound to the cell and the mask artifact that produced them.

    This exists because of the vgg19 incident: identical tensors extracted at a
    different point produce a plausible number with no error. Carrying the
    provenance on the array itself is what lets :func:`fit_and_score` refuse a
    mismatch instead of returning a result nobody can tell is wrong.

    ``image_ids`` and ``masks_sha256`` are required, not optional: a review
    showed that a tag alone is falsifiable, and that ``split`` was being stored
    and never checked.
    """

    condition: str
    backbone: str
    split: str
    features: np.ndarray
    labels: np.ndarray
    image_ids: tuple[str, ...]
    masks_sha256: str

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; expected one of {CONDITIONS}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; expected one of {SPLITS}")
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"{len(self.features)} feature rows against {len(self.labels)} labels"
            )
        if len(self.image_ids) != len(self.features):
            raise ValueError(
                f"{len(self.image_ids)} image_ids against {len(self.features)} feature rows"
            )
        if not is_sha256(self.masks_sha256):
            raise ValueError(f"masks_sha256 is not a sha256 digest: {self.masks_sha256!r}")


def fit_and_score(
    train: ConditionedFeatures,
    val: ConditionedFeatures,
    n_classes: int,
    seed: int = 42,
) -> np.ndarray:
    """Fit the headline ``logreg`` probe on train and score val, same cell only.

    Each of the 24 cells (6 backbones x 4 conditions) owns its probe. Crossing
    a probe from one cell with features from another is the failure this guard
    exists to make impossible.
    """
    if train.split != "train" or val.split != "val":
        raise ValueError(
            f"split roles are wrong: got train={train.split!r}, val={val.split!r}. "
            "Fitting on the validation split inflates every metric in the cell."
        )
    if train.condition != val.condition:
        raise ValueError(
            f"condition mismatch: probe fitted on {train.condition!r} cannot score "
            f"{val.condition!r} features"
        )
    if train.backbone != val.backbone:
        raise ValueError(
            f"backbone mismatch: probe fitted on {train.backbone!r} cannot score "
            f"{val.backbone!r} features"
        )
    if train.masks_sha256 != val.masks_sha256:
        raise ValueError(
            f"masks_sha256 mismatch: train used {train.masks_sha256[:12]}..., val used "
            f"{val.masks_sha256[:12]}.... Both halves of a cell must come from the same "
            "mask artifact or the comparison spans two different inputs."
        )
    overlap = set(train.image_ids) & set(val.image_ids)
    if overlap:
        raise ValueError(
            f"train/val overlap leaks {len(overlap)} image(s), e.g. {sorted(overlap)[:3]}. "
            "The probe would be scored on images it was fitted on."
        )
    if train.features.shape[1] != val.features.shape[1]:
        raise ValueError(
            f"feature dim mismatch: train {train.features.shape[1]} vs val "
            f"{val.features.shape[1]}"
        )

    return logreg_predict_proba(
        train.features,
        train.labels,
        val.features,
        n_classes=n_classes,
        seed=seed,
    )


def compare_to_bank(
    bank_features: np.ndarray,
    c1_features: np.ndarray,
    tolerance: float,
) -> dict:
    """Does condition (0) reproduce the published bank cache?

    This is the gate that catches an extraction drift at the root rather than
    by symptom. Condition (0) is, by definition, the bank's own pipeline with
    no mask applied -- so if its features do not land on the cached ones, the
    extraction path has changed and every C1 comparison is being made against
    a baseline that is not the published one. The nearest-vs-bilinear defect
    found in review would have been caught here on the first run.

    ``tolerance`` is an absolute bound on the feature values. 1e-4 is the
    working default: re-extracting on a different device moved the demo's
    probabilities by 7.85e-05, so anything above that is drift, not hardware.
    """
    if not math.isfinite(tolerance):
        raise ValueError(
            f"tolerance must be finite, got {tolerance!r}. A NaN tolerance turns the "
            "gate into a no-op: every comparison `diff > nan` is False, so any drift "
            "reports as reproducing."
        )
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(
            f"tolerance {tolerance!r} is too loose; the maximum is {MAX_TOLERANCE:g}. "
            "A gate that accepts anything is not a gate."
        )

    if bank_features.shape != c1_features.shape:
        raise ValueError(
            f"shape mismatch: bank {bank_features.shape} vs C1 {c1_features.shape}; "
            "these are not the same images or not the same backbone"
        )

    per_row = np.abs(bank_features.astype(np.float64) - c1_features.astype(np.float64)).max(axis=1)
    # NaN compares False against everything, so an explicit check is required or
    # a corrupted feature row would report as reproducing.
    over = ~(per_row <= tolerance)
    worst_row = int(np.argmax(per_row))

    return {
        "reproduces": bool(not over.any()),
        "tolerance": float(tolerance),
        "n_rows": int(len(per_row)),
        "n_rows_over_tolerance": int(over.sum()),
        "max_abs_diff": float(per_row.max()),
        "mean_abs_diff": float(per_row.mean()),
        "worst_row": worst_row,
    }
