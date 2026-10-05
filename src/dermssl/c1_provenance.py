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

    **What it must be compared against.** The bank pipeline RECOMPUTED in the
    same process, at the same precision -- not the stored cache. The cache was
    written under ``torch.amp.autocast("cuda")``, so the bank does not
    reproduce its own cache below ~1.6e-02, and a tight bound against it fails
    for a reason that has nothing to do with C1's preprocessing. Comparing a
    recomputation isolates exactly the thing under test.

    ``tolerance`` is an absolute bound on the feature values;
    :data:`DEFAULT_PRIMARY_TOLERANCE` (1e-5) is the default, because two
    identical computations in one process should land on each other. The
    earlier 1e-4 was calibrated on the demo's PROBABILITIES re-extracted on
    another device (7.85e-05) -- a different quantity, measured a different
    way, then applied to features under mixed precision.

    For the recomputation-vs-cache comparison, use
    :func:`compare_to_published_cache`, which reports rather than decides.
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


#: Bound for the SECONDARY check (recomputed bank vs the stored cache). Loose
#: on purpose: the cache was written under ``torch.amp.autocast("cuda")`` and
#: the verification need not run at the same precision. Measured on the pod
#: 2026-09-26, recomputing the bank's own pipeline against its own cache gave
#: max_abs_diff 1.64e-02 with the cosine still at 1.000.
PUBLISHED_CACHE_MAX_ABS_DIFF = 5e-2

#: ...and the companion bound that a loose magnitude cannot provide. Precision
#: changes a vector's length, not its direction; a different image changes the
#: direction. Below this, the rows are not the same picture.
PUBLISHED_CACHE_MIN_COSINE = 0.999

#: Bound for the PRIMARY gate, which compares C1's condition (0) against the
#: bank pipeline recomputed IN THE SAME PROCESS at the SAME precision. Nothing
#: about mixed precision applies there, so the bound is what an identical
#: computation actually produces.
#:
#: The old 1e-4 default was calibrated on the demo's PROBABILITIES re-extracted
#: on another device (7.85e-05), not on features under autocast, and it was
#: applied against the stored cache. That made the gate unfailable for a reason
#: unrelated to C1: the bank does not reproduce its own cache below 1.6e-02.
DEFAULT_PRIMARY_TOLERANCE = 1e-5


def _validate_bound(name: str, value: float, *, upper: float | None = None) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f"{name} debe ser finito, recibi {value!r}. Un umbral NaN convierte el "
            "chequeo en un no-op: toda comparacion contra NaN es False."
        )
    if value <= 0:
        raise ValueError(f"{name} debe ser positivo, recibi {value!r}")
    if upper is not None and value > upper:
        raise ValueError(
            f"{name}={value!r} pasa de {upper!r}; un chequeo que acepta cualquier cosa "
            "no es un chequeo"
        )
    return float(value)


def compare_to_published_cache(
    recomputed: np.ndarray,
    cached: np.ndarray,
    *,
    max_abs_diff: float = PUBLISHED_CACHE_MAX_ABS_DIFF,
    min_cosine: float = PUBLISHED_CACHE_MIN_COSINE,
) -> dict:
    """Secondary check: does the recomputed bank still match the stored cache?

    This REPORTS; it does not decide on its own. Its failure means the cache
    was produced at a precision or on a device the verification is not
    reproducing, which has to be written down -- but it says nothing about
    whether C1's preprocessing is correct, and that is what the primary gate
    (:func:`compare_to_bank` against a recomputation in the same process) is
    for.

    Two bounds, because neither is sufficient alone:

    * ``max_abs_diff`` is loose enough to survive float16, and would therefore
      let a genuinely different image through whenever the features are small.
    * ``min_cosine`` is per row and catches exactly that: precision changes a
      vector's length, not the direction it points.

    A zero row is refused rather than scored: its cosine is ``0/0``, and NaN
    compares False against the floor, so the row would pass without anyone
    having compared anything.
    """
    max_abs_diff = _validate_bound("max_abs_diff", max_abs_diff)
    min_cosine = _validate_bound("min_cosine", min_cosine, upper=1.0)

    if recomputed.shape != cached.shape:
        raise ValueError(
            f"shape mismatch: recalculado {recomputed.shape} vs cache {cached.shape}; "
            "no son las mismas imagenes ni el mismo backbone"
        )

    left = recomputed.astype(np.float64)
    right = cached.astype(np.float64)

    reasons: list[str] = []

    n_non_finite = int((~np.isfinite(left)).sum() + (~np.isfinite(right)).sum())
    if n_non_finite:
        reasons.append(
            f"{n_non_finite} valores no finitos (nan/inf): toda comparacion contra ellos "
            "es False, asi que pasarian sin haberse comparado"
        )

    per_row_diff = np.abs(left - right).max(axis=1)
    # ~(x <= tol) rather than (x > tol): NaN has to fail, not vanish.
    rows_over = ~(per_row_diff <= max_abs_diff)
    worst_row = int(np.nanargmax(per_row_diff)) if np.isfinite(per_row_diff).any() else 0

    left_norms = np.linalg.norm(left, axis=1)
    right_norms = np.linalg.norm(right, axis=1)
    zero_rows = (left_norms == 0) | (right_norms == 0)
    if zero_rows.any():
        reasons.append(
            f"{int(zero_rows.sum())} filas de norma cero: un vector nulo no tiene "
            "direccion, su coseno seria 0/0 y pasaria el piso sin compararse"
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        cosines = np.einsum("ij,ij->i", left, right) / (left_norms * right_norms)
    rows_below_cosine = ~(cosines >= min_cosine)
    # Zero rows are already reported above; do not double-count them here.
    rows_below_cosine &= ~zero_rows

    if rows_over.any():
        reasons.append(
            f"max_abs_diff {np.nanmax(per_row_diff):.3e} por encima de {max_abs_diff:g} "
            f"en {int(rows_over.sum())}/{len(per_row_diff)} filas"
        )
    if rows_below_cosine.any():
        reasons.append(
            f"coseno por fila por debajo de {min_cosine:g} en "
            f"{int(rows_below_cosine.sum())}/{len(cosines)} filas: no son la misma imagen"
        )

    finite_cosines = cosines[np.isfinite(cosines)]
    return {
        "agrees": not reasons,
        "max_abs_diff": float(np.nanmax(per_row_diff)) if per_row_diff.size else 0.0,
        "mean_abs_diff": float(np.nanmean(per_row_diff)) if per_row_diff.size else 0.0,
        "min_cosine": float(finite_cosines.min()) if finite_cosines.size else float("nan"),
        "n_rows": int(len(per_row_diff)),
        "n_rows_over_max_abs_diff": int(rows_over.sum()),
        "n_rows_below_cosine": int(rows_below_cosine.sum()),
        "n_zero_rows": int(zero_rows.sum()),
        "n_non_finite": n_non_finite,
        "worst_row": worst_row,
        "max_abs_diff_bound": float(max_abs_diff),
        "min_cosine_bound": float(min_cosine),
        "reasons": reasons,
    }
