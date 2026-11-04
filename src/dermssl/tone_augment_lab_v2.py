"""Mask-anchored CIELAB skin tone augmentation, Corbin eq. 7 with a measured ITA.

Sibling of :mod:`dermssl.tone_augment_lab`, the same way ``tone_augment_pairs``
was a sibling of ``tone_transforms``. ``tone_augment_lab`` is frozen by
contract and NOT modified here: its outputs back already-reported evidence,
and the paired comparison this module exists to run depends on that file
staying byte-identical.

Why a sibling instead of an edit: ``tone_augment_lab.sample_ita_delta`` draws
``ita_delta`` from a hash, decoupled from the image it is applied to. Corbin &
Marques (IEEE Access 11:78339, 2023) eq. 7 defines it as
``ita_delta = ITA_measured - ITA_objetivo`` -- anchored to a skin tone actually
present in the frame. The null result on the hash-sampled version (6 seeds, 12
runs, delta bounded to +0,0034) tested whether an image-independent Lab shift
helps; it never tested eq. 7, because there was no measured ITA to anchor to.

What changed to make this possible: ``fitzpatrick.estimate_fst`` run with the
teacher-026 segmentation mask reaches MAE 0,91 degrees against an expert-mask
ceiling on ISIC 2018 (n=1.813), bias -0,03 -- see
``docs/progress/escalera-ita-mascara.md``, step 1-4. That is the measurement
eq. 7 needs and did not have before.

Two deliberate departures from the paper, both declared rather than silently
assumed:

* Corbin samples ``ITA'`` inside the *target Fitzpatrick bucket*'s range. This
  module does not: ``plan-segmentacion/README.md`` section 1 documents that
  the in-repo ITA-to-FST mapping (Groh's fixed thresholds) is not validated --
  22,9% accuracy against a 51,3% majority-class baseline on PAD-UFES-20
  (Alipour et al., Skin Research and Technology 2026, PMC13459198). Treating a
  bucket label as ground truth for the anchor would import that error into the
  sample. ``ITA_objetivo`` is instead drawn directly in continuous ITA
  degrees, over the span Groh's two finite calibration points cover
  (``fitzpatrick.FST_THRESHOLDS[0]`` to ``[-1]``, i.e. 40 to -25): the same
  numbers already used in this repo to anchor the ITA axis, used here as a
  plain degree span rather than as bucket boundaries or labels.
* The magnitude cap is not re-derived. ``tone_augment_lab.MAX_ABS_ITA_DELTA``
  (25,0 degrees) already is Corbin's "two Fitzpatrick category" visual-quality
  limit translated to degrees, with its derivation documented in that module's
  own docstring. Deriving a second number here from a different source would
  leave two inconsistent caps in the repo; this module imports and reuses the
  one that already exists.

Two further rules this module enforces that eq. 7 alone does not:

* Abstention. Eq. 7 has no value when ``ITA_measured`` is undefined --
  ``fitzpatrick.MIN_B_FOR_ITA`` reports roughly 30% of masked ISIC 2018 images
  as ``ita_indeterminate`` (``docs/progress/escalera-ita-mascara.md``). The
  default here is to abstain: no transform, no invented delta, the case
  recorded with its reason. This default has not been formally ratified; if
  it changes, this module changes with it.
* A 50/50 mixing gate, independent of the eq. 7 draw, so half of any paired
  run's treatment-arm images keep their original tone even when ITA is
  measurable -- the same mixing ratio used elsewhere in this project's tone
  augmentations.

Both the objetivo draw and the mixing draw are hash-derived from
``(seed, step, image_sha256)``, under their own key namespace, for the same
reason ``tone_augment_lab.sample_ita_delta`` is: the paired control arm must
see an identical data order, so this module may not touch any RNG the sampler
also uses.

Lesion preservation, the boundary feather, the gamut-clipping count and the
b*-sign-flip count are not reimplemented: this module measures ITA and derives
``ita_delta``, then hands the pixels to
``tone_augment_lab.augment_skin_tone`` unchanged.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dermssl.fitzpatrick import FST_THRESHOLDS, estimate_fst
from dermssl.tone_augment_lab import (
    DEFAULT_FEATHER_SIGMA,
    MAX_ABS_ITA_DELTA,
    augment_skin_tone,
)
from dermssl.tone_normalization import _as_binary_mask, _check_image_range

# Groh's two finite calibration cut points (fitzpatrick.FST_THRESHOLDS is
# descending: 40, 23, 12, 0, -25), reused as the span of the ITA continuum to
# sample ``ITA_objetivo`` from. Plain degree bounds, not bucket boundaries:
# this module never calls fitzpatrick.fst_bucket or looks up a per-bucket
# range, because the bucket LABELS are what plan-segmentacion/README.md
# documents as unvalidated. The underlying numbers are not in question.
ITA_OBJETIVO_HIGH = FST_THRESHOLDS[0]
ITA_OBJETIVO_LOW = FST_THRESHOLDS[-1]

# Independent of eq. 7: half of the images that pass abstention still keep
# their original tone. A draw above this threshold selects the transform.
MIX_THRESHOLD = 0.5


def _validate_key(seed: int, step: int, image_sha256: str) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {seed!r}")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("image_sha256 must be a 64-character hex digest")
    if any(character not in "0123456789abcdef" for character in image_sha256):
        raise ValueError("image_sha256 must be lowercase hex")


def _hash_unit_interval(namespace: str, seed: int, step: int, image_sha256: str) -> float:
    """Deterministic draw in ``[0, 1)``, independent of any mutable RNG."""
    key = f"{namespace}:{seed}:{step}:{image_sha256}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def sample_ita_objetivo(
    seed: int,
    step: int,
    image_sha256: str,
    *,
    low: float = ITA_OBJETIVO_LOW,
    high: float = ITA_OBJETIVO_HIGH,
) -> float:
    """Deterministic target ITA in degrees, drawn independently of the image.

    Uniform over ``[low, high]`` (default: Groh's two finite calibration
    points, 40 to -25 degrees). This value does not depend on the image's own
    measured ITA -- that independence is what lets eq. 7 pull an image toward
    a plausible tone region rather than kicking it away from wherever it
    already sits.
    """
    _validate_key(seed, step, image_sha256)
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError(f"low/high must be finite with low <= high, got {low!r}, {high!r}")
    raw = _hash_unit_interval("tone-augment-lab-v2-objetivo", seed, step, image_sha256)
    return float(low + raw * (high - low))


def sample_apply_draw(seed: int, step: int, image_sha256: str) -> float:
    """Deterministic draw in ``[0, 1)`` for the 50/50 mixing gate.

    Separate key namespace from :func:`sample_ita_objetivo`: the mixing
    decision and the target ITA must not be correlated draws of the same
    underlying number.
    """
    _validate_key(seed, step, image_sha256)
    return _hash_unit_interval("tone-augment-lab-v2-apply", seed, step, image_sha256)


def compute_ita_delta(
    measured_ita: float,
    seed: int,
    step: int,
    image_sha256: str,
    *,
    max_abs: float = MAX_ABS_ITA_DELTA,
    low: float = ITA_OBJETIVO_LOW,
    high: float = ITA_OBJETIVO_HIGH,
) -> dict:
    """Corbin eq. 7 with a continuous, anchor-free target: ``ITA - ITA_objetivo``.

    Returns the objetivo drawn, the uncapped delta, the delta actually usable
    (clipped to ``[-max_abs, max_abs]``), and whether the cap bound.
    """
    if not np.isfinite(measured_ita):
        raise ValueError(f"measured_ita must be finite, got {measured_ita!r}")
    objetivo = sample_ita_objetivo(seed, step, image_sha256, low=low, high=high)
    raw_delta = measured_ita - objetivo
    applied_delta = float(np.clip(raw_delta, -max_abs, max_abs))
    return {
        "ita_objetivo": objetivo,
        "ita_delta_raw": float(raw_delta),
        "ita_delta": applied_delta,
        "ita_delta_capped_by_objetivo_draw": bool(abs(raw_delta) > max_abs),
    }


def _empty_geometry_fields() -> dict:
    """Keys ``tone_augment_lab.summarize_augmentation_records`` expects.

    Every non-applied branch below returns this shape so the frozen module's
    aggregator can summarize v2 records without modification.
    """
    return {
        "ita_delta_requested": 0.0,
        "ita_delta_applied": 0.0,
        "ita_delta_capped": False,
        "feather_sigma": 0.0,
        "skin_fraction": 0.0,
        "lesion_fraction": 0.0,
        "skin_median_l_before": None,
        "skin_median_b_before": None,
        "delta_l_requested": 0.0,
        "delta_l_realized": 0.0,
        "delta_b_requested": 0.0,
        "delta_b_realized": 0.0,
        "gamut_clipped_pixels": 0,
        "skin_median_l_after": None,
        "skin_median_b_after": None,
        "b_sign_flipped_pixels": 0,
    }


def augment_skin_tone_v2(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    seed: int,
    step: int,
    image_sha256: str,
    valid_fov: np.ndarray | None = None,
    *,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
    max_abs_ita_delta: float = MAX_ABS_ITA_DELTA,
    mix_threshold: float = MIX_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """Measure ITA with the mask, derive eq. 7's delta, then apply the frozen geometry.

    Three ways this returns identity (no pixel changed, ``applied is False``):

    * ITA is not measurable (``fitzpatrick.estimate_fst`` status other than
      ``"ok"`` -- degenerate mask, too little skin, or ``ita_indeterminate``).
      Recorded as ``fallback_reason="ita_unmeasured_<status>"``. This is the
      abstention default: no delta is invented when eq. 7 has no anchor.
    * ITA is measurable but the 50/50 mixing draw did not select this image.
      Recorded as ``fallback_reason="not_selected_by_mix"``.
    * The frozen ``augment_skin_tone`` itself falls back (its own degenerate
      lesion mask or insufficient skin area guard). Its ``fallback_reason``
      passes through unchanged.
    """
    img = _check_image_range(img)
    lesion_mask = _as_binary_mask(lesion_mask, "lesion_mask")
    fov = None if valid_fov is None else _as_binary_mask(valid_fov, "valid_fov")
    _validate_key(seed, step, image_sha256)

    # fitzpatrick.estimate_fst's border_mask is True=NOT-skin; valid_fov here
    # is True=IN-FIELD, the opposite polarity, matching how
    # tone_augment_lab._skin_selector treats the same argument.
    border_mask = None if fov is None else ~fov
    measurement = estimate_fst(img, mask=lesion_mask, border_mask=border_mask)

    base = {
        "ita_measured": measurement.ita,
        "ita_measurement_status": measurement.status,
        "ita_measurement_skin_fraction": measurement.skin_fraction,
    }

    if measurement.status != "ok":
        return img.copy(), {
            **base,
            **_empty_geometry_fields(),
            "applied": False,
            "fallback_reason": f"ita_unmeasured_{measurement.status}",
            "ita_objetivo": None,
            "ita_delta_raw": None,
            "apply_draw": None,
            "selected_for_transform": False,
        }

    draw = sample_apply_draw(seed, step, image_sha256)
    if draw <= mix_threshold:
        return img.copy(), {
            **base,
            **_empty_geometry_fields(),
            "applied": False,
            "fallback_reason": "not_selected_by_mix",
            "ita_objetivo": None,
            "ita_delta_raw": None,
            "apply_draw": draw,
            "selected_for_transform": False,
        }

    delta_record = compute_ita_delta(
        measurement.ita, seed, step, image_sha256, max_abs=max_abs_ita_delta
    )
    out, geometry_record = augment_skin_tone(
        img,
        lesion_mask,
        fov,
        ita_delta=delta_record["ita_delta"],
        feather_sigma=feather_sigma,
        max_abs_ita_delta=max_abs_ita_delta,
    )
    return out, {
        **base,
        **geometry_record,
        "ita_objetivo": delta_record["ita_objetivo"],
        "ita_delta_raw": delta_record["ita_delta_raw"],
        "apply_draw": draw,
        "selected_for_transform": True,
    }
