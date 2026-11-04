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

REVISION. The first version of this module sampled ``ITA_objetivo`` from a
fixed global span (Groh's two finite calibration points, -25 to 40), decoupled
from the measured ITA. On a realistic ISIC 2018 distribution (median ITA
~56), that span sits almost entirely below the measured value, so
``ita_delta = measured - objetivo`` comes out large and positive on nearly
every image, and ``tone_augment_lab``'s cap then clips 80%+ of them to the
same constant +25,0 -- measured directly: 79,9% of 5.000 synthetic draws at a
realistic median landed within 0,025 degrees of the cap. A delta that is the
same number on 80% of images is not anchored to anything; eq. 7 degenerates
back into the undirected constant this module exists to replace.

The fix: ``ITA_objetivo`` is now sampled in a window CENTERED ON THE MEASURED
ITA, ``U(ita_medido - W, ita_medido + W)``, rather than in a fixed global span.
This both anchors the target to the image (every image gets its own window)
and makes the cap structural rather than a clipping event: because
``ita_delta = ita_medido - ita_objetivo`` and ``ita_objetivo`` never leaves
``[ita_medido - W, ita_medido + W]``, the delta is already in ``[-W, W]`` by
construction, with zero images landing on the boundary by truncation.

``W`` reuses ``tone_augment_lab.MAX_ABS_ITA_DELTA`` (25,0 degrees), Corbin's
"two Fitzpatrick category" visual-quality limit translated to degrees, with
its derivation documented in that module's own docstring. ``W`` is NOT a
second independent re-derivation of that number: it is the same constant,
repurposed here as a window radius instead of an absolute cap. Avoiding a
second, possibly inconsistent, derivation was deliberate in the first version
of this module and remains deliberate here; what changed is how the constant
is used, not where it comes from.

Two deliberate departures from the paper remain, both declared rather than
silently assumed:

* Corbin samples ``ITA'`` inside the *target Fitzpatrick bucket*'s range, a
  step that requires trusting a bucket label. This module never does:
  ``plan-segmentacion/README.md`` section 1 documents that the in-repo
  ITA-to-FST mapping (Groh's fixed thresholds) is not validated -- 22,9%
  accuracy against a 51,3% majority-class baseline on PAD-UFES-20 (Alipour et
  al., Skin Research and Technology 2026, PMC13459198). This module never
  calls ``fitzpatrick.fst_bucket`` or looks up a per-bucket range; the window
  above is centered on a continuous measured ITA, not on a bucket.
* The window radius equals the magnitude cap, not re-derived separately (see
  above).

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
``tone_augment_lab.augment_skin_tone`` unchanged. That geometry can still push
skin b* negative on an image whose measured b* sits close to
``fitzpatrick.MIN_B_FOR_ITA`` (5,0) and draws a strongly negative delta inside
the window; this is measured and reported via the inherited
``skin_median_b_after`` and ``b_sign_flipped_pixels`` fields, not hard-blocked
-- the same stance ``tone_augment_lab`` itself takes, and for the same reason:
a sign flip is a real change the encoder will see, so it is published rather
than hidden by a gate.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dermssl.fitzpatrick import estimate_fst
from dermssl.tone_augment_lab import (
    DEFAULT_FEATHER_SIGMA,
    MAX_ABS_ITA_DELTA,
    augment_skin_tone,
)
from dermssl.tone_normalization import _as_binary_mask, _check_image_range

# Reused as the window radius around the measured ITA, not as a global span
# and not as a post-hoc clip bound. See the module docstring's "REVISION".
TARGET_WINDOW_DEGREES = MAX_ABS_ITA_DELTA

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


def _hash_signed_unit_interval(namespace: str, seed: int, step: int, image_sha256: str) -> float:
    """Deterministic draw in ``[-1, 1)``, independent of any mutable RNG."""
    key = f"{namespace}:{seed}:{step}:{image_sha256}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    raw = int.from_bytes(digest[:4], "big") / 2**32
    return float(2.0 * raw - 1.0)


def sample_ita_objetivo(
    measured_ita: float,
    seed: int,
    step: int,
    image_sha256: str,
    *,
    window: float = TARGET_WINDOW_DEGREES,
) -> float:
    """Deterministic target ITA, drawn in ``U(measured_ita - window, measured_ita + window)``.

    Centered on ``measured_ita`` on purpose: this is what makes the eq. 7
    delta depend on the image rather than collapsing to a fixed global
    tendency (see the module docstring's "REVISION").
    """
    if not np.isfinite(measured_ita):
        raise ValueError(f"measured_ita must be finite, got {measured_ita!r}")
    _validate_key(seed, step, image_sha256)
    if not np.isfinite(window) or window < 0:
        raise ValueError(f"window must be finite and non-negative, got {window!r}")
    raw = _hash_signed_unit_interval("tone-augment-lab-v2-objetivo", seed, step, image_sha256)
    return float(measured_ita + raw * window)


def sample_apply_draw(seed: int, step: int, image_sha256: str) -> float:
    """Deterministic draw in ``[0, 1)`` for the 50/50 mixing gate.

    Separate key namespace from :func:`sample_ita_objetivo`: the mixing
    decision and the target ITA must not be correlated draws of the same
    underlying number.
    """
    _validate_key(seed, step, image_sha256)
    raw = _hash_signed_unit_interval("tone-augment-lab-v2-apply", seed, step, image_sha256)
    return float((raw + 1.0) / 2.0)


def compute_ita_delta(
    measured_ita: float,
    seed: int,
    step: int,
    image_sha256: str,
    *,
    window: float = TARGET_WINDOW_DEGREES,
) -> dict:
    """Corbin eq. 7, ``ITA_measured - ITA_objetivo``, with the window above.

    ``ita_delta_raw`` and ``ita_delta`` are always equal: the window makes the
    result land in ``[-window, window]`` by construction, so there is no
    separate clipping step and nothing to truncate. Both keys are kept so a
    downstream manifest has one stable name regardless of which augmentation
    version produced the record.
    """
    objetivo = sample_ita_objetivo(measured_ita, seed, step, image_sha256, window=window)
    delta = measured_ita - objetivo
    return {
        "ita_objetivo": objetivo,
        "ita_delta_raw": float(delta),
        "ita_delta": float(delta),
        "ita_delta_capped": False,
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
    window: float = TARGET_WINDOW_DEGREES,
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

    The manifest always carries ``ita_measured``, ``ita_measurement_status``,
    ``ita_objetivo``, ``ita_delta_raw`` and ``ita_delta_applied`` (the last via
    the geometry record), ``None`` where a branch has no measurement or no
    draw to report.
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

    delta_record = compute_ita_delta(measurement.ita, seed, step, image_sha256, window=window)
    out, geometry_record = augment_skin_tone(
        img,
        lesion_mask,
        fov,
        ita_delta=delta_record["ita_delta"],
        feather_sigma=feather_sigma,
        max_abs_ita_delta=window,
    )
    return out, {
        **base,
        **geometry_record,
        "ita_objetivo": delta_record["ita_objetivo"],
        "ita_delta_raw": delta_record["ita_delta_raw"],
        "apply_draw": draw,
        "selected_for_transform": True,
    }
