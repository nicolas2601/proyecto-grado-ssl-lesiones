"""Data-driven skin tone normalization in CIELAB, with lesion preservation.

This is the inference-time counterpart of :mod:`dermssl.circle`. The difference
matters and is easy to blur:

* :func:`dermssl.circle.shift_skin_tone` applies a *fixed* FST-bucket shift. It
  adds variation. It is a perturbation, used as augmentation or stress test.
* This module estimates a *target* L* from skin pixels and maps each image
  toward it. It removes variation. It is a normalization, used as a
  deterministic preprocessing stage before the encoder.

Leakage control is enforced, not documented: :func:`fit_tone_reference` returns
a reference bound to a ``fold_id`` and to a digest of the image ids it was
fitted on, and :func:`normalize_skin_tone` refuses to apply a reference whose
``fold_id`` does not match the fold being transformed. Fitting the target on the
full dataset, including evaluation folds, would leak the evaluation
distribution into preprocessing and inflate the benchmark.

Scope limits, stated as measured rather than as intended:

* Only the L* channel is modified *in Lab*. a* and b* are left untouched there,
  because lesion chroma carries diagnostic information. However, converting
  back to sRGB clips out-of-gamut values, and on chromatic skin that clipping
  does shift the realized a*/b* and reduces the realized L* shift. The records
  report ``gamut_clipped_pixels`` and ``delta_l_realized`` so this is visible
  instead of assumed.
* The mapping is an additive shift of the skin median, not an affine match of
  mean and variance. One auditable scalar per image, and no rescaling of the
  contrast inside the skin region.
* This is normalization of acquisition and illumination, not of Fitzpatrick
  phototype. ITA and L* correspond poorly to FST under uncontrolled capture, so
  no fairness or phototype claim follows from applying this transform.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dermssl.circle import lab_to_srgb
from dermssl.fitzpatrick import (
    MAX_MASK_AREA_FRACTION,
    MIN_MASK_AREA_FRACTION,
    MIN_SKIN_FRACTION,
    srgb_to_lab,
)
from dermssl.mask_manifest import mask_sha256

# Largest L* correction a single image may receive. A mask error can otherwise
# turn a wrong skin estimate into an extreme shift; the cap bounds the damage
# and the records report how often it binds.
MAX_ABS_DELTA_L = 12.0


def _as_binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Reject a soft mask instead of silently thresholding it at zero.

    ``np.asarray(x, dtype=bool)`` turns a probability map into all-True, which
    makes every image degenerate and sends the whole run to identity. The
    contrast would then report that tone normalization changed nothing, when in
    fact it never ran. That failure must be loud.
    """
    arr = np.asarray(mask)
    if arr.dtype != bool:
        finite = np.isfinite(arr)
        if not finite.all() or not np.isin(arr[finite], (0, 1)).all():
            raise ValueError(
                f"{name} must be boolean or strictly binary; got dtype={arr.dtype} "
                f"with values outside {{0, 1}} (a soft mask must be thresholded first)"
            )
    return arr.astype(bool)


def _check_image_range(img: np.ndarray) -> np.ndarray:
    """sRGB floats must already be in [0, 1], or preservation is a lie.

    Clipping after restoring protected pixels would silently alter them, so an
    out-of-range frame is refused rather than repaired.
    """
    arr = np.asarray(img, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("image contains non-finite values")
    low, high = float(arr.min()), float(arr.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f"image must be sRGB floats in [0, 1]; got range [{low}, {high}]")
    return arr


def _skin_selector(lesion_mask: np.ndarray, valid_fov: np.ndarray | None) -> np.ndarray:
    """Skin is everything that is neither lesion nor outside the field of view."""
    skin = ~lesion_mask
    if valid_fov is not None:
        skin &= valid_fov
    return skin


def _lesion_fraction(lesion_mask: np.ndarray, valid_fov: np.ndarray | None) -> float:
    """Lesion area measured inside the usable field of view.

    Measuring against the whole frame would call a lesion that fills the
    dermatoscope aperture "small", and the fallback reason recorded in the run
    report would then be misleading.
    """
    if valid_fov is None:
        return float(lesion_mask.mean())
    usable = int(np.count_nonzero(valid_fov))
    if usable == 0:
        return 1.0
    return float(np.count_nonzero(lesion_mask & valid_fov) / usable)


def skin_tone_statistic(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    valid_fov: np.ndarray | None = None,
) -> dict:
    """Summarize the L* distribution of predicted skin for one image.

    Returns a record rather than a bare float so the caller can audit why an
    image was skipped. ``usable`` is False when the mask is degenerate or there
    is too little skin to estimate anything; in that case ``median_l`` is None
    and the image must go through unchanged and be counted.
    """
    img = _check_image_range(img)
    lesion_mask = _as_binary_mask(lesion_mask, "lesion_mask")
    fov = None if valid_fov is None else _as_binary_mask(valid_fov, "valid_fov")
    skin = _skin_selector(lesion_mask, fov)
    skin_fraction = float(skin.mean())
    lesion_fraction = _lesion_fraction(lesion_mask, fov)

    if not (MIN_MASK_AREA_FRACTION <= lesion_fraction <= MAX_MASK_AREA_FRACTION):
        reason = "degenerate_lesion_mask"
    elif skin_fraction < MIN_SKIN_FRACTION:
        reason = "insufficient_skin_area"
    else:
        reason = None

    if reason is not None:
        return {
            "usable": False,
            "skip_reason": reason,
            "median_l": None,
            "iqr_l": None,
            "skin_fraction": skin_fraction,
            "lesion_fraction": lesion_fraction,
        }

    skin_l = srgb_to_lab(img)[..., 0][skin]
    q25, q75 = np.percentile(skin_l, [25.0, 75.0])
    return {
        "usable": True,
        "skip_reason": None,
        "median_l": float(np.median(skin_l)),
        "iqr_l": float(q75 - q25),
        "skin_fraction": skin_fraction,
        "lesion_fraction": lesion_fraction,
    }


def fit_tone_reference(
    statistics: list[dict],
    *,
    fold_id: str,
    image_ids: list[str] | None = None,
) -> dict:
    """Fit the target L* from training-fold statistics only.

    ``statistics`` must come exclusively from images in the training folds of
    the current split. ``fold_id`` binds the reference to that fold;
    :func:`normalize_skin_tone` refuses a mismatched fold, so a reference cannot
    be reused across folds by accident.

    ``image_ids`` are hashed into ``fitted_ids_sha256`` so the disjointness of
    the fitting set against an evaluation fold can be *demonstrated* afterwards
    rather than asserted. When they are not supplied the digest is None and
    ``ids_bound`` is False: the absence of evidence is recorded as absence, not
    papered over with a reassuring string.
    """
    usable = [
        s["median_l"]
        for s in statistics
        if s.get("usable") and s.get("median_l") is not None and np.isfinite(s["median_l"])
    ]
    if not usable:
        raise ValueError("no usable skin statistics: cannot fit a tone reference")
    values = np.asarray(usable, dtype=float)

    digest = None
    if image_ids is not None:
        hasher = hashlib.sha256()
        for image_id in sorted(image_ids):
            hasher.update(image_id.encode("utf-8"))
            hasher.update(b"\x00")
        digest = hasher.hexdigest()

    return {
        "fold_id": fold_id,
        "target_l": float(np.median(values)),
        "n_images_fitted": int(values.size),
        "n_images_skipped": int(len(statistics) - values.size),
        "fitted_spread_l": float(np.percentile(values, 75.0) - np.percentile(values, 25.0)),
        "fitted_ids_sha256": digest,
        "n_fitted_ids": None if image_ids is None else len(image_ids),
        "ids_bound": image_ids is not None,
    }


def normalize_skin_tone(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    valid_fov: np.ndarray | None = None,
    *,
    reference: dict,
    fold_id: str,
    max_abs_delta_l: float = MAX_ABS_DELTA_L,
) -> tuple[np.ndarray, dict]:
    """Shift skin L* toward the reference target, preserving protected pixels.

    ``reference`` comes from :func:`fit_tone_reference` and ``fold_id`` is the
    fold being transformed. A mismatch raises: that is the leakage guard.

    Returns the transformed image and a record of what actually happened. The
    record is not optional bookkeeping: a null result is uninterpretable without
    knowing the realized magnitudes, how often the cap bound, how many pixels
    left the gamut and how many images fell back to identity.
    """
    if reference.get("fold_id") != fold_id:
        raise ValueError(
            f"tone reference was fitted on fold {reference.get('fold_id')!r} but is being "
            f"applied to fold {fold_id!r}; refusing to leak a reference across folds"
        )
    target_l = reference.get("target_l")
    if target_l is None or not np.isfinite(target_l):
        raise ValueError(f"tone reference has a non-finite target_l: {target_l!r}")

    img = _check_image_range(img)
    lesion_mask = _as_binary_mask(lesion_mask, "lesion_mask")
    fov = None if valid_fov is None else _as_binary_mask(valid_fov, "valid_fov")
    stat = skin_tone_statistic(img, lesion_mask, fov)

    base_record = {
        "fold_id": fold_id,
        "target_l": float(target_l),
        "skin_median_l_before": stat["median_l"],
        "skin_iqr_l_before": stat["iqr_l"],
        "skin_fraction": stat["skin_fraction"],
        "lesion_fraction": stat["lesion_fraction"],
    }

    if not stat["usable"]:
        # Identity, counted. Never transform a frame whose protection cannot be
        # established: a corrupt mask must not become a silent full-frame edit.
        return img.copy(), {
            **base_record,
            "applied": False,
            "fallback_reason": stat["skip_reason"],
            "delta_l_requested": 0.0,
            "delta_l_applied": 0.0,
            "delta_l_realized": 0.0,
            "delta_l_capped": False,
            "gamut_clipped_pixels": 0,
            "skin_median_l_after": stat["median_l"],
        }

    delta_requested = float(target_l) - float(stat["median_l"])
    delta_applied = float(np.clip(delta_requested, -max_abs_delta_l, max_abs_delta_l))

    lab = srgb_to_lab(img)
    skin = _skin_selector(lesion_mask, fov)
    out_lab = lab.copy()
    out_lab[..., 0] = np.where(skin, np.clip(lab[..., 0] + delta_applied, 0.0, 100.0), lab[..., 0])
    out = np.clip(lab_to_srgb(out_lab), 0.0, 1.0)
    # Restore protected pixels from the original, pixel for pixel. The Lab round
    # trip is not exact enough to rely on for preservation. Clipping happens
    # before this restore so protected pixels are never touched by it.
    out = np.where(skin[..., None], out, img)

    # Gamut loss is where the realized shift is lost, and it leaves no trace in
    # Lab. Count skin pixels pinned at a channel bound that were not already
    # pinned there before the shift.
    at_bound = (out <= 0.0) | (out >= 1.0)
    was_at_bound = (img <= 0.0) | (img >= 1.0)
    gamut_clipped = int(np.count_nonzero(skin & (at_bound & ~was_at_bound).any(axis=-1)))

    after = float(np.median(srgb_to_lab(out)[..., 0][skin]))
    return out, {
        **base_record,
        "applied": True,
        "fallback_reason": None,
        "delta_l_requested": delta_requested,
        "delta_l_applied": delta_applied,
        "delta_l_realized": after - float(stat["median_l"]),
        "delta_l_capped": bool(abs(delta_requested) > max_abs_delta_l),
        "gamut_clipped_pixels": gamut_clipped,
        "skin_median_l_after": after,
    }


def normalize_skin_tone_from_mask_record(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    mask_record: dict,
    *,
    reference: dict,
    fold_id: str,
    max_abs_delta_l: float = MAX_ABS_DELTA_L,
) -> tuple[np.ndarray, dict]:
    """Normalize only when the mask passes QA and its provenance checks out.

    The augmentation path already has this gate
    (:func:`dermssl.circle.shift_skin_tone_from_mask_record`). It matters more
    here: these masks come from a segmenter applied out of domain, whose quality
    on the target cohort is not measured, and this path feeds a reported number
    rather than a training-time perturbation.
    """
    if not bool(mask_record.get("eligible")):
        raise ValueError("mask record is not eligible for tone normalization")
    if mask_record.get("mask_semantics") != "true=lesion":
        raise ValueError("mask record must declare true=lesion semantics")
    checked = _as_binary_mask(lesion_mask, "lesion_mask")
    if mask_record.get("mask_sha256") != mask_sha256(checked):
        raise ValueError("mask provenance does not match the delivered pixels")
    return normalize_skin_tone(
        img,
        checked,
        None,
        reference=reference,
        fold_id=fold_id,
        max_abs_delta_l=max_abs_delta_l,
    )


def summarize_normalization_records(records: list[dict]) -> dict:
    """Aggregate per-image records into the counts a run report must publish.

    Both the requested and the realized magnitudes are reported. Publishing only
    the requested one would overstate the intervention whenever the gamut or the
    cap absorbed part of it.
    """
    applied = [r for r in records if r.get("applied")]
    requested = np.asarray([r["delta_l_applied"] for r in applied], dtype=float)
    realized = np.asarray([r["delta_l_realized"] for r in applied], dtype=float)
    iqrs = [r["skin_iqr_l_before"] for r in applied if r.get("skin_iqr_l_before") is not None]
    fallbacks: dict[str, int] = {}
    for record in records:
        reason = record.get("fallback_reason")
        if reason is not None:
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
    return {
        "images": len(records),
        "transformed": len(applied),
        "fallbacks": len(records) - len(applied),
        "fallback_reason_counts": fallbacks,
        "capped": int(sum(1 for r in applied if r["delta_l_capped"])),
        "gamut_clipped_pixels_total": int(sum(r["gamut_clipped_pixels"] for r in applied)),
        "images_with_gamut_clipping": int(
            sum(1 for r in applied if r["gamut_clipped_pixels"] > 0)
        ),
        "delta_l_applied_mean": float(requested.mean()) if requested.size else None,
        "delta_l_realized_mean": float(realized.mean()) if realized.size else None,
        "delta_l_realized_abs_max": float(np.abs(realized).max()) if realized.size else None,
        "skin_iqr_l_before_median": float(np.median(iqrs)) if iqrs else None,
        "delta_l_sign_counts": {
            "negative": int(np.count_nonzero(requested < 0.0)),
            "zero": int(np.count_nonzero(requested == 0.0)),
            "positive": int(np.count_nonzero(requested > 0.0)),
        },
    }
