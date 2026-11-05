"""Two-channel CIELAB skin tone augmentation for SSL pretraining.

This is the *training-time* counterpart of :mod:`dermssl.tone_normalization`.
The distinction is the same one that module draws between itself and
:func:`dermssl.circle.shift_skin_tone`, and it is worth keeping sharp:

* ``tone_normalization`` estimates a target L* and maps images toward it. It
  removes variation. Deterministic preprocessing, applied at inference.
* This module samples a bounded tone displacement per (seed, step, image) and
  applies it. It adds variation. An augmentation, applied during training.

Why it exists: every inference-time intervention measured on the frozen encoder
degraded classification (fixed L* perturbation -1,92 pp, tone normalization
-2,49 pp, background suppression -7,75 pp on PAD macro-F1), and the damage grew
monotonically with how much of the image was altered. That is the signature of a
train/test distribution mismatch, not of a bad transform. Corbin & Marques
(IEEE Access 11:78339, 2023) obtain their gains by training on the transformed
distribution instead, so the transform moves to pretraining.

Geometry, from their equations 8 and 9::

    b = b + (ita_delta * 0.50)
    L = L - (ita_delta * 0.12)

b* carries roughly four times the displacement of L*, because that is how the
two channels co-move along the skin tone locus. Shifting L* alone - which is
what this project did until now - lands the image at an (L*, b*) pair that no
real skin tone occupies.

Two declared deviations from the paper, both measured rather than assumed:

* Corbin derives ``ita_delta`` as ``ITA - ITA'`` toward a target Fitzpatrick
  type. That is unavailable here: on 3,000 sampled pool images, 44.3% return
  ``ita_indeterminate`` because their median skin b* is 0.74 (p05 -7.84), under
  ``fitzpatrick.MIN_B_FOR_ITA``. The skin selection is not at fault - skin
  fraction is 0.77 in both the passing and failing groups - these are polarized
  or immersion dermoscopy frames where the ITA angle saturates and reports only
  the sign of b*. So the magnitude is sampled from a bounded range instead. The
  geometry is the paper's; the anchor is not.
* The dermoscope field of view is *not* the cause of that rate, and passing
  ``border_mask = ~field_validity_mask`` moves it from 44.5% to 44.4%. Recorded
  so the question is not reopened.

Lesion preservation and Corbin's step 6 (Gaussian smoothing of the lesion/skin
boundary) pull against each other: a symmetric feather bleeds transformed colour
*into* the lesion, which is what the preservation contract exists to prevent.
Resolved by computing the blend weight on skin and then forcing it to zero on
the lesion, so the transition is gradual on the skin side and the lesion keeps
its original pixels exactly.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy import ndimage

from dermssl.circle import lab_to_srgb
from dermssl.fitzpatrick import srgb_to_lab
from dermssl.mask_manifest import mask_sha256
from dermssl.tone_normalization import (
    _as_binary_mask,
    _check_image_range,
    _lesion_fraction,
    _skin_selector,
)
from dermssl.fitzpatrick import (
    MAX_MASK_AREA_FRACTION,
    MIN_MASK_AREA_FRACTION,
    MIN_SKIN_FRACTION,
)

# Corbin & Marques, equations 8 and 9. Not tunable: they are the paper's
# geometry, and calibrating them against macro-F1 would be fitting the metric.
B_PER_ITA_DEGREE = 0.50
L_PER_ITA_DEGREE = 0.12

# Corbin caps the transform at two Fitzpatrick types and reports that larger
# jumps looked wrong. Translated to degrees with the ITA ranges that paper
# cites (T1 >55, T2 41-55, T3 28-41, T4 19-28, T5 10-19, T6 <=10), two adjacent
# steps span 18 to 27 degrees. A declared choice, not a measured optimum.
MAX_ABS_ITA_DELTA = 25.0

# Boundary feather, in pixels of the image being transformed. Softens the hard
# mask edge that the error diagnosis of 2026-11-02 observed as "colour changes
# bounded by the masks". Declared before measuring, like the cap.
DEFAULT_FEATHER_SIGMA = 2.0


def sample_ita_delta(
    seed: int,
    step: int,
    image_sha256: str,
    *,
    max_abs: float = MAX_ABS_ITA_DELTA,
) -> float:
    """Deterministic tone displacement for one image at one step.

    Hash-derived on purpose, following ``moco_tone_views.tone_assignment``: the
    paired control arm must see the identical data order and batch boundaries,
    so the augmentation may not draw from any RNG the sampler also uses. Two
    arms differing in one factor is the whole design; a shared mutable RNG
    would silently make them differ in two.

    Returns a value in ``[-max_abs, +max_abs]``.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {seed!r}")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("image_sha256 must be a 64-character hex digest")
    if any(character not in "0123456789abcdef" for character in image_sha256):
        raise ValueError("image_sha256 must be lowercase hex")
    if not np.isfinite(max_abs) or max_abs < 0:
        raise ValueError(f"max_abs must be finite and non-negative, got {max_abs!r}")

    key = f"tone-augment-lab-v1:{seed}:{step}:{image_sha256}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    # 32 bits of the digest mapped to [-1, 1]; the discretization is 2^-31 of
    # the range, far below any effect this transform can have.
    raw = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return float((2.0 * raw - 1.0) * max_abs)


def _blend_weight(
    skin: np.ndarray,
    lesion_mask: np.ndarray,
    feather_sigma: float,
) -> np.ndarray:
    """Alpha for blending transformed skin over the original image.

    One in deep skin, tapering toward the lesion boundary, and exactly zero on
    every lesion pixel. The zeroing is applied *after* the blur, which is what
    keeps Corbin's smoothing from writing into the protected region.
    """
    if feather_sigma <= 0:
        alpha = skin.astype(np.float64)
    else:
        alpha = ndimage.gaussian_filter(skin.astype(np.float64), sigma=feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)
    return np.where(lesion_mask, 0.0, alpha)


def augment_skin_tone(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    valid_fov: np.ndarray | None = None,
    *,
    ita_delta: float,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
    max_abs_ita_delta: float = MAX_ABS_ITA_DELTA,
) -> tuple[np.ndarray, dict]:
    """Displace skin L* and b* by ``ita_delta`` degrees' worth of tone.

    Returns the augmented image and a record of what actually happened in both
    channels. The record is not bookkeeping: a null training result cannot be
    read without knowing the realized magnitudes, how often the cap bound, how
    much of the image left the gamut, and how many frames fell back to identity.

    A missing or degenerate mask yields identity with a reason, never a silent
    full-frame edit - the same refusal ``circle.shift_skin_tone`` makes.
    """
    if not np.isfinite(ita_delta):
        raise ValueError(f"ita_delta must be finite, got {ita_delta!r}")
    if feather_sigma < 0 or not np.isfinite(feather_sigma):
        raise ValueError(f"feather_sigma must be finite and non-negative, got {feather_sigma!r}")

    img = _check_image_range(img)
    lesion_mask = _as_binary_mask(lesion_mask, "lesion_mask")
    fov = None if valid_fov is None else _as_binary_mask(valid_fov, "valid_fov")

    skin = _skin_selector(lesion_mask, fov)
    skin_fraction = float(skin.mean())
    lesion_fraction = _lesion_fraction(lesion_mask, fov)

    applied_delta = float(np.clip(ita_delta, -max_abs_ita_delta, max_abs_ita_delta))
    lab = srgb_to_lab(img)
    base = {
        "ita_delta_requested": float(ita_delta),
        "ita_delta_applied": applied_delta,
        "ita_delta_capped": bool(abs(ita_delta) > max_abs_ita_delta),
        "feather_sigma": float(feather_sigma),
        "skin_fraction": skin_fraction,
        "lesion_fraction": lesion_fraction,
        "skin_median_l_before": float(np.median(lab[..., 0][skin])) if skin.any() else None,
        "skin_median_b_before": float(np.median(lab[..., 2][skin])) if skin.any() else None,
    }

    if not (MIN_MASK_AREA_FRACTION <= lesion_fraction <= MAX_MASK_AREA_FRACTION):
        reason = "degenerate_lesion_mask"
    elif skin_fraction < MIN_SKIN_FRACTION:
        reason = "insufficient_skin_area"
    else:
        reason = None

    if reason is not None:
        return img.copy(), {
            **base,
            "applied": False,
            "fallback_reason": reason,
            "delta_l_requested": 0.0,
            "delta_l_realized": 0.0,
            "delta_b_requested": 0.0,
            "delta_b_realized": 0.0,
            "gamut_clipped_pixels": 0,
            "skin_median_l_after": base["skin_median_l_before"],
            "skin_median_b_after": base["skin_median_b_before"],
            "b_sign_flipped_pixels": 0,
        }

    delta_b = applied_delta * B_PER_ITA_DEGREE
    delta_l = -applied_delta * L_PER_ITA_DEGREE

    if applied_delta == 0.0:
        # Short-circuit, not an optimization. The sRGB -> Lab -> sRGB round trip
        # is accurate to about one ULP, so without this a zero-magnitude call
        # would return an image that differs from its input in the last bit of
        # every skin pixel. The control arm of a paired run is entitled to byte
        # identity when nothing was asked of the transform.
        return img.copy(), {
            **base,
            "applied": True,
            "fallback_reason": None,
            "delta_l_requested": 0.0,
            "delta_l_realized": 0.0,
            "delta_b_requested": 0.0,
            "delta_b_realized": 0.0,
            "gamut_clipped_pixels": 0,
            "skin_median_l_after": base["skin_median_l_before"],
            "skin_median_b_after": base["skin_median_b_before"],
            "b_sign_flipped_pixels": 0,
        }

    out_lab = lab.copy()
    out_lab[..., 0] = np.clip(lab[..., 0] + delta_l, 0.0, 100.0)
    out_lab[..., 2] = lab[..., 2] + delta_b
    shifted = np.clip(lab_to_srgb(out_lab), 0.0, 1.0)

    alpha = _blend_weight(skin, lesion_mask, feather_sigma)[..., None]
    out = alpha * shifted + (1.0 - alpha) * img
    # Hard restore on the lesion. The blend weight is already zero there, so
    # this is belt and braces against a future change to _blend_weight.
    out = np.where(lesion_mask[..., None], img, out)

    # Gamut loss leaves no trace in Lab, so count skin pixels pinned at a
    # channel bound that were not pinned there before.
    #
    # Counted on ``shifted`` - the transform's own output - and deliberately NOT
    # on the blended ``out``. In the feathered ring alpha < 1, so
    # alpha*shifted + (1-alpha)*img lands strictly inside [0, 1] even where
    # shifted was pinned at a bound, and counting there silently undercounts the
    # loss. Measured on one saturating frame: 3655 clipped pixels at sigma 0
    # against 1287 at sigma 4, same colour and same delta. The quantity that
    # makes a null training result interpretable is how much the transform
    # pushed out of gamut, not how much survived the blend.
    at_bound = (shifted <= 0.0) | (shifted >= 1.0)
    was_at_bound = (img <= 0.0) | (img >= 1.0)
    gamut_clipped = int(np.count_nonzero(skin & (at_bound & ~was_at_bound).any(axis=-1)))

    after_lab = srgb_to_lab(out)
    after_l = float(np.median(after_lab[..., 0][skin]))
    after_b = float(np.median(after_lab[..., 2][skin]))

    # 44% of the pool sits near b* = 0, so a displacement can cross the axis.
    # Measured and published rather than avoided: ITA is undefined there anyway,
    # and a sign flip is a real change to the image the encoder will see.
    #
    # "Crossed" means strictly opposite signs. A pixel sitting at exactly
    # b* = 0 is on the axis rather than on a side of it, so moving off zero is
    # not a crossing. Spelled out instead of relying on sign(0) == 0 falling
    # out of a product, which hid the decision.
    before_b = lab[..., 2]
    after_b_plane = after_lab[..., 2]
    crossed = ((before_b < 0.0) & (after_b_plane > 0.0)) | (
        (before_b > 0.0) & (after_b_plane < 0.0)
    )
    flipped = int(np.count_nonzero(skin & crossed))

    return out, {
        **base,
        "applied": True,
        "fallback_reason": None,
        "delta_l_requested": float(delta_l),
        "delta_l_realized": after_l - float(base["skin_median_l_before"]),
        "delta_b_requested": float(delta_b),
        "delta_b_realized": after_b - float(base["skin_median_b_before"]),
        "gamut_clipped_pixels": gamut_clipped,
        "skin_median_l_after": after_l,
        "skin_median_b_after": after_b,
        "b_sign_flipped_pixels": flipped,
    }


def augment_skin_tone_from_mask_record(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    mask_record: dict,
    *,
    ita_delta: float,
    valid_fov: np.ndarray | None = None,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
    max_abs_ita_delta: float = MAX_ABS_ITA_DELTA,
) -> tuple[np.ndarray, dict]:
    """Augment only when the mask pixels match the record that vouches for them.

    Mirrors ``tone_normalization.normalize_skin_tone_from_mask_record``: the
    adapter binds pixels to the JSON contract so a stale, quarantined or altered
    mask cannot be passed off as evidence of lesion preservation.
    """
    semantics = mask_record.get("mask_semantics")
    if semantics != "true=lesion":
        raise ValueError(
            f"mask_record declares mask_semantics={semantics!r}; this module requires "
            "'true=lesion' and will not guess the polarity"
        )
    if not mask_record.get("eligible", False):
        raise ValueError("mask_record is not eligible; refusing to transform behind a failed QA")

    declared = mask_record.get("mask_sha256")
    actual = mask_sha256(_as_binary_mask(lesion_mask, "lesion_mask"))
    if declared != actual:
        raise ValueError(
            f"mask pixels hash to {actual} but the record declares {declared}; "
            "refusing to use a mask whose provenance cannot be demonstrated"
        )

    out, record = augment_skin_tone(
        img,
        lesion_mask,
        valid_fov,
        ita_delta=ita_delta,
        feather_sigma=feather_sigma,
        max_abs_ita_delta=max_abs_ita_delta,
    )
    return out, {**record, "mask_sha256": actual}


def summarize_augmentation_records(records: list[dict]) -> dict:
    """Aggregate a run's augmentation records into the numbers a report needs."""
    total = len(records)
    applied = [r for r in records if r.get("applied")]
    reasons: dict[str, int] = {}
    for record in records:
        if not record.get("applied"):
            reason = record.get("fallback_reason") or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1

    def _mean(key: str) -> float | None:
        values = [float(r[key]) for r in applied if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    def _abs_max(key: str) -> float | None:
        values = [abs(float(r[key])) for r in applied if r.get(key) is not None]
        return float(np.max(values)) if values else None

    def _abs_mean(key: str) -> float | None:
        values = [abs(float(r[key])) for r in applied if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "images": total,
        "transformed": len(applied),
        "transformed_fraction": (len(applied) / total) if total else 0.0,
        "fallbacks_by_reason": reasons,
        "capped": sum(1 for r in applied if r.get("ita_delta_capped")),
        # Two numbers, because one of them was wearing the other's name until
        # 2026-11-04. The field used to be published as
        # ``ita_delta_applied_abs_mean`` while computing a SIGNED mean, which on
        # the real pool read 0.0201 and invited the conclusion that the
        # transformation does almost nothing - on a run whose individual rows
        # carry ita_delta of 19.73 and whose delta_b reaches 12.54. The signed
        # mean is near zero because the shifts are symmetric by design, which is
        # evidence the sampler is unbiased, not evidence of a weak dose.
        #
        # Both are kept and correctly named: the signed mean shows the sampler
        # is centred, the absolute mean shows the typical magnitude actually
        # applied. Manifests written before this date carry only the old key
        # with the signed value; do not compare them across the rename.
        "ita_delta_applied_mean": _mean("ita_delta_applied"),
        "ita_delta_applied_abs_mean": _abs_mean("ita_delta_applied"),
        "ita_delta_applied_abs_max": _abs_max("ita_delta_applied"),
        "delta_l_requested_mean": _mean("delta_l_requested"),
        "delta_l_realized_mean": _mean("delta_l_realized"),
        "delta_l_realized_abs_max": _abs_max("delta_l_realized"),
        "delta_b_requested_mean": _mean("delta_b_requested"),
        "delta_b_realized_mean": _mean("delta_b_realized"),
        "delta_b_realized_abs_max": _abs_max("delta_b_realized"),
        "gamut_clipped_pixels_total": sum(int(r.get("gamut_clipped_pixels", 0)) for r in applied),
        "images_with_gamut_clipping": sum(
            1 for r in applied if int(r.get("gamut_clipped_pixels", 0)) > 0
        ),
        "b_sign_flipped_pixels_total": sum(
            int(r.get("b_sign_flipped_pixels", 0)) for r in applied
        ),
        "images_with_b_sign_flip": sum(
            1 for r in applied if int(r.get("b_sign_flipped_pixels", 0)) > 0
        ),
    }
