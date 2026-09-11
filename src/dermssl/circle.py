"""Stage 2 mitigation: CIRCLe-style LAB color transformer, FST-shift sampling,
and synthetic-blend policy, replicated at the trainable-probe level only
(Pakzad et al., arXiv:2208.13528; Corbin & Marques, IEEE Access 2023).

Why this module exists, and what it is NOT
--------------------------------------------
Per proposal.md, Stage 2 is "gated on Stage 1 showing measurable bias" - no real
Stage 1 Kaggle run has happened yet (PR3's ``scripts/kaggle_fst_manifest.py`` is
still a manual step pending a GPU session), so there is no evidence yet that
bias mitigation is even needed. This module is Stage 2's MACHINERY: a
deterministic color transform, a bounded FST-shift sampler, and a blend
policy - built and tested ahead of time so it is ready to use the moment a real
``fst_manifest.csv`` exists. Nothing here measures or claims a real fairness
improvement; :func:`should_run_stage2` is the explicit stage-gate that keeps
this module from running against real data until Stage 1 says it is worth it
(spec.md, circle-adaptation, "Stage-Gate Dependency on Stage 1").

Pure numpy, no scikit-image at runtime - same reasoning as fitzpatrick.py's D2
in design.md: ``src/dermssl/__init__.py::MODULE_ORDER`` modules are inlined and
executed by the notebook-build test, so a module-level ``import skimage`` here
would break both that test and the Kaggle cell.
"""

from __future__ import annotations

import numpy as np

from dermssl.fitzpatrick import (
    FST_THRESHOLDS,
    MAX_MASK_AREA_FRACTION,
    MIN_MASK_AREA_FRACTION,
    srgb_to_lab,
)
from dermssl.gates import FAIL

# Maximum FST-bucket shift a synthetic sample may request (spec.md req 4:
# "MUST NOT allow a synthetic shift of more than 2 FST levels from the source
# image's estimated bucket").
MAX_FST_SHIFT = 2

# L* units shifted per FST-bucket level. A heuristic magnitude, not a measured
# one: darker target buckets (higher bucket number) move L* down. Chosen so a
# 1-bucket shift is visually apparent without pushing L* out of [0, 100] for a
# 2-bucket shift on a typical skin-tone swatch.
FST_SHIFT_L_STEP = 8.0

# D65 illuminant, 2-degree observer - MUST match fitzpatrick.py's private
# ``_D65_X/Y/ZN`` and ``_SRGB_TO_XYZ`` bit-for-bit, or lab_to_srgb stops being
# the inverse of fitzpatrick.srgb_to_lab (tests/test_circle.py's round-trip
# test pins this down). Duplicated rather than imported because fitzpatrick.py
# keeps those constants private (leading underscore) by design.
_D65_XN = 0.95047
_D65_YN = 1.0
_D65_ZN = 1.08883

_SRGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)


def _f_inv(t: np.ndarray) -> np.ndarray:
    """Inverse of fitzpatrick.py's ``_f`` (CIE Lab nonlinearity), same
    piecewise breakpoint (0.008856) so the round trip matches skimage's
    ``lab2xyz`` convention within the same tolerance ``srgb_to_lab`` holds."""
    cube = t**3
    linear = (t - 16.0 / 116.0) / 7.787
    return np.where(cube > 0.008856, cube, linear)


def _gamma_encode(linear: np.ndarray) -> np.ndarray:
    """Linear-light [0, 1] -> sRGB companded [0, 1]. Inverse of fitzpatrick's
    ``_inverse_gamma``."""
    low = linear * 12.92
    high = 1.055 * np.power(np.clip(linear, 1e-12, None), 1.0 / 2.4) - 0.055
    return np.where(linear <= 0.0031308, low, high)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab back to sRGB, float in [0, 1], shape (..., 3).

    Inverse of ``fitzpatrick.srgb_to_lab`` (D65, 2-degree observer). Output is
    clipped to a valid [0, 1] range: an arbitrary Lab triple is not always a
    representable sRGB color, and silently returning out-of-gamut values would
    violate the "valid LAB/RGB range" contract this module is tested against.
    """
    lab = np.asarray(lab, dtype=float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xr, yr, zr = _f_inv(fx), _f_inv(fy), _f_inv(fz)
    xyz = np.stack([xr * _D65_XN, yr * _D65_YN, zr * _D65_ZN], axis=-1)

    linear = xyz @ _XYZ_TO_SRGB.T
    linear = np.clip(linear, 0.0, None)  # guard tiny negative float noise
    srgb = _gamma_encode(linear)
    return np.clip(srgb, 0.0, 1.0)


def clamp_fst_shift(requested_shift: int, *, max_shift: int = MAX_FST_SHIFT) -> int:
    """Clamp a requested FST-bucket shift to ``[-max_shift, max_shift]``.

    spec.md req 4: "MUST clamp the shift to the maximum allowed ... and MUST
    NOT silently apply the larger shift" - this is the explicit clamp, never a
    silent pass-through of an out-of-bound request.
    """
    if requested_shift > max_shift:
        return max_shift
    if requested_shift < -max_shift:
        return -max_shift
    return requested_shift


def target_fst_bucket(
    source_bucket: int, requested_shift: int, *, max_shift: int = MAX_FST_SHIFT
) -> int:
    """Resulting FST bucket (1..6) after a possibly-clamped shift.

    spec.md scenario: source bucket 2, requested shift 2 -> target bucket 4.
    A requested shift beyond ``max_shift`` clamps first (see
    :func:`clamp_fst_shift`), then the result is bounded to the valid bucket
    range so a shift near the ends of the scale cannot produce bucket 0 or 7.
    """
    shift = clamp_fst_shift(requested_shift, max_shift=max_shift)
    n_buckets = len(FST_THRESHOLDS) + 1
    return int(min(max(source_bucket + shift, 1), n_buckets))


def _is_degenerate_mask(mask: np.ndarray) -> bool:
    """Same bounds as fitzpatrick.py's private ``_is_degenerate`` (D3: one
    mask contract, two producers)."""
    area = float(mask.mean())
    return not (MIN_MASK_AREA_FRACTION <= area <= MAX_MASK_AREA_FRACTION)


def shift_skin_tone(
    img: np.ndarray, mask: np.ndarray | None, shift_buckets: int, *, max_shift: int = MAX_FST_SHIFT
) -> np.ndarray:
    """Deterministic LAB color transformer: shifts skin-pixel L* by
    ``shift_buckets`` FST-bucket levels; lesion pixels (``mask`` True) are
    passed through pixel-for-pixel unchanged.

    ``img`` is sRGB float (H, W, 3) in [0, 1]. ``mask`` is the same contract
    as ``fitzpatrick.estimate_fst``: bool (H, W), True = lesion; a non-bool
    mask is coerced (mirrors fitzpatrick's own dtype guard so ``~mask`` stays
    logical negation, not bitwise NOT on raw integers).

    Raises ``ValueError`` when the mask is missing or degenerate (area outside
    1%-90%) instead of silently transforming the whole frame - spec.md req 2:
    "MUST NOT silently apply an unprotected color shift over the lesion".
    """
    if mask is None:
        raise ValueError("shift_skin_tone requires an explicit lesion mask")
    mask = np.asarray(mask, dtype=bool)
    if _is_degenerate_mask(mask):
        raise ValueError(
            f"degenerate lesion mask (area={float(mask.mean()):.4f}); "
            "cannot protect the lesion region"
        )

    shift = clamp_fst_shift(shift_buckets, max_shift=max_shift)
    img = np.asarray(img, dtype=float)
    lab = srgb_to_lab(img)

    shifted_L = np.clip(lab[..., 0] - shift * FST_SHIFT_L_STEP, 0.0, 100.0)
    shifted_lab = lab.copy()
    shifted_lab[..., 0] = np.where(mask, lab[..., 0], shifted_L)

    out = lab_to_srgb(shifted_lab)
    out = np.where(mask[..., None], img, out)  # exact passthrough over the lesion
    return np.clip(out, 0.0, 1.0)


def synthetic_blend_mask(
    n_samples: int, *, target_ratio: float = 0.5, seed: int = 42
) -> np.ndarray:
    """Boolean mask (length ``n_samples``) selecting which samples get an
    FST-shifted synthetic counterpart, sized so ``target_ratio`` of the batch
    is synthetic (spec.md req 4, default ~50/50). Deterministic for a given
    ``seed``.
    """
    if not (0.0 <= target_ratio <= 1.0):
        raise ValueError(f"target_ratio must be within [0, 1], got {target_ratio!r}")

    n_synthetic = int(round(n_samples * target_ratio))
    mask = np.zeros(n_samples, dtype=bool)
    if n_synthetic == 0 or n_samples == 0:
        return mask
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_samples, size=n_synthetic, replace=False)
    mask[idx] = True
    return mask


def should_run_stage2(gate_status: str) -> bool:
    """Stage-gate dependency on Stage 1 (spec.md req 1): Stage 2 is only
    required to run when Stage 1's fairness gate reports ``FAIL``. It is
    skippable - not forced - when Stage 1 reports ``PASS`` or ``SKIP``.
    """
    return gate_status == FAIL
