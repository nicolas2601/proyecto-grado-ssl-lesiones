"""Mask-guided ITA-based Fitzpatrick Skin Type (FST) estimation.

Why this module exists
-----------------------
The repo documents skin-tone bias as a known limit (HAM10000 skews FST
I-III) but never measures it. This module turns that limit into a number:
every image gets an FST bucket (1-6) or an explicit "unavailable" marker,
computed from skin pixels only (the lesion is excluded via a mask).

Pure numpy, no scikit-image at runtime (D2 in design.md). ``scikit-image``
is a *test oracle* only, used in ``tests/test_fitzpatrick.py`` to check the
LAB conversion against ``skimage.color.rgb2lab`` within ``atol=1e-4`` -
never imported here. ``src/dermssl/__init__.py::MODULE_ORDER`` modules are
inlined and executed by the notebook-build test, so a module-level
``import skimage`` here would break both that test and the Kaggle cell.

The ``lesion_mask`` import below is used as a bare name (never
``lesion_guided.lesion_mask``) on purpose: ``pretrain/lesion_guided`` is
inlined into the same notebook cell, immediately before this module (see
``MODULE_ORDER``), and ``scripts/build_notebook_v5.py``'s
``module_source()`` strips this import line during inlining - by the time
this file's code runs in the notebook, ``lesion_mask`` already exists as a
plain function in that shared namespace. A package-qualified reference
would leave a dangling ``lesion_guided`` name and break the Kaggle cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dermssl.pretrain.lesion_guided import lesion_mask

# Groh et al. empirical ITA thresholds (degrees), descending. Bucket 1 is the
# lightest FST, bucket 6 the darkest.
FST_THRESHOLDS = (55.0, 41.0, 28.0, 19.0, 10.0)

# Below this fraction of skin pixels, the estimate is not trustworthy.
MIN_SKIN_FRACTION = 0.05

# Mirrors pretrain.lesion_guided's degenerate-mask bounds: outside this
# range a mask is not a plausible single-lesion segmentation.
MIN_MASK_AREA_FRACTION = 0.01
MAX_MASK_AREA_FRACTION = 0.90

# Near-black pixels are treated as image border/vignette artifacts (common
# in dermoscopy captures), not skin - excluded regardless of the lesion mask
# (spec.md: "Skin Pixel Selection" MUST exclude border/vignette too).
BORDER_LUMINANCE_THRESHOLD = 0.02

# D65 reference white, 2-degree observer. Matches skimage's
# ``xyz_tristimulus_values(illuminant="D65", observer="2")`` exactly so the
# LAB oracle test can hold at atol=1e-4.
_D65_XN = 0.95047
_D65_YN = 1.0
_D65_ZN = 1.08883

# sRGB -> XYZ (D65). Matches skimage.color.colorconv.xyz_from_rgb exactly.
_SRGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)


def _inverse_gamma(channel: np.ndarray) -> np.ndarray:
    """sRGB companding removal: gamma-encoded [0, 1] -> linear-light [0, 1]."""
    linear_low = channel / 12.92
    linear_high = ((channel + 0.055) / 1.055) ** 2.4
    return np.where(channel <= 0.04045, linear_low, linear_high)


def _f(t: np.ndarray) -> np.ndarray:
    """CIE Lab nonlinearity, piecewise cube-root with a linear floor near 0.

    Uses the same rounded constants as ``skimage.color.xyz2lab`` (0.008856,
    7.787, 16/116) rather than the more "exact" (6/29)**3 form, so the two
    implementations agree within the LAB oracle test's ``atol=1e-4``.
    """
    cube_root = np.cbrt(t)
    linear = 7.787 * t + 16.0 / 116.0
    return np.where(t > 0.008856, cube_root, linear)


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert an sRGB image, float in [0, 1], shape (H, W, 3), to CIE Lab.

    D65 illuminant, 2-degree observer - matches ``skimage.color.rgb2lab``
    within ``atol=1e-4`` (see ``tests/test_fitzpatrick.py``).
    """
    linear = _inverse_gamma(img)
    xyz = linear @ _SRGB_TO_XYZ.T

    xr = xyz[..., 0] / _D65_XN
    yr = xyz[..., 1] / _D65_YN
    zr = xyz[..., 2] / _D65_ZN

    fx, fy, fz = _f(xr), _f(yr), _f(zr)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def ita_degrees(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Individual Typology Angle, in degrees: ``arctan((L*-50)/b*) * 180/pi``.

    Plain ``arctan`` of the ratio - the literature formula (Kolm/Chardon,
    Groh et al.) - not ``arctan2`` of the two components: ``arctan2`` is
    quadrant-aware and would wrap negative-``b*`` readings into (90, 180],
    which contradicts the "ITA cannot leave [-90, 90]" invariant that
    design.md's error table asserts "by arctan construction".
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.degrees(np.arctan((np.asarray(L) - 50.0) / np.asarray(b)))


def fst_bucket(ita: float) -> int:
    """Map an aggregate ITA (degrees) to a Fitzpatrick bucket (1-6).

    ``FST_THRESHOLDS`` are the Groh et al. empirical cut points, descending.
    A value exactly ON a threshold belongs to the DARKER (higher-numbered)
    bucket - i.e. thresholds are exclusive lower bounds for the lighter
    bucket, matching the ``>`` comparisons in the original ITA-to-FST table.
    """
    bucket = len(FST_THRESHOLDS) + 1
    for i, threshold in enumerate(FST_THRESHOLDS, start=1):
        if ita > threshold:
            bucket = i
            break
    return bucket


@dataclass
class FSTEstimate:
    """Result of one FST estimation call. Never dropped silently (spec.md:
    "Explicit Bucket-or-Unavailable Output") - always either a bucket or an
    explicit ``status`` explaining why one is not available."""

    bucket: int | None  # 1..6, None when unavailable
    ita: float  # NaN when unavailable
    skin_fraction: float
    status: str  # "ok" | "no_skin" | "degenerate_mask" | "ita_out_of_range"


def _is_degenerate(mask: np.ndarray) -> bool:
    """True when ``mask``'s area is outside the plausible single-lesion
    range (1%-90%) - too small to be a real segmentation, or so large it
    likely captured the whole frame instead of one lesion."""
    area = float(mask.mean())
    return not (MIN_MASK_AREA_FRACTION <= area <= MAX_MASK_AREA_FRACTION)


def estimate_fst(img: np.ndarray, mask: np.ndarray | None = None) -> FSTEstimate:
    """Mask-guided FST estimate for one sRGB image, float (H, W, 3) in [0, 1].

    ``mask`` is the primary lesion segmentation (bool (H, W), True = lesion -
    e.g. a U-Net's output in Phase 3). When omitted, or when it is
    degenerate (area outside 1%-90%), the pure-numpy Otsu fallback from
    ``pretrain.lesion_guided`` is used instead (D3: two-tier masking, one
    contract - both return bool (H, W), skin = ~mask).

    A non-bool ``mask`` (e.g. an int/uint8 0/1 array) is coerced to bool:
    downstream code relies on ``~mask`` being logical negation, which on an
    integer dtype would instead be bitwise NOT on the raw integers (wrong
    fancy-indexing into ``lab[skin]``, silently, no error raised).
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    gray = img.mean(axis=-1)
    border = gray < BORDER_LUMINANCE_THRESHOLD

    if mask is None or _is_degenerate(mask):
        mask = lesion_mask(gray)

    if _is_degenerate(mask):
        skin = ~mask & ~border
        return FSTEstimate(
            bucket=None,
            ita=float("nan"),
            skin_fraction=float(skin.mean()),
            status="degenerate_mask",
        )

    skin = ~mask & ~border
    skin_fraction = float(skin.mean())

    if skin_fraction < MIN_SKIN_FRACTION:
        return FSTEstimate(
            bucket=None, ita=float("nan"), skin_fraction=skin_fraction, status="no_skin"
        )

    lab = srgb_to_lab(img)
    skin_lab = lab[skin]
    L_med = np.median(skin_lab[:, 0])
    b_med = np.median(skin_lab[:, 2])
    ita = float(ita_degrees(np.asarray(L_med), np.asarray(b_med)))

    # Finite ITA cannot leave [-90, 90] by arctan construction (see
    # ita_degrees) - but an achromatic skin median at L*=50 makes the
    # (L*-50)/b* ratio a literal 0/0, which is NaN, and NaN comparisons
    # are always False. Check explicitly (never a bare assert, which would
    # both crash the whole batch on one atypical image - the opposite of
    # lesion_guided.py's philosophy - and be silently skipped entirely
    # under python -O / PYTHONOPTIMIZE=1) and degrade gracefully, matching
    # every other "no result" branch in this function and the contract
    # documented on FSTEstimate.status.
    if not (-90.0 <= ita <= 90.0):
        return FSTEstimate(
            bucket=None, ita=ita, skin_fraction=skin_fraction, status="ita_out_of_range"
        )

    return FSTEstimate(
        bucket=fst_bucket(ita), ita=ita, skin_fraction=skin_fraction, status="ok"
    )
