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
"""

from __future__ import annotations

import numpy as np

# Groh et al. empirical ITA thresholds (degrees), descending. Bucket 1 is the
# lightest FST, bucket 6 the darkest.
FST_THRESHOLDS = (55.0, 41.0, 28.0, 19.0, 10.0)

# Below this fraction of skin pixels, the estimate is not trustworthy.
MIN_SKIN_FRACTION = 0.05

# Mirrors pretrain.lesion_guided's degenerate-mask bounds: outside this
# range a mask is not a plausible single-lesion segmentation.
MIN_MASK_AREA_FRACTION = 0.01
MAX_MASK_AREA_FRACTION = 0.90

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

    Uses ``arctan2`` rather than plain ``arctan`` so the sign of ``b*`` is
    preserved instead of raising/NaN-ing on ``b* == 0`` - matching the
    "impossible to leave [-90, 90]" invariant from design.md's error table.
    """
    return np.degrees(np.arctan2(np.asarray(L) - 50.0, np.asarray(b)))


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
