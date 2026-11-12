"""Perilesional skin photometry for /api/analyze (research only).

Reuses ``dermssl.e5_roi`` for the ring, the exclusions, the abstention rules and the
ITA b* guard, and ``dermssl.fitzpatrick`` for sRGB -> CIELAB (D65).

The ITA categories below are a colorimetric scheme (Chardon et al. 1991, as used by
Del Bino et al. 2006, Pigment Cell Research 19:606-614, doi:10.1111/j.1600-0749.2006.00338.x).
They are NOT Fitzpatrick types and NOT the Groh FST buckets (``fitzpatrick.FST_THRESHOLDS``).
Class bounds are half-open: a value exactly on a bound falls in the lower class.
"""

from __future__ import annotations

import numpy as np

from dermssl import e5_roi

from .segmenter_preprocess import Preprocessing, hair_mask_at_image_size

REGION = "perilesional_ring"
CATEGORY_SCHEME = "Chardon 1991 / Del Bino 2006 ITA categories (colorimetric, not Fitzpatrick)"
ITA_INDETERMINATE = "ita_indeterminate"
_RESULT_DECIMALS = 2
# (lower bound, label of the class ABOVE it): ita > bound => label. Anything not above the last bound is "dark".
_ITA_ABOVE = (
    (55.0, "very_light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
)
_UNAVAILABLE = frozenset({e5_roi.NO_MASK, e5_roi.DEGENERATE_MASK, e5_roi.RING_TOO_SMALL})


def ita_category(ita: float | None) -> str | None:
    if ita is None:
        return None
    for bound, label in _ITA_ABOVE:
        if ita > bound:
            return label
    return "dark"


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), _RESULT_DECIMALS)


def _base(estimate: e5_roi.RoiEstimate) -> dict:
    return {
        "region": REGION,
        "category_scheme": CATEGORY_SCHEME,
        "n_px": estimate.n_px,
        "valid_fraction": _round(estimate.valid_fraction),
    }


def _ok(estimate: e5_roi.RoiEstimate) -> dict:
    return {
        **_base(estimate),
        "status": "ok",
        "reason": None,
        "median_L": _round(estimate.L_median),
        "median_b": _round(estimate.b_median),
        "ita_deg": _round(estimate.ita),
        "ita_category": ita_category(estimate.ita),
    }


def _indeterminate(estimate: e5_roi.RoiEstimate) -> dict:
    return {
        **_base(estimate),
        "status": "indeterminate",
        "reason": ITA_INDETERMINATE,
        "median_L": _round(estimate.L_median),
        "median_b": _round(estimate.b_median),
        "ita_deg": None,
        "ita_category": None,
    }


def _unavailable(estimate: e5_roi.RoiEstimate) -> dict:
    return {
        **_base(estimate),
        "status": "unavailable",
        "reason": estimate.status,
        "median_L": None,
        "median_b": None,
        "ita_deg": None,
        "ita_category": None,
    }


def _hair_detector(hair: Preprocessing | None):
    if hair is None or hair.hair is None:
        return None
    return lambda rgb_u8: hair_mask_at_image_size(rgb_u8, hair)


def measure_photometry(rgb: np.ndarray, lesion: np.ndarray | None, hair: Preprocessing | None = None) -> dict:
    """Photometry block of the analyze contract for a uint8 RGB image and an optional lesion mask.

    ``hair`` is the segmenter's preprocessing config; when it has a hair step, hair is excluded from the ring
    with the same kernel and threshold the segmenter uses. A mask on a different grid raises ``ValueError``.
    """
    image = np.asarray(rgb, dtype=np.float64) / 255.0
    hair_fn = _hair_detector(hair)
    estimate = e5_roi.ring_estimate(image, lesion, hair_fn=hair_fn)
    if estimate.status == e5_roi.OK:
        return _ok(estimate)
    if estimate.status in _UNAVAILABLE:
        return _unavailable(estimate)
    return _indeterminate(estimate)
