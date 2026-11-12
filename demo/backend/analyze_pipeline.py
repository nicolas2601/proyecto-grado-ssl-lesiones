"""Segmentation and photometry for /api/analyze. Never raises: failures become contract reasons.

The classifier input is never touched here; these blocks only describe the image.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .photometry import CATEGORY_SCHEME, REGION, measure_photometry
from .segmenter_service import THRESHOLD, SegmenterError

PHOTOMETRY_ERROR = "photometry_error"
SEGMENTER_ERROR = "segmenter_error"
_MS_PER_SECOND = 1000


def _ms(seconds: float) -> int:
    return int(round(seconds * _MS_PER_SECOND))


def _unavailable_segmentation(reason: str, width: int, height: int) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "image_width": width,
        "image_height": height,
        "mask_width": None,
        "mask_height": None,
        "mask_png_base64": None,
        "contour": None,
        "area_fraction": None,
        "threshold": THRESHOLD,
        "identity": None,
    }


def _segmentation(segmenter: Any, rgb: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
    height, width = rgb.shape[:2]
    try:
        result = segmenter.segment(rgb)
    except SegmenterError as exc:
        return _unavailable_segmentation(exc.reason, width, height), None
    except Exception:  # noqa: BLE001 - segmentation must never fail the request
        return _unavailable_segmentation(SEGMENTER_ERROR, width, height), None
    return {"status": "ok", "reason": None, **result.payload}, result.full_mask


def _unavailable_photometry(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "region": REGION,
        "category_scheme": CATEGORY_SCHEME,
        "median_L": None,
        "median_b": None,
        "ita_deg": None,
        "ita_category": None,
        "n_px": None,
        "valid_fraction": None,
    }


def preprocessing_of(segmenter: Any) -> Any:
    """The segmenter's preprocessing config when it is loaded; None otherwise (no hair exclusion then)."""
    getter = getattr(segmenter, "preprocessing", None)
    return getter() if callable(getter) else None


def _photometry(rgb: np.ndarray, mask: np.ndarray | None, hair: Any = None) -> dict[str, Any]:
    try:
        return measure_photometry(rgb, mask, hair=hair)
    except Exception:  # noqa: BLE001 - photometry must never fail the request
        return _unavailable_photometry(PHOTOMETRY_ERROR)


def run_extras(segmenter: Any, rgb: np.ndarray) -> dict[str, Any]:
    """Segmentation block, photometry block and their timings for one uint8 RGB image."""
    started = perf_counter()
    segmentation, mask = _segmentation(segmenter, rgb)
    segmented = perf_counter()
    photometry = _photometry(rgb, mask, preprocessing_of(segmenter))
    finished = perf_counter()
    return {
        "segmentation": segmentation,
        "photometry": photometry,
        "timing_ms": {"segmentation": _ms(segmented - started), "photometry": _ms(finished - segmented)},
    }
