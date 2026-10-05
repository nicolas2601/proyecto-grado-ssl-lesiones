"""Preparation of a raw saliency map: range, sign and per-pixel map validation.

Split out of ``uncertainty_module`` so each file stays small. Nothing here
decides labels: it only puts a class-agnostic map into [0, 1] and fixes its
arbitrary sign before the thresholds are applied.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

DEFAULT_BORDER_FRACTION = 0.1


def as_field(
    field: np.ndarray | None, shape: tuple[int, int], name: str = "field"
) -> np.ndarray | None:
    """Validate an optional per-pixel map and return it as a boolean array."""
    if field is None:
        return None
    field = np.asarray(field).astype(bool)
    if field.shape != shape:
        raise ValueError(f"{name} shape {field.shape} does not match saliency shape {shape}")
    return field


def normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [0, 1]; a constant map becomes zeros."""
    saliency = np.asarray(saliency, dtype=np.float32)
    span = float(saliency.max() - saliency.min())
    if span <= 0.0:
        return np.zeros_like(saliency)
    return (saliency - saliency.min()) / span


def _frame_ring_and_center(shape: tuple[int, int], border_fraction: float):
    height, width = shape
    margin_y = max(1, int(round(height * border_fraction)))
    margin_x = max(1, int(round(width * border_fraction)))
    border = np.ones(shape, dtype=bool)
    border[margin_y:-margin_y, margin_x:-margin_x] = False
    center = np.zeros(shape, dtype=bool)
    center[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True
    return border, center


def _field_ring_and_center(field: np.ndarray, border_fraction: float):
    """Ring just inside the field edge, and the inner core of the field."""
    distance = ndimage.distance_transform_edt(field)
    radius = math.sqrt(float(field.sum()) / math.pi)
    border = field & (distance <= max(1.0, border_fraction * radius))
    center = field & (distance > 0.5 * radius)
    return border, center


def orient_saliency(
    saliency: np.ndarray,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
    field: np.ndarray | None = None,
) -> np.ndarray:
    """Flip the map (1 - s) when the border is brighter than the center.

    A class-agnostic map has an arbitrary sign; lesions are centered and skin
    fills the border, so the border-versus-center contrast fixes the sign
    without any label. Expects a map already in [0, 1].

    On a vignetted image the frame is black, so the frame ring carries no skin
    and the sign never flips. With a ``field`` the ring is taken just inside
    the field edge instead, where the skin actually is.
    """
    field = as_field(field, saliency.shape)
    border, center = None, None
    if field is not None:
        border, center = _field_ring_and_center(field, border_fraction)
        if not (border.any() and center.any()):
            border = None
    if border is None:
        border, center = _frame_ring_and_center(saliency.shape, border_fraction)
    if saliency[border].mean() > saliency[center].mean():
        return 1.0 - saliency
    return saliency
