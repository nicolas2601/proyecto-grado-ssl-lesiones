"""Uncertainty Module (UM): saliency map -> pseudo-mask with an ignore band.

Pure numpy/scipy, in the style of USL-Net (Li et al. 2024, Section 3.4):

* saliency below ``low`` (0.35) is background;
* saliency above ``high`` (0.65) is a lesion candidate;
* the band in between is uncertain and ignored by the loss;
* candidates are restricted to the dermoscope field of view, so the vignette
  can never be voted lesion;
* among candidates, connectivity filtering drops small regions and a centrality
  filter keeps the single region closest to the image center; every other
  candidate region becomes uncertain (ignored), not background;
* a survivor covering more than ``max_area_fraction`` of the field is the field
  itself, not a lesion, and the sample is excluded;
* if nothing survives, if the survivor is too far from the center, or if the
  field is empty, the whole sample is excluded (all pixels ignored);
* pixels outside the field are background: they are neither skin nor lesion,
  and leaving them uncertain would waste most of the loss on the vignette.

Output labels follow ``cache_io``: ``BACKGROUND``, ``FOREGROUND``, ``IGNORE``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from cache_io import BACKGROUND, FOREGROUND, IGNORE

DEFAULT_LOW = 0.35
DEFAULT_HIGH = 0.65
# USL-Net uses alpha = 500 pixels at 224x224 (about 1 percent of the image).
DEFAULT_MIN_AREA_FRACTION = 500 / (224 * 224)
# Fraction of the half-diagonal; farther regions are treated as border noise.
DEFAULT_MAX_CENTER_DISTANCE = 0.5
# A region covering more than this fraction of the field is the field itself.
DEFAULT_MAX_AREA_FRACTION = 0.6
_EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=int)

REASON_VALID = ""
REASON_NO_REGION = "no_region"
REASON_NO_FIELD = "no_field"
REASON_AREA = "area"


@dataclass(frozen=True)
class PseudoMaskResult:
    mask: np.ndarray
    valid: bool
    reason: str = REASON_VALID


def _excluded(shape: tuple[int, int], reason: str) -> PseudoMaskResult:
    """An excluded sample: every pixel ignored, with the reason attached."""
    return PseudoMaskResult(
        mask=np.full(shape, IGNORE, dtype=np.uint8), valid=False, reason=reason
    )


def as_field(field: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Validate an optional field of view and return it as a boolean array."""
    if field is None:
        return None
    field = np.asarray(field).astype(bool)
    if field.shape != shape:
        raise ValueError(f"field shape {field.shape} does not match saliency shape {shape}")
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
    border_fraction: float = 0.1,
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


def select_central_region(
    candidates: np.ndarray, min_area: int, max_center_distance: float
) -> np.ndarray | None:
    """Keep the connected region whose centroid is closest to the image center.

    Regions smaller than ``min_area`` pixels are discarded first. The centroid
    distance is normalized by the half-diagonal; a winner farther than
    ``max_center_distance`` means the sample has no trustworthy lesion.
    """
    labels, count = ndimage.label(candidates, structure=_EIGHT_CONNECTIVITY)
    if count == 0:
        return None
    height, width = candidates.shape
    center = np.array([(height - 1) / 2.0, (width - 1) / 2.0])
    half_diagonal = math.hypot(height, width) / 2.0

    areas = np.bincount(labels.ravel())[1:]
    centroids = ndimage.center_of_mass(candidates, labels, index=range(1, count + 1))
    best_label, best_key = None, None
    for label_index, (area, centroid) in enumerate(zip(areas, centroids, strict=True), start=1):
        if area < min_area:
            continue
        distance = float(np.linalg.norm(np.asarray(centroid) - center)) / half_diagonal
        key = (distance, -area)
        if best_key is None or key < best_key:
            best_label, best_key = label_index, key
    if best_label is None or best_key[0] > max_center_distance:
        return None
    return labels == best_label


def saliency_to_pseudo_mask(
    saliency: np.ndarray,
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    max_center_distance: float = DEFAULT_MAX_CENTER_DISTANCE,
    max_area_fraction: float = DEFAULT_MAX_AREA_FRACTION,
    field: np.ndarray | None = None,
    normalize: bool = True,
) -> PseudoMaskResult:
    """Convert one saliency map into a three-valued pseudo-mask.

    ``field`` is the dermoscope field of view (bool, True = usable pixel); both
    area limits are fractions of its area, or of the frame when it is absent.
    """
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"thresholds must satisfy 0 <= low < high <= 1, got {low}, {high}")
    saliency = normalize_saliency(saliency) if normalize else np.asarray(saliency, np.float32)
    field = as_field(field, saliency.shape)
    reference = float(saliency.size) if field is None else float(field.sum())
    if reference <= 0.0:
        return _excluded(saliency.shape, REASON_NO_FIELD)

    candidates = saliency > high
    if field is not None:
        candidates = candidates & field
    min_area = max(1, math.ceil(min_area_fraction * reference))
    foreground = select_central_region(candidates, min_area, max_center_distance)
    if foreground is None:
        return _excluded(saliency.shape, REASON_NO_REGION)
    if float(foreground.sum()) / reference > max_area_fraction:
        return _excluded(saliency.shape, REASON_AREA)

    mask = np.full(saliency.shape, IGNORE, dtype=np.uint8)
    mask[saliency < low] = BACKGROUND
    if field is not None:
        mask[~field] = BACKGROUND
    mask[foreground] = FOREGROUND
    return PseudoMaskResult(mask=mask, valid=True)


def _convert_one(
    raw: np.ndarray,
    field: np.ndarray | None,
    *,
    orient: bool,
    normalize: bool,
    mask_kwargs: dict,
) -> PseudoMaskResult:
    normalized = normalize_saliency(raw) if normalize else raw
    if orient:
        normalized = orient_saliency(normalized, field=field)
    return saliency_to_pseudo_mask(normalized, normalize=False, field=field, **mask_kwargs)


def saliency_array_to_pseudo_masks(
    saliency: np.ndarray,
    out: np.ndarray,
    *,
    orient: bool,
    normalize: bool = True,
    chunk_size: int = 256,
    fields: np.ndarray | None = None,
    **mask_kwargs,
) -> dict:
    """Convert an ``(N, H, W)`` saliency array into ``out`` and report statistics.

    ``orient`` applies the border-versus-center sign fix and ``normalize`` the
    per-image min-max; use both for raw CCAM maps. Student predictions are
    already probabilities with a learned sign, so they use neither.
    ``fields`` is the optional ``(N, H, W)`` field of view cache.
    """
    total = len(saliency)
    chunk_size = max(1, chunk_size)
    reasons = {REASON_NO_REGION: 0, REASON_NO_FIELD: 0, REASON_AREA: 0}
    fractions = {FOREGROUND: 0.0, IGNORE: 0.0}
    for start in range(0, total, chunk_size):
        chunk = np.asarray(saliency[start : start + chunk_size], dtype=np.float32)
        field_chunk = None if fields is None else np.asarray(fields[start : start + chunk_size])
        for offset, raw in enumerate(chunk):
            field = None if field_chunk is None else field_chunk[offset]
            result = _convert_one(
                raw, field, orient=orient, normalize=normalize, mask_kwargs=mask_kwargs
            )
            out[start + offset] = result.mask
            if not result.valid:
                reasons[result.reason] = reasons.get(result.reason, 0) + 1
            for label in fractions:
                fractions[label] += float((result.mask == label).mean())
    return _summarize(total, reasons, fractions, field_used=fields is not None)


def _summarize(total: int, reasons: dict, fractions: dict, *, field_used: bool) -> dict:
    """Totals per exclusion reason; unknown reasons still count towards the total."""
    excluded = sum(reasons.values())
    scale = total if total else 1
    return {
        "images": total,
        "excluded": excluded,
        "excluded_no_region": reasons.get(REASON_NO_REGION, 0),
        "excluded_no_field": reasons.get(REASON_NO_FIELD, 0),
        "excluded_by_area": reasons.get(REASON_AREA, 0),
        "excluded_fraction": excluded / scale if total else 0.0,
        "mean_foreground_fraction": fractions[FOREGROUND] / scale if total else 0.0,
        "mean_ignore_fraction": fractions[IGNORE] / scale if total else 0.0,
        "field_used": field_used,
    }
