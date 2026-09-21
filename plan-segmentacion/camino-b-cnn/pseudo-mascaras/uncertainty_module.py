"""Uncertainty Module (UM): saliency map -> pseudo-mask with an ignore band.

Pure numpy/scipy, in the style of USL-Net (Li et al. 2024, Section 3.4):

* saliency below ``low`` (0.35) is background;
* saliency above ``high`` (0.65) is a lesion candidate;
* the band in between is uncertain and ignored by the loss;
* among candidates, connectivity filtering drops small regions and a centrality
  filter keeps the single region closest to the image center; every other
  candidate region becomes uncertain (ignored), not background;
* if nothing survives, or the survivor is too far from the center, the whole
  sample is excluded (all pixels ignored).

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
_EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=int)


@dataclass(frozen=True)
class PseudoMaskResult:
    mask: np.ndarray
    valid: bool


def normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [0, 1]; a constant map becomes zeros."""
    saliency = np.asarray(saliency, dtype=np.float32)
    span = float(saliency.max() - saliency.min())
    if span <= 0.0:
        return np.zeros_like(saliency)
    return (saliency - saliency.min()) / span


def orient_saliency(saliency: np.ndarray, border_fraction: float = 0.1) -> np.ndarray:
    """Flip the map (1 - s) when the border is brighter than the center.

    A class-agnostic map has an arbitrary sign; lesions are centered and skin
    fills the border, so the border-versus-center contrast fixes the sign
    without any label. Expects a map already in [0, 1].
    """
    height, width = saliency.shape
    margin_y = max(1, int(round(height * border_fraction)))
    margin_x = max(1, int(round(width * border_fraction)))
    border = np.ones_like(saliency, dtype=bool)
    border[margin_y:-margin_y, margin_x:-margin_x] = False
    center = np.zeros_like(border)
    center[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True
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
    normalize: bool = True,
) -> PseudoMaskResult:
    """Convert one saliency map into a three-valued pseudo-mask."""
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"thresholds must satisfy 0 <= low < high <= 1, got {low}, {high}")
    saliency = normalize_saliency(saliency) if normalize else np.asarray(saliency, np.float32)
    min_area = max(1, math.ceil(min_area_fraction * saliency.size))

    foreground = select_central_region(saliency > high, min_area, max_center_distance)
    mask = np.full(saliency.shape, IGNORE, dtype=np.uint8)
    if foreground is None:
        return PseudoMaskResult(mask=mask, valid=False)
    mask[saliency < low] = BACKGROUND
    mask[foreground] = FOREGROUND
    return PseudoMaskResult(mask=mask, valid=True)


def saliency_array_to_pseudo_masks(
    saliency: np.ndarray,
    out: np.ndarray,
    *,
    orient: bool,
    normalize: bool = True,
    chunk_size: int = 256,
    **mask_kwargs,
) -> dict:
    """Convert an ``(N, H, W)`` saliency array into ``out`` and report statistics.

    ``orient`` applies the border-versus-center sign fix and ``normalize`` the
    per-image min-max; use both for raw CCAM maps. Student predictions are
    already probabilities with a learned sign, so they use neither.
    """
    total = len(saliency)
    excluded = 0
    foreground_fraction_sum = 0.0
    ignore_fraction_sum = 0.0
    for start in range(0, total, chunk_size):
        chunk = np.asarray(saliency[start : start + chunk_size], dtype=np.float32)
        for offset, raw in enumerate(chunk):
            normalized = normalize_saliency(raw) if normalize else raw
            if orient:
                normalized = orient_saliency(normalized)
            result = saliency_to_pseudo_mask(normalized, normalize=False, **mask_kwargs)
            out[start + offset] = result.mask
            excluded += 0 if result.valid else 1
            foreground_fraction_sum += float((result.mask == FOREGROUND).mean())
            ignore_fraction_sum += float((result.mask == IGNORE).mean())
    return {
        "images": total,
        "excluded": excluded,
        "excluded_fraction": excluded / total if total else 0.0,
        "mean_foreground_fraction": foreground_fraction_sum / total if total else 0.0,
        "mean_ignore_fraction": ignore_fraction_sum / total if total else 0.0,
    }
