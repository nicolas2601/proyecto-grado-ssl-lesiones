"""Lesion-guided multi-crop sampling (thesis novelty 3). Label-free.

Why this module exists
----------------------
DINO's multi-crop samples local views UNIFORMLY. In dermoscopy the lesion is
roughly centered and the clinical signal concentrates at the BORDER (the B in
the ABCD rule: border irregularity). v6 biases local crops toward the border
ring of the lesion, computed from pixels alone (Otsu threshold), never from a
label. When the segmentation is degenerate (no lesion, or "lesion" covering
the frame) the sampler falls back to uniform - guided cropping must only ever
add signal, never crash a run on an atypical image.

Pure numpy, no scipy: morphology is implemented with array shifts so the code
that runs on the cluster is the code tested here.
"""

from __future__ import annotations

import numpy as np

# Outside these mask-area bounds the segmentation is considered degenerate.
MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.90


def otsu_threshold(gray: np.ndarray, bins: int = 256) -> float:
    """Classic Otsu: the threshold maximizing between-class variance."""
    hist, edges = np.histogram(gray.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5

    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    centers = (edges[:-1] + edges[1:]) / 2
    cum_mean = np.cumsum(hist * centers)
    grand_mean = cum_mean[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.where(valid, cum_mean / np.maximum(weight_bg, 1e-12), 0.0)
    mean_fg = np.where(
        valid, (grand_mean - cum_mean) / np.maximum(weight_fg, 1e-12), 0.0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return float(centers[int(between.argmax())])


def lesion_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of the lesion: pixels darker than the Otsu threshold."""
    return gray < otsu_threshold(gray)


def _shift_or(mask: np.ndarray) -> np.ndarray:
    """One 4-neighborhood dilation step via padded shifts."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _shift_and(mask: np.ndarray) -> np.ndarray:
    """One 4-neighborhood erosion step via padded shifts."""
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def border_ring(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Ring of ``width`` dilation/erosion steps around the mask boundary."""
    dilated, eroded = mask, mask
    for _ in range(width):
        dilated = _shift_or(dilated)
        eroded = _shift_and(eroded)
    return dilated & ~eroded


def sample_centers(
    rng: np.random.Generator, region: np.ndarray, n: int
) -> np.ndarray:
    """``n`` (row, col) centers drawn uniformly from the True pixels of ``region``."""
    coords = np.argwhere(region)
    if len(coords) == 0:
        raise ValueError("cannot sample from an empty region")
    picks = rng.integers(0, len(coords), size=n)
    return coords[picks]


def guided_crop_centers(
    gray: np.ndarray, rng: np.random.Generator, n: int, ring_width: int = 3
) -> tuple[np.ndarray, str]:
    """Centers for local crops plus a note saying which strategy fired.

    The note is logged per-image on the cluster so the fallback rate is
    visible in MLflow: a silent 100% fallback would mean the novelty was
    never actually exercised.
    """
    mask = lesion_mask(gray)
    area = mask.mean()
    if MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION:
        ring = border_ring(mask, ring_width)
        if ring.any():
            return sample_centers(rng, ring, n), "border_ring"
    everywhere = np.ones_like(gray, dtype=bool)
    return sample_centers(rng, everywhere, n), "fallback_uniform"


def crop_slices(
    center: tuple[int, int], size: int, shape: tuple[int, int]
) -> tuple[slice, slice]:
    """Square crop of ``size`` around ``center``, clamped inside ``shape``."""
    half = size // 2
    row = int(np.clip(center[0] - half, 0, max(shape[0] - size, 0)))
    col = int(np.clip(center[1] - half, 0, max(shape[1] - size, 0)))
    return slice(row, row + size), slice(col, col + size)
