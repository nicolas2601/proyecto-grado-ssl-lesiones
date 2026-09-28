"""Optional per-image arrays of the FST manifest, validated against the masks.

Both live in the cache directory next to ``manifest.csv``: ``field.npy`` (the
dermoscope field) and ``images.npy`` (the pixels the cache stage wrote, hair
removed when the cache was built with ``--hair-removal``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RGB_CHANNELS = 3


def load_field(path: Path | None, masks: np.ndarray) -> np.ndarray | None:
    """``field.npy`` (N, H, W) memmap, or None when not requested."""
    if path is None:
        return None
    field = np.load(path, mmap_mode="r")
    if field.shape != masks.shape:
        raise ValueError(f"field {field.shape} does not match pseudo_masks {masks.shape}")
    return field


def load_cache_images(path: Path | None, masks: np.ndarray) -> np.ndarray | None:
    """``images.npy`` (N, H, W, 3) uint8 memmap aligned with the masks, or None."""
    if path is None:
        return None
    images = np.load(path, mmap_mode="r")
    if images.dtype != np.uint8:
        raise ValueError(f"cache images must be uint8, got {images.dtype}")
    expected = (*masks.shape, RGB_CHANNELS)
    if images.shape != expected:
        raise ValueError(f"cache images {images.shape} do not match pseudo_masks {expected}")
    return images


def cache_pixels(images: np.ndarray, index: int) -> np.ndarray:
    """One cache frame as float32 RGB in [0, 1], the estimator's input range."""
    return np.asarray(images[index], dtype=np.float32) / 255.0
