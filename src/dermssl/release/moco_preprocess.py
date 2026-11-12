"""Exact preprocessing for the MoCo classifier release."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_image(image: Image.Image, size: int = 224) -> np.ndarray:
    """EXIF-correct, RGB-convert and bicubic-squash to normalized NCHW."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize((size, size), Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    pixels = (pixels - _MEAN) / _STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
