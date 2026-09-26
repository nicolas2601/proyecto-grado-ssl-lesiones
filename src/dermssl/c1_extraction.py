"""Turning (image, pseudo-mask, condition) into one feature vector.

Split from :mod:`dermssl.mask_conditions` because this half needs torch and the
other half does not: the geometry and pooling rules stay unit-testable without
a GPU, and only the forward pass lives here.

Two deliberate asymmetries, both of which a reviewer should check rather than
assume:

1. ``bbox_crop`` does NOT run Resize(255)+CenterCrop(224). It resizes the crop
   straight to 224. Running the baseline pipeline on an already-cropped region
   would centre-crop the lesion a second time and cut exactly what the
   condition is trying to isolate.
2. ``mask_pool`` reads ``x_norm_patchtokens``; the other three read the pooled
   output. This is the one place where the earlier vgg19-class bug could
   reappear, so the mask given to the pooling must have travelled through the
   SAME geometry as the image -- which is why ``build_eval_input`` returns the
   transformed mask alongside the tensor instead of leaving the caller to
   recompute it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dermssl.c1_mask_provenance import TrackedMask
from dermssl.mask_conditions import (
    CONDITIONS,
    apply_neutral_background,
    bbox_with_margin,
)

RESOLUTION = 224
RESIZE_FACTOR = 1.14
PATCH = 14
GRID = RESOLUTION // PATCH  # 16 -> 256 tokens
DEFAULT_MARGIN = 0.10


@dataclass(frozen=True)
class PreparedInput:
    """An image and its mask after identical geometric treatment."""

    image: np.ndarray  # uint8 HWC, RESOLUTION x RESOLUTION
    mask: np.ndarray  # bool HW, same size as image
    used_bbox_fallback: bool


def _pil_resize(array: np.ndarray, width: int, height: int, nearest: bool) -> np.ndarray:
    """Resize through PIL, which is exactly what torchvision uses on PIL input.

    Hand-rolling bilinear here was a mistake: the bank feeds PIL images into
    ``transforms.Resize``, and PIL applies its own filtering when downsampling.
    A numpy reimplementation lands close but not equal, which is the difference
    between reproducing the published cache and not. Reuse the same library
    rather than re-deriving it.
    """
    from PIL import Image  # noqa: PLC0415 - PIL carries no torch dependency

    if nearest:
        pil = Image.fromarray(array.astype(np.uint8) * 255)
        out = pil.resize((width, height), Image.NEAREST)
        return np.asarray(out) > 127

    return np.asarray(
        Image.fromarray(array).resize((width, height), Image.BILINEAR), dtype=np.uint8
    )


def _resize_shorter_side(array: np.ndarray, size: int) -> np.ndarray:
    """torchvision ``Resize(size)`` semantics: SHORTER side -> size, aspect kept.

    An int argument does NOT mean a square. For a 600x450 dermatoscopic image
    the short side (450) becomes 255 and the long side follows to 340; forcing
    a 255x255 square would distort the lesion and shift what CenterCrop(224)
    keeps.
    """
    height, width = array.shape[:2]
    if width <= height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)

    return _pil_resize(array, new_width, new_height, nearest=array.dtype == np.bool_)


def _resize_square(array: np.ndarray, size: int) -> np.ndarray:
    """Force an exact square. Used only by bbox_crop, which resizes its crop directly."""
    return _pil_resize(array, size, size, nearest=array.dtype == np.bool_)


def _center_crop(array: np.ndarray, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return array[top : top + size, left : left + size]


_INTERMEDIATE = int(RESOLUTION * RESIZE_FACTOR)  # 255, not 256


def _baseline_geometry_image(image: np.ndarray) -> np.ndarray:
    """Resize(255, BILINEAR) then CenterCrop(224) -- the bank's geometry."""
    return _center_crop(_resize_shorter_side(image, _INTERMEDIATE), RESOLUTION)


def _baseline_geometry_mask(mask: np.ndarray) -> np.ndarray:
    """Same geometry as the image, but resampled nearest to stay binary."""
    return _center_crop(_resize_shorter_side(mask, _INTERMEDIATE), RESOLUTION)


def build_eval_input(
    image: np.ndarray,
    lesion: TrackedMask,
    condition: str,
    margin_frac: float = DEFAULT_MARGIN,
) -> PreparedInput:
    """Apply a condition's geometry to the image AND its mask together.

    ``lesion`` must be a :class:`~dermssl.mask_conditions.TrackedMask`, not a
    bare array: that is what keeps a reference mask from being passed in and
    quietly turning C1 into a supervised experiment.
    """
    if not isinstance(lesion, TrackedMask):
        raise TypeError(
            "lesion must be a TrackedMask carrying its step-0 manifest, not a bare "
            "array. Provenance is what enforces the SSL rule here."
        )
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {CONDITIONS}. "
            "A typo used to fall through to the baseline and mislabel the cell."
        )

    mask_array = lesion.array
    if image.shape[:2] != mask_array.shape[:2]:
        raise ValueError(f"image {image.shape[:2]} and mask {mask_array.shape[:2]} disagree")
    lesion = mask_array

    if condition == "bbox_crop":
        box = bbox_with_margin(lesion, margin_frac)
        if box is None:
            # Empty mask: fall back to the untouched baseline geometry and say so.
            return PreparedInput(
                _baseline_geometry_image(image), _baseline_geometry_mask(lesion), True
            )
        y0, y1, x0, x1 = box
        return PreparedInput(
            _resize_square(image[y0:y1, x0:x1], RESOLUTION),
            _resize_square(lesion[y0:y1, x0:x1], RESOLUTION),
            False,
        )

    if condition == "neutral_bg":
        image = apply_neutral_background(image, lesion)

    return PreparedInput(_baseline_geometry_image(image), _baseline_geometry_mask(lesion), False)
