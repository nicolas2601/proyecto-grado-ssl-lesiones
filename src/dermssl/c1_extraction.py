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

    @property
    def empty_after_geometry(self) -> bool:
        """No lesion survived to the 224 grid.

        Derived rather than stored so it cannot drift from the mask it
        describes: a stored flag set at construction is one refactor away from
        being right about a mask that has since changed.

        What it costs depends on the condition, which is why the gate lives in
        :mod:`dermssl.c1_cell_gates` and not here: ``neutral_bg`` greys the
        whole image, ``mask_pool`` falls back to CLS, and ``none`` does not
        care at all.
        """
        return not bool(self.mask.any())


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
    """torchvision ``CenterCrop`` semantics, including how it rounds.

    The offset is ``int(round(margin / 2.0))``, not ``margin // 2``. They agree
    on an even margin and differ by one row on an odd one, which is exactly the
    HAM10000 case: ``Resize(255)`` on a 600x450 image gives 255 rows, so the
    vertical margin is 31 and floor lands one row above the bank. The features
    still looked plausible -- same mean, same norm -- and the bank gate failed
    with max_abs_diff 9.764e-01 on 64/64 rows.
    """
    height, width = array.shape[:2]
    top = max(0, int(round((height - size) / 2.0)))
    left = max(0, int(round((width - size) / 2.0)))
    return array[top : top + size, left : left + size]


_INTERMEDIATE = int(RESOLUTION * RESIZE_FACTOR)  # 255, not 256


def _baseline_geometry_image(image: np.ndarray) -> np.ndarray:
    """Resize(255, BILINEAR) then CenterCrop(224) -- the bank's geometry."""
    return _center_crop(_resize_shorter_side(image, _INTERMEDIATE), RESOLUTION)


def _baseline_geometry_mask(mask: np.ndarray) -> np.ndarray:
    """Same geometry as the image, but resampled nearest to stay binary."""
    return _center_crop(_resize_shorter_side(mask, _INTERMEDIATE), RESOLUTION)


def mask_area_retained_after_crop(mask: np.ndarray) -> float:
    """Fraction of the lesion that survives the CenterCrop, in [0, 1].

    The bank geometry is not changed -- it is the one the v6 numbers were
    produced with, and Resize+CenterCrop is the standard DINOv2 evaluation
    protocol. It is measured instead, because on HAM10000 (450x600) the crop
    discards about 102 pixels of each side in the original grid, so a lesion
    near an edge can vanish and only show up downstream as a fallback.

    The denominator is the mask after the SAME resize, NOT the mask in the
    original grid. Dividing by the original would measure the downscale, not
    the crop: Resize(255) on a 450x600 image takes area to 0,321 of its
    original value for a lesion that loses nothing, so a threshold written
    against that ratio fires on every image.

    Returns NaN for an empty mask: there was no lesion to keep, and reporting
    1,0 would let "the segmenter found nothing" average in as a perfect score.
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise TypeError(
            f"expected a bool mask, got {array.dtype}; a uint8 mask would count "
            "255 as one pixel of lesion per 255, not as one pixel"
        )
    resized = _resize_shorter_side(array, _INTERMEDIATE)
    before_crop = int(resized.sum())
    if before_crop == 0:
        return float("nan")
    return float(_center_crop(resized, RESOLUTION).sum()) / before_crop


def summarise_crop_retention(masks) -> dict:
    """Per-batch view of what the bank crop costs, for the step-(e) gate.

    Two ways to end up with no lesion, kept apart because they are different
    failures and call for opposite responses:

    * ``n_empty_from_segmenter`` -- the mask was already empty. That is the
      segmenter finding nothing, and the crop had nothing to do with it.
    * ``n_emptied_by_crop`` -- the mask had a lesion and lost all of it to the
      CenterCrop. That is the geometry.

    Averaging the two hides whichever is smaller. ``n_below_half`` is counted
    only over the masks that HAD a lesion, and both its numerator and its
    denominator are reported, because a bare rate over a denominator nobody
    states is how a gate ends up measured against the wrong population.
    """
    masks = list(masks)
    if not masks:
        raise ValueError(
            "no se puede resumir un lote vacio de mascaras: cero imagenes satisface "
            "todos los umbrales de forma vacua y el gate quedaria en no-op"
        )

    retained: list[float] = []
    n_empty_from_segmenter = 0
    n_emptied_by_crop = 0
    n_below_half = 0

    for mask in masks:
        array = np.asarray(mask)
        value = mask_area_retained_after_crop(array)
        retained.append(value)
        if not array.any():
            n_empty_from_segmenter += 1
            continue
        if value == 0.0:
            n_emptied_by_crop += 1
        if value < 0.5:
            n_below_half += 1

    n_images = len(masks)
    n_with_lesion = n_images - n_empty_from_segmenter
    return {
        "n_images": n_images,
        "n_with_lesion": n_with_lesion,
        "n_empty_from_segmenter": n_empty_from_segmenter,
        "n_emptied_by_crop": n_emptied_by_crop,
        "n_below_half": n_below_half,
        "fraction_empty_from_segmenter": n_empty_from_segmenter / n_images,
        "fraction_emptied_by_crop": n_emptied_by_crop / n_images,
        # None rather than 0.0: with no lesion anywhere there is no rate to
        # report, and 0.0 would read as "nothing fell below half".
        "fraction_below_half": (n_below_half / n_with_lesion) if n_with_lesion else None,
        "retained": retained,
    }


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
