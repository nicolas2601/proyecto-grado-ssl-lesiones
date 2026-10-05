"""The images each E3 variant feeds to the frozen extractor (docs/33 section 3).

One original-resolution RGB image plus its ``student_r3`` lesion mask give nine
uint8 conditions, each already in the geometry the extractor takes:

``full``          the B71 view: bank resize + centre crop, resolution 224;
``crop_view``     F1: the margin box cut from that 224 view (so a low-resolution
                  crop, resized back to 224), mask carried through the same geometry;
``crop_orig224``  F2: the margin box cut from the ORIGINAL pixels, resized to 224;
``crop_orig448``  the 448 flavour of F2 (same box, resized to 448);
``shift0..3``     F3: the margin box translated along the four diagonals, from the original;
``ring``          F5: only the perilesional ring (outside the mask, minus hair, specular
                  and clipped pixels) kept; everything else ImageNet-mean grey; cropped to
                  the ring's bounding box, resized to 224.

A condition that cannot be built (empty mask, lesion outside the centre crop, no clean
ring pixel) falls back to the ``full`` view of the same size and is listed in the
returned ``fallbacks`` so the caller counts it: a high count means the variant is quietly
degenerating into the baseline. Crops are resized to an exact square (as C1's
``bbox_crop``), which distorts the aspect of the box.

``resize_crop(image_array, resolution)`` is injected: the pod passes the torchvision
``torch_v71.resize_crop_uint8`` (the B71 protocol); tests pass a PIL-only equivalent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dermssl import c1_extraction as ex
from dermssl import e3_descriptors as desc
from dermssl import e3_geometry as geo
from dermssl.mask_conditions import apply_neutral_background

SHIFT_CONDITIONS = tuple(f"shift{k}" for k in range(geo.N_SHIFTS))
CONDITIONS = ("full", "crop_view", "crop_orig224", "crop_orig448", *SHIFT_CONDITIONS, "ring")
BASE, WIDE = 224, 448
SPECULAR_MIN = 240  # every channel at or above: a specular highlight (heuristic, not registered)
CLIPPED_MIN = 255  # any channel at the maximum: saturated (heuristic, not registered)

ResizeCrop = Callable[[np.ndarray, int], np.ndarray]
HairFn = Callable[[np.ndarray], np.ndarray]


def side_of(condition: str) -> int:
    return WIDE if condition == "crop_orig448" else BASE


def artifact_flags(image: np.ndarray, hair_fn: HairFn | None = None) -> np.ndarray:
    """Specular, clipped and (when a detector is given) hair pixels."""
    image = np.asarray(image)
    flags = (image.min(axis=2) >= SPECULAR_MIN) | (image.max(axis=2) >= CLIPPED_MIN)
    if hair_fn is not None:
        flags = flags | np.asarray(hair_fn(image), dtype=bool)
    return flags


def ring_keep(image: np.ndarray, mask: np.ndarray, hair_fn: HairFn | None = None) -> np.ndarray:
    """The perilesional ring minus the flagged pixels."""
    return geo.ring_mask(mask) & ~artifact_flags(image, hair_fn)


def _crop(image: np.ndarray, box: geo.Box, side: int) -> np.ndarray:
    y0, y1, x0, x1 = box
    return ex._resize_square(image[y0:y1, x0:x1], side)


def _view_crop(view: np.ndarray, view_mask: np.ndarray) -> np.ndarray | None:
    box = geo.lesion_box(view_mask)
    return None if box is None else _crop(view, box, BASE)


def _ring_image(image: np.ndarray, mask: np.ndarray, hair_fn: HairFn | None) -> np.ndarray | None:
    keep = ring_keep(image, mask, hair_fn)
    if not keep.any():
        return None
    outer = geo.lesion_box(geo.dilate(mask, geo.ring_width(mask)), 0.0)
    return _crop(apply_neutral_background(image, keep), outer, BASE)


def render(image: np.ndarray, mask: np.ndarray, *, resize_crop: ResizeCrop,
           hair_fn: HairFn | None = None) -> tuple[dict[str, np.ndarray], frozenset[str]]:
    """``(condition -> uint8 HWC, fallbacks)`` for one image; see the module docstring."""
    mask = np.asarray(mask, dtype=bool)
    full = {BASE: resize_crop(image, BASE), WIDE: None}
    box, shifts = geo.lesion_box(mask), geo.shifted_boxes(mask)
    built = {
        "crop_view": _view_crop(full[BASE], ex._baseline_geometry_mask(mask)),
        "crop_orig224": None if box is None else _crop(image, box, BASE),
        "crop_orig448": None if box is None else _crop(image, box, WIDE),
        "ring": _ring_image(image, mask, hair_fn) if mask.any() else None,
        **{name: None if shifts is None else _crop(image, shifts[k], BASE)
           for k, name in enumerate(SHIFT_CONDITIONS)},
    }
    fallbacks = frozenset(name for name, value in built.items() if value is None)
    if "crop_orig448" in fallbacks:
        full[WIDE] = resize_crop(image, WIDE)
    out = {"full": full[BASE]}
    for name in CONDITIONS[1:]:
        out[name] = built[name] if name not in fallbacks else full[side_of(name)]
    return out, fallbacks


def describe(image: np.ndarray, mask: np.ndarray, hair_fn: HairFn | None = None) -> np.ndarray:
    """The F4 descriptors with the artefact-free ring as the skin reference (plain ring if none)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return desc.descriptors(image, mask)
    clean = ring_keep(image, mask, hair_fn)
    return desc.descriptors(image, mask, skin=clean if clean.any() else None)
