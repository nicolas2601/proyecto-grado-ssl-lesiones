"""Hair removal before any saliency or segmentation (S2, docs/26 3.4).

Classical and label-free: a morphological black-hat with a kernel larger
than a hair's width but smaller than a lesion finds thin dark structures,
a threshold turns them into a mask, and TELEA inpainting fills them from the
surrounding skin. Running it in the ``cache`` stage means every later stage
(contrastive, CCAM, pseudo-masks, student, FST) sees the same clean image.
"""

from __future__ import annotations

import numpy as np

KERNEL = 9  # px at 224: wider than a hair, far narrower than a lesion
THRESHOLD = 10  # black-hat response (grey levels) that counts as hair
INPAINT_RADIUS = 3
# S2-prime shape filter, registered in docs/27 before running: a component
# stays only if its minimum-area rectangle is >= 15 px long and its mean
# width (area / length) is <= 5 px. Pigment network and globules are short
# blobs; hairs are thin lines, and crossing hairs stay thin even though
# their joint rectangle is not elongated.
MIN_HAIR_LENGTH = 15
MAX_MEAN_WIDTH = 5.0
# Above this inpainted fraction the "clean" image is mostly paint: counted and
# reported apart, never excluded (registered in docs/27 before running).
HEAVY_INPAINT_FRACTION = 0.30


def hair_mask(rgb: np.ndarray, kernel: int = KERNEL, threshold: int = THRESHOLD) -> np.ndarray:
    import cv2  # noqa: PLC0415

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    response = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, element)
    return (response > threshold).astype(np.uint8)


def _elongated(stats_mask: np.ndarray) -> bool:
    import cv2  # noqa: PLC0415

    points = cv2.findNonZero(stats_mask)
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    long_side = max(w, h, 1.0)
    return long_side >= MIN_HAIR_LENGTH and len(points) / long_side <= MAX_MEAN_WIDTH


def elongated_hair_mask(rgb: np.ndarray, kernel: int = KERNEL, threshold: int = THRESHOLD) -> np.ndarray:
    """The black-hat mask keeping only hair-shaped (long, thin) components."""
    import cv2  # noqa: PLC0415

    raw = hair_mask(rgb, kernel, threshold)
    count, labels = cv2.connectedComponents(raw, connectivity=8)
    keep = np.zeros_like(raw)
    for label in range(1, count):
        component = (labels == label).astype(np.uint8)
        if _elongated(component):
            keep |= component
    return keep


def remove_hair(
    rgb: np.ndarray,
    kernel: int = KERNEL,
    threshold: int = THRESHOLD,
    radius: int = INPAINT_RADIUS,
    shaped: bool = False,
) -> tuple[np.ndarray, float]:
    """``(clean_rgb, hair_fraction)``; a hairless image comes back untouched.

    ``shaped=True`` is S2-prime: only elongated components are inpainted.
    """
    import cv2  # noqa: PLC0415

    mask = (elongated_hair_mask if shaped else hair_mask)(rgb, kernel, threshold)
    fraction = float(mask.mean())
    if not mask.any():
        return rgb.copy(), 0.0
    return cv2.inpaint(np.ascontiguousarray(rgb), mask, radius, cv2.INPAINT_TELEA), fraction
