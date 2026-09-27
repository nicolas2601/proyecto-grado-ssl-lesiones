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


def hair_mask(rgb: np.ndarray, kernel: int = KERNEL, threshold: int = THRESHOLD) -> np.ndarray:
    import cv2  # noqa: PLC0415

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    response = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, element)
    return (response > threshold).astype(np.uint8)


def remove_hair(
    rgb: np.ndarray,
    kernel: int = KERNEL,
    threshold: int = THRESHOLD,
    radius: int = INPAINT_RADIUS,
) -> tuple[np.ndarray, float]:
    """``(clean_rgb, hair_fraction)``; a hairless image comes back untouched."""
    import cv2  # noqa: PLC0415

    mask = hair_mask(rgb, kernel, threshold)
    fraction = float(mask.mean())
    if not mask.any():
        return rgb.copy(), 0.0
    return cv2.inpaint(np.ascontiguousarray(rgb), mask, radius, cv2.INPAINT_TELEA), fraction
