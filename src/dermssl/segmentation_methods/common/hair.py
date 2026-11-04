"""Hair removal module for v8 segmentation methods.

Pure OpenCV + NumPy implementation of morphological Black-hat and Telea
inpainting (S2, docs/26 3.4). Completely self-contained for cluster and CI runtime.
"""
from __future__ import annotations

import numpy as np

KERNEL = 9  # px at 224: wider than a hair, far narrower than a lesion
THRESHOLD = 10  # black-hat response (grey levels) that counts as hair
INPAINT_RADIUS = 3
MIN_HAIR_LENGTH = 15
MAX_MEAN_WIDTH = 5.0
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
    if points is None:
        return False
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
    """Remove hair from uint8 RGB image of shape (H, W, 3).

    Returns:
        (clean_rgb, hair_fraction)
    """
    import cv2  # noqa: PLC0415

    mask = (elongated_hair_mask if shaped else hair_mask)(rgb, kernel, threshold)
    fraction = float(mask.mean())
    if not mask.any():
        return rgb.copy(), 0.0
    return cv2.inpaint(np.ascontiguousarray(rgb), mask, radius, cv2.INPAINT_TELEA), fraction


def create_directional_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """Create a 1D linear structuring element rotated at angle_degrees."""
    import cv2  # noqa: PLC0415

    size = length if length % 2 != 0 else length + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    # Horizontal line
    kernel[center, :] = 1
    # Rotate line to target orientation
    M = cv2.getRotationMatrix2D((center, center), angle_degrees, 1.0)
    rotated = cv2.warpAffine(kernel, M, (size, size), flags=cv2.INTER_NEAREST)
    return (rotated > 0).astype(np.uint8)


def directional_hair_mask(
    rgb: np.ndarray,
    lengths: list[int] | None = None,
    angles: list[float] | None = None,
    threshold: int = THRESHOLD,
) -> np.ndarray:
    """Multi-scale directional morphological black-hat filter for dense hair rescue.

    Applies linear structuring elements across multiple orientations and lengths to
    isolate thin linear keratin fibers without eroding isotropic or curved lesion boundaries.
    """
    import cv2  # noqa: PLC0415

    if lengths is None:
        lengths = [9, 13, 17]
    if angles is None:
        angles = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    max_response = np.zeros_like(grey)

    for length in lengths:
        for angle in angles:
            k = create_directional_kernel(length, angle)
            # Morphological black-hat highlights dark thin structures matching the linear kernel
            bh = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, k)
            max_response = np.maximum(max_response, bh)

    binary = (max_response > threshold).astype(np.uint8)
    return binary


def remove_hair_directional(
    rgb: np.ndarray,
    lengths: list[int] | None = None,
    threshold: int = THRESHOLD,
    radius: int = INPAINT_RADIUS,
) -> tuple[np.ndarray, float]:
    """Remove hair using multi-scale directional morphological filter and inpainting."""
    import cv2  # noqa: PLC0415

    mask = directional_hair_mask(rgb, lengths=lengths, threshold=threshold)
    fraction = float(mask.mean())
    if not mask.any():
        return rgb.copy(), 0.0
    return cv2.inpaint(np.ascontiguousarray(rgb), mask, radius, cv2.INPAINT_TELEA), fraction

