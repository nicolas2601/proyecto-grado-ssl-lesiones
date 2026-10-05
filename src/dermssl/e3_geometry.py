"""Lesion geometry for the E3 segmentation variants (docs/33 section 3), pure numpy.

Boxes are ``(y0, y1, x0, x1)``, half open, like ``mask_conditions.bbox_with_margin``,
which is reused for the 10 % margin box. Everything here works on a boolean lesion
mask (``mask_conditions.lesion_mask``: 255, the uncertainty band, is NOT lesion).
An empty mask returns ``None`` from the box helpers so the caller decides the
fallback and counts it.
"""

from __future__ import annotations

import numpy as np

from dermssl.mask_conditions import bbox_with_margin

Box = tuple[int, int, int, int]

MARGIN = 0.10  # docs/33 E3 F1: crop margin around the lesion box
SHIFT_FRACTION = 0.15  # F3: shift of each crop, as a fraction of the box side (my choice, not registered)
SHIFT_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # K = 4 diagonal shifts
N_SHIFTS = len(SHIFT_DIRECTIONS)
RING_WIDTH_FRACTION = 0.25  # F5: ring width as a fraction of the equivalent diameter (my choice)
MIN_RING_WIDTH = 3


def lesion_box(mask: np.ndarray, margin: float = MARGIN) -> Box | None:
    return bbox_with_margin(mask, margin)


def _slide_into(start: int, size: int, limit: int) -> int:
    return int(min(max(start, 0), limit - size))


def shifted_boxes(mask: np.ndarray, margin: float = MARGIN,
                  shift: float = SHIFT_FRACTION) -> list[Box] | None:
    """The margin box translated along the four diagonals, same size, kept inside the image."""
    box = lesion_box(mask, margin)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    height, width = mask.shape[:2]
    dy, dx = int(round((y1 - y0) * shift)), int(round((x1 - x0) * shift))
    out = []
    for sy, sx in SHIFT_DIRECTIONS:
        ny = _slide_into(y0 + sy * dy, y1 - y0, height)
        nx = _slide_into(x0 + sx * dx, x1 - x0, width)
        out.append((ny, ny + (y1 - y0), nx, nx + (x1 - x0)))
    return out


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev dilation: ``radius`` rounds of 3x3 growth (a square, not a disc)."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, radius)):
        padded = np.pad(out, 1)
        h, w = out.shape
        out = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                out |= padded[dy:dy + h, dx:dx + w]
    return out


def erode(mask: np.ndarray) -> np.ndarray:
    """One round of 4-neighbour erosion; the image border counts as background."""
    m = np.asarray(mask, dtype=bool)
    padded = np.pad(m, 1)
    return (m & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])


def boundary(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & ~erode(mask)


def equivalent_diameter(mask: np.ndarray) -> float:
    return float(np.sqrt(4.0 * int(np.count_nonzero(mask)) / np.pi))


def ring_width(mask: np.ndarray) -> int:
    return max(MIN_RING_WIDTH, int(round(RING_WIDTH_FRACTION * equivalent_diameter(mask))))


def ring_mask(mask: np.ndarray, width: int | None = None) -> np.ndarray:
    """Pixels within ``width`` of the lesion and outside it (the perilesional context)."""
    mask = np.asarray(mask, dtype=bool)
    return dilate(mask, ring_width(mask) if width is None else width) & ~mask
