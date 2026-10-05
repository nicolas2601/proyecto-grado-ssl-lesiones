"""Shape and colour descriptors of the lesion mask (E3 F4, docs/33 section 3).

Seven numbers per image, computed on the ORIGINAL-resolution image and mask:

* ``asymmetry``: 1 - overlap of the lesion with its point reflection about its
  centroid along each image axis, averaged over the two axes (0 = symmetric);
* ``border_irregularity``: 1 - solidity (lesion area over the area of its convex
  hull, hull taken over pixel corners), 0 for a convex lesion. A concavity
  measure: unlike a perimeter ratio it does not depend on the staircase bias of
  a pixel perimeter;
* ``lab_std_l/a/b``: standard deviation of CIE Lab inside the lesion (colour variegation);
* ``relative_diameter``: equivalent diameter over the shorter image side;
* ``contrast_de``: CIE76 distance between the mean Lab of the lesion and of the
  skin pixels (the perilesional ring unless a ``skin`` mask is given).

An empty mask gives NaN everywhere; :func:`standardize` fills NaN with the fit
mean (z = 0) and counts it. Pure numpy plus scipy's convex hull.
"""

from __future__ import annotations

import numpy as np

from dermssl import e3_geometry as geo
from dermssl.fitzpatrick import srgb_to_lab

DESCRIPTOR_NAMES = ("asymmetry", "border_irregularity", "lab_std_l", "lab_std_a", "lab_std_b",
                    "relative_diameter", "contrast_de")
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)
STD_FLOOR = 1e-12


def asymmetry(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    keys = ys.astype(np.int64) * (2 * width) + xs  # reflected x may reach 2 * width
    overlaps = []
    for reflect_y, reflect_x in ((True, False), (False, True)):
        ry = np.rint(2 * ys.mean() - ys).astype(np.int64) if reflect_y else ys
        rx = np.rint(2 * xs.mean() - xs).astype(np.int64) if reflect_x else xs
        inside = (ry >= 0) & (ry < height) & (rx >= 0) & (rx < width)
        hits = np.isin(ry[inside] * (2 * width) + rx[inside], keys).sum()
        overlaps.append(hits / len(ys))
    return float(1.0 - np.mean(overlaps))


def border_irregularity(mask: np.ndarray) -> float:
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    ys, xs = np.nonzero(geo.boundary(mask))
    corners = np.concatenate([np.stack([ys + dy, xs + dx], axis=1)
                              for dy in (0, 1) for dx in (0, 1)]).astype(float)
    try:
        hull_area = float(ConvexHull(corners).volume)  # 2-D: volume is the area
    except (QhullError, ValueError):
        return float("nan")
    return float(max(0.0, 1.0 - np.count_nonzero(mask) / hull_area))


def _lab(image: np.ndarray, where: np.ndarray) -> np.ndarray:
    return srgb_to_lab(np.asarray(image, dtype=np.float64)[where] / 255.0)


def contrast_de(image: np.ndarray, mask: np.ndarray, skin: np.ndarray | None = None) -> float:
    skin = geo.ring_mask(mask) if skin is None else np.asarray(skin, dtype=bool) & ~mask
    if not skin.any():
        return float("nan")
    return float(np.linalg.norm(_lab(image, mask).mean(axis=0) - _lab(image, skin).mean(axis=0)))


def descriptors(image: np.ndarray, mask: np.ndarray, skin: np.ndarray | None = None) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(N_DESCRIPTORS, np.nan)
    short_side = min(mask.shape[:2])
    lab_std = _lab(image, mask).std(axis=0)
    return np.array([
        asymmetry(mask), border_irregularity(mask), *lab_std,
        geo.equivalent_diameter(mask) / short_side, contrast_de(image, mask, skin),
    ])


def fit_standardizer(fit: np.ndarray) -> dict:
    """Column mean and std of the fit split, ignoring NaN; a constant column gets std 1."""
    fit = np.asarray(fit, dtype=np.float64)
    std = np.nanstd(fit, axis=0)
    return {"mean": np.nanmean(fit, axis=0).tolist(), "std": np.where(std > STD_FLOOR, std, 1.0).tolist()}


def apply_standardizer(stats: dict, x: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """``(z, n_filled)``: NaN become z = 0, i.e. the fit mean; ``n_filled`` counts them per column."""
    x = np.asarray(x, dtype=np.float64)
    z = (x - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    missing = np.isnan(z)
    return np.where(missing, 0.0, z), [int(n) for n in missing.sum(axis=0)]


def standardize(fit: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    stats = fit_standardizer(fit)
    z_fit, _ = apply_standardizer(stats, fit)
    z_other, filled = apply_standardizer(stats, other)
    return z_fit, z_other, {**stats, "n_filled": filled}
