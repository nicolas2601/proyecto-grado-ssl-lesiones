"""S-diag: DINOv2 segments the lesion without training anything (docs/27).

The method was registered on 2026-09-28, before this code existed. Per image:

1. the last-layer patch tokens are L2-normalised one by one;
2. PCA over that image's own tokens; the first component is the saliency map;
3. its sign is fixed so the central quarter of the grid (4x4 of 16x16,
   8x8 of 32x32) has the larger mean projection;
4. Otsu threshold -> binary patch map -> nearest-neighbour resize to the
   evaluation grid, the same way the student's masks are compared;
5. the lesion is the connected component that contains the image centre,
   or the one nearest to it when none does.

The k-means secondary replaces steps 2-4 with k = 2 clusters over the same
tokens; the cluster at the centre is the candidate, then step 5 applies.

Nothing here is fitted on more than one image, and no reference mask is read.
Pure numpy / scikit-learn / scikit-image: importable with no torch.
"""

from __future__ import annotations

import numpy as np

METHODS = ("pca", "kmeans")
KMEANS_CLUSTERS = 2
KMEANS_N_INIT = 10
# 8-connectivity: diagonal neighbours belong to the same component.
CONNECTIVITY = 2


def _as_grid_tokens(tokens: np.ndarray, grid: int) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[0] != grid * grid:
        raise ValueError(f"tokens {tokens.shape} do not fill a {grid}x{grid} grid")
    return tokens


def l2_normalise(tokens: np.ndarray) -> np.ndarray:
    """Each token scaled to unit norm; an all-zero token stays zero."""
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    return tokens / np.where(norms > 0, norms, 1.0)


def central_block(grid: int) -> tuple[slice, slice]:
    """The central quarter of the grid: 4x4 of 16x16, 8x8 of 32x32."""
    size = grid // 4
    start = (grid - size) // 2
    return slice(start, start + size), slice(start, start + size)


def pca_first_component_map(tokens: np.ndarray, grid: int) -> np.ndarray:
    """First principal component of one image's tokens, centre made positive.

    The projection is centred, so "the centre has the larger mean" is the same
    as "the centre's mean is positive"; the SVD's arbitrary sign is undone here.
    """
    unit = l2_normalise(_as_grid_tokens(tokens, grid))
    centred = unit - unit.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    projection = (centred @ components[0]).reshape(grid, grid)
    rows, cols = central_block(grid)
    return -projection if projection[rows, cols].mean() < 0 else projection


def otsu_binary(values: np.ndarray) -> np.ndarray:
    """Strictly above the Otsu threshold is lesion; a flat map has no lesion."""
    from skimage.filters import threshold_otsu

    values = np.asarray(values, dtype=np.float64)
    if values.max() == values.min():
        return np.zeros(values.shape, dtype=bool)
    return values > threshold_otsu(values)


def upsample_nearest(binary: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize that samples each output pixel's centre (as PIL)."""
    binary = np.asarray(binary)
    rows = np.floor((np.arange(shape[0]) + 0.5) * binary.shape[0] / shape[0]).astype(int)
    cols = np.floor((np.arange(shape[1]) + 0.5) * binary.shape[1] / shape[1]).astype(int)
    return binary[np.ix_(rows, cols)].astype(bool)


def central_component(binary: np.ndarray) -> np.ndarray:
    """The component containing pixel (H//2, W//2), else the nearest one.

    "Nearest" is the smallest Euclidean distance from that pixel to any pixel
    of the component; ties go to the first pixel in row-major order.
    """
    from skimage.measure import label

    binary = np.asarray(binary, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary)
    labels = label(binary, connectivity=CONNECTIVITY)
    centre = (binary.shape[0] // 2, binary.shape[1] // 2)
    chosen = labels[centre]
    if chosen == 0:
        rows, cols = np.nonzero(binary)
        nearest = np.argmin((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2)
        chosen = labels[rows[nearest], cols[nearest]]
    return labels == chosen


def kmeans_map(tokens: np.ndarray, grid: int, seed: int) -> np.ndarray:
    """k = 2 cluster labels of the L2-normalised tokens, as a grid."""
    from sklearn.cluster import KMeans

    unit = l2_normalise(_as_grid_tokens(tokens, grid))
    model = KMeans(n_clusters=KMEANS_CLUSTERS, n_init=KMEANS_N_INIT, random_state=seed)
    return model.fit_predict(unit).reshape(grid, grid)


def _candidate(method: str, tokens: np.ndarray, grid: int, shape, seed: int) -> np.ndarray:
    if method == "pca":
        return upsample_nearest(otsu_binary(pca_first_component_map(tokens, grid)), shape)
    if method == "kmeans":
        labels = upsample_nearest(kmeans_map(tokens, grid, seed), shape)
        return labels == labels[shape[0] // 2, shape[1] // 2]
    raise ValueError(f"unknown S-diag method {method!r}; expected one of {METHODS}")


def segment(method: str, tokens: np.ndarray, grid: int, shape, seed: int) -> np.ndarray:
    """Boolean lesion mask of ``shape`` for one image's patch tokens."""
    return central_component(_candidate(method, tokens, grid, tuple(shape), seed))
