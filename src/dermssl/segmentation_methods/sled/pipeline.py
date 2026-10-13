"""Modern own SLED adaptation; sklearn IsolationForest, no Java or pyod."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from skimage.segmentation import slic
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from dermssl.w3_components import DegenerateCase, normalize_sled_scores, fuse_sled_scales, sled_location_selection
from .entropy import merge_and_refine


@dataclass(frozen=True)
class SLEDConfig:
    scales: tuple[int, ...] = tuple(range(200, 700, 50))
    compactness: float = 10.
    self_tuning_k: int = 30
    knn_k: int = 50
    centroid_fraction: float = .3
    trees: int = 100
    seed: int = 42
    refinement_passes: int = 30
    selection_variant: str = 'gaussian_corrected'
    centroid_mode: str = 'correct_all'


def affinity(colors, centroids, shape, cfg):
    n = len(colors)
    if n < 3: raise DegenerateCase('At least three superpixels required')
    local_k = min(n // 2, cfg.self_tuning_k); knn = min(cfg.knn_k, local_k)
    if local_k < 1 or knn < 1: raise ValueError('Positive graph neighbor count required')
    distances = cdist(colors, colors)
    distant = cdist(centroids, centroids) > cfg.centroid_fraction * sum(shape)
    distances[distant] = 1e20
    np.fill_diagonal(distances, 0)
    ordered = np.sort(distances, axis=1)
    bandwidth = ordered[:, local_k]
    threshold = ordered[:, knn]
    connected = (distances < threshold[:, None]) | (distances < threshold[None, :])
    weights = np.exp(-np.square(distances) / np.maximum(np.outer(bandwidth, bandwidth), 1e-10))
    weights *= connected & ~distant; np.fill_diagonal(weights, 0)
    return weights


def ght_threshold(values, nu=1e30, tau=.1, kappa=1e-30, omega=.9):
    """Own vectorized evaluation of GHT threshold objective; histogram ties averaged."""
    values = np.asarray(values, float)
    if values.size < 2 or not np.isfinite(values).all() or np.ptp(values) == 0:
        raise DegenerateCase('GHT requires nonconstant finite scores')
    counts, boundaries = np.histogram(values, bins=255)
    locations = (boundaries[:-1]+boundaries[1:])/2
    mass = np.cumsum(counts); moment = np.cumsum(counts*locations); second = np.cumsum(counts*locations**2)
    left = np.maximum(mass[:-1], 1e-30); right = np.maximum(mass[-1]-mass[:-1], 1e-30)
    ldev = np.maximum(second[:-1] - moment[:-1]**2/left, 0.)
    rdev = np.maximum(second[-1]-second[:-1] - (moment[-1]-moment[:-1])**2/right, 0.)
    objective = np.zeros(len(left))
    for count, dev, prior in [(left, ldev, omega), (right, rdev, 1-omega)]:
        prob = count/(left+right)
        variance = np.maximum((prob*nu*tau**2+dev)/(prob*nu+count), 1e-30)
        objective += -dev/variance-count*np.log(variance)+2*(count+kappa*prior)*np.log(count)
    return float(locations[:-1][objective == objective.max()].mean())


def single_scale(rgb, valid, segments, cfg):
    image = rgb.astype(np.float64)/255
    labels = slic(image, n_segments=segments, compactness=cfg.compactness,
                  mask=valid, start_label=0, channel_axis=-1)
    ids = np.unique(labels[valid]); n = len(ids)
    remap = np.full(labels.shape, -1, int)
    colors = np.zeros((n, 3)); centers = np.zeros((n, 2))
    for i, original in enumerate(ids):
        pixels = (labels == original) & valid
        remap[pixels] = i; colors[i] = image[pixels].mean(0); centers[i] = np.argwhere(pixels).mean(0)
    if cfg.centroid_mode == 'upstream_zero': centers[0] = 0
    elif cfg.centroid_mode != 'correct_all': raise ValueError('Unknown centroid mode')
    graph = affinity(colors, centers, valid.shape, cfg)
    partition, audit = merge_and_refine(graph, max_refinement_passes=cfg.refinement_passes)
    groups = np.full(valid.shape, -1, int); groups[valid] = partition[remap[valid]]
    grouped = np.zeros_like(image)
    for group in np.unique(groups[valid]):
        pixels = (groups == group) & valid; grouped[pixels] = image[pixels].mean(0)
    variances, cuts = [], []
    for channel in range(3):
        x = grouped[..., channel][valid]
        if np.ptp(x) == 0: variances.append(0.); cuts.append(float(x[0])); continue
        cut = threshold_otsu(x); low, high = x[x <= cut], x[x > cut]
        variances.append(float(len(low)*len(high)*(low.mean()-high.mean())**2) if len(low) and len(high) else 0.)
        cuts.append(cut)
    channel = int(np.argmax(variances)); variance = variances[channel]
    if variance <= 0: raise DegenerateCase('No between-class contrast at this scale')
    dark = (grouped[..., channel] <= cuts[channel]) & valid
    gray = rgb2gray(image)
    if not dark.any() or not (valid & ~dark).any(): raise DegenerateCase('Empty bisection class')
    background = valid & ~dark if gray[dark].mean() < gray[valid & ~dark].mean() else dark
    train_nodes = np.unique(remap[background])
    if len(train_nodes) < 2: raise DegenerateCase('Not enough healthy-skin superpixels for IForest')
    forest = IsolationForest(n_estimators=cfg.trees, contamination=.1, random_state=cfg.seed,
                             n_jobs=1, max_samples='auto').fit(colors[train_nodes])
    # Invert the contamination-adjusted sklearn decision function: high=anomaly.
    node_scores = -forest.decision_function(colors)
    score = np.zeros(valid.shape); score[valid] = node_scores[remap[valid]]
    return normalize_sled_scores(score), variance, audit


def segment(rgb, valid=None, cfg=SLEDConfig()):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('uint8 RGB image required')
    valid = np.ones(rgb.shape[:2], bool) if valid is None else np.asarray(valid)
    if valid.shape != rgb.shape[:2] or not np.isin(valid,[0,1]).all() or not valid.any():
        raise DegenerateCase('Matching nonempty binary valid mask required')
    valid = valid.astype(bool)
    if not cfg.scales or len(set(cfg.scales)) != len(cfg.scales) or any(n < 3 for n in cfg.scales):
        raise ValueError('Unique superpixel scales >=3 required')
    scores, variances, audits = [], [], []
    for scale in cfg.scales:
        score, variance, audit = single_scale(rgb, valid, scale, cfg)
        scores.append(score); variances.append(variance); audits.append(audit)
    fused = fuse_sled_scales(scores, variances); threshold = ght_threshold(fused[valid])
    mask = sled_location_selection(fused >= threshold, valid, variant=cfg.selection_variant)
    return {'probability': fused, 'mask': mask, 'valid': valid,
            'metadata': {'solver': 'own_python_h2', 'threshold': threshold, 'scales': list(cfg.scales),
                         'centroid_mode': cfg.centroid_mode, 'selection_variant': cfg.selection_variant,
                         'scale_audits': audits}}
