"""Label-free W3 components; these alone are not complete SLED/USL-Net replicas."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage


class DegenerateCase(ValueError):
    """An input cannot support a trustworthy pseudo-label; retain it as a failure."""


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape or not np.isin(value, [0, 1]).all():
        raise ValueError('A matching binary 2D mask is required')
    return value.astype(bool)


def gaussian_weights(shape: tuple[int, int], sigma_fraction: float = .15) -> np.ndarray:
    """Full-size normalized Gaussian, centered between pixels for even dimensions."""
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Positive integer height and width required')
    if not np.isfinite(sigma_fraction) or sigma_fraction <= 0:
        raise ValueError('Positive finite sigma fraction required')
    h, w = shape
    yy = np.arange(h, dtype=np.float64) - (h - 1) / 2
    xx = np.arange(w, dtype=np.float64) - (w - 1) / 2
    radius_squared = yy[:, None] ** 2 + xx[None, :] ** 2
    # Subtract maximum log weight so even very small sigma stays normalizable.
    variance = (sigma_fraction * max(h, w)) ** 2
    if not np.isfinite(variance) or variance == 0:
        raise ValueError('Sigma is outside the representable range')
    log_weights = -(radius_squared - radius_squared.min()) / (2 * variance)
    weights = np.exp(log_weights)
    return weights / weights.sum()


def sled_location_selection(candidate: np.ndarray, valid: np.ndarray, *,
                            variant: str = 'gaussian_corrected',
                            constrain_filled_holes: bool = True) -> np.ndarray:
    """SLED location heuristic, with corrected and upstream-area arms explicit.

    A central circle intersected with a rectangle weights entire components; it
    does not crop the winning component. 8-connectivity matches skimage's 2D
    default. Clipping filled holes to valid is a separately recorded safety change.
    """
    candidate = np.asarray(candidate)
    if candidate.ndim != 2 or not all(candidate.shape):
        raise ValueError('Nonempty 2D candidate required')
    shape = candidate.shape
    valid = _binary(valid, shape)
    candidate = _binary(candidate, shape)
    if variant not in {'gaussian_corrected', 'upstream_area'}:
        raise ValueError('Unknown SLED selection variant')
    h, w = shape
    yy, xx = np.indices(shape)
    radius = int(.8 * max(h, w) / 2)
    circle = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius ** 2
    rectangle = ((yy >= int(.1 * h)) & (yy <= int(.9 * h)) &
                 (xx >= int(.1 * w)) & (xx <= int(.9 * w)))
    support = circle & rectangle
    labels, count = ndimage.label(candidate & valid, structure=np.ones((3, 3)))
    weights = gaussian_weights(shape) if variant == 'gaussian_corrected' else np.ones(shape)
    scores = ndimage.sum(weights * support, labels, index=np.arange(1, count + 1))
    if count == 0 or not np.any(scores > 0):
        raise DegenerateCase('No component overlaps SLED location support')
    winner = int(np.argmax(scores)) + 1
    result = ndimage.binary_fill_holes(labels == winner)
    return result & valid if constrain_filled_holes else result


def normalize_sled_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not scores.size or not np.isfinite(scores).all():
        raise DegenerateCase('Invalid SLED anomaly score map')
    span = scores.max() - scores.min()
    if span <= 0 or not np.isfinite(span):
        raise DegenerateCase('Constant SLED anomaly score map')
    return (scores - scores.min()) / span


def fuse_sled_scales(scores: list[np.ndarray], variances: list[float]) -> np.ndarray:
    """Stable form of exp((variance-min)/min), preserving nondegenerate weights."""
    v = np.asarray(variances, dtype=np.float64)
    if v.ndim != 1 or not len(v) or len(scores) != len(v) or not np.isfinite(v).all() or v.min() <= 0:
        raise DegenerateCase('Positive finite between-class variances required')
    maps = np.stack(scores)
    if maps.ndim != 3 or not np.isfinite(maps).all() or np.any((maps < 0) | (maps > 1)):
        raise DegenerateCase('Matching normalized 2D score maps required')
    with np.errstate(over='raise', invalid='raise'):
        try:
            logits = (v - v.max()) / v.min()
        except FloatingPointError as exc:
            raise DegenerateCase('Unrepresentable scale-weight ratio') from exc
    weights = np.exp(logits)
    weights /= weights.sum()
    return np.tensordot(weights, maps, axes=(0, 0))


@dataclass(frozen=True)
class USLTextPolicy:
    low: float = .35
    high: float = .65
    min_area: int = 500
    connectivity: int = 4
    # The paper says "too far" but does not give a numerical cutoff: no invented default.
    max_centroid_distance: float | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.low < self.high <= 1):
            raise ValueError('Ordered saliency thresholds required')
        if type(self.min_area) is not int or self.min_area <= 0 or self.connectivity not in (4, 8):
            raise ValueError('Positive area and connectivity 4 or 8 required')
        if self.max_centroid_distance is not None and (not np.isfinite(self.max_centroid_distance) or self.max_centroid_distance < 0):
            raise ValueError('Nonnegative finite center-distance cutoff required')


def usl_text_pseudolabel(saliency: np.ndarray, policy: USLTextPolicy = USLTextPolicy()) -> tuple[np.ndarray, dict]:
    """Text interpretation: choose minimum SUM of pixel distances, not centroid.

    Input is foreground saliency already oriented to [0,1]. Output 0/0.5/1.
    Boundaries equal to .35/.65 remain uncertain (explicit tie decision).
    The omitted connectivity and distance cutoff are declared policy choices.
    No lesion means abstention, never a silent all-background training target.
    """
    s = np.asarray(saliency, dtype=np.float64)
    if s.ndim != 2 or not all(s.shape) or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Finite 2D foreground saliency in [0,1] required')
    structure = ndimage.generate_binary_structure(2, 1 if policy.connectivity == 4 else 2)
    labels, count = ndimage.label(s > policy.high, structure=structure)
    center = np.asarray(s.shape, dtype=float) / 2  # H/2,W/2 in eq.2, not (H-1)/2.
    regions = []
    for label in range(1, count + 1):
        pixels = np.argwhere(labels == label)
        if len(pixels) < policy.min_area:
            continue
        weight = float(np.linalg.norm(pixels - center, axis=1).sum())
        distance = float(np.linalg.norm(pixels.mean(axis=0) - center))
        regions.append((weight, label, distance))
    if not regions:
        raise DegenerateCase('USL-Net has no eligible foreground region')
    weight, winner, distance = min(regions)
    if policy.max_centroid_distance is not None and distance > policy.max_centroid_distance:
        raise DegenerateCase('Foreground exceeds declared center-distance cutoff')
    out = np.full(s.shape, .5, dtype=np.float32)
    out[s < policy.low] = 0
    out[labels == winner] = 1
    return out, {'eligible_regions': len(regions), 'centrality_sum': weight,
                 'selected_area': int((labels == winner).sum()), 'centroid_distance_pixels': distance,
                 'connectivity': policy.connectivity, 'distance_cutoff_specified': policy.max_centroid_distance is not None}
