"""Independent SegRNorm and SegXNorm diagnostic baselines.

The paper specifies the normalized channels, global Otsu thresholding, Gaussian
smoothing, small-component removal, and hole filling, but omits concrete values
for ROI extraction, adjustment, smoothing, and morphology. Those choices are
explicit in :class:`SegNormConfig` and are never used for pseudo-label training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Literal

import numpy as np

Method = Literal["rnorm", "xnorm"]
ROI = tuple[int, int, int, int] | None
ROIMode = Literal["circular", "full"]
Polarity = Literal["bright", "dark"]
PROVENANCE = "isolated SegRNorm/SegXNorm implementation; paper-unspecified choices are explicit configuration"
PAPER_VARIANT = "diagnostic_baseline_not_exact_reproduction"
POLARITY_RATIONALE = "bright is retained as the implementation default for backward-compatible diagnostics; the paper does not disambiguate lesion polarity"
PAPER_POLARITY_AMBIGUITY = "paper description does not define whether lesions are brighter or darker than the normalized background"
ROI_ASSUMPTION = "circular ROI detects pixels above the black-field threshold, fits a centered circle to their bounding box, and applies the configured inward margin; this is a diagnostic assumption, not an exact paper reproduction"


@dataclass(frozen=True)
class SegNormConfig:
    """Reproducibility parameters for the underspecified paper operations."""

    roi: ROI | ROIMode = None
    roi_black_threshold: float = 0.02
    roi_margin: int = 0
    adjustment: Literal["minmax", "none"] = "minmax"
    sigma: float = 1.0
    kernel_size: int = 5
    min_component_pixels: int = 64
    polarity: Polarity = "bright"


@dataclass(frozen=True)
class SegNormResult:
    mask: np.ndarray
    normalized_channel: np.ndarray
    threshold: float | None
    diagnostics: dict[str, object] = field(default_factory=dict)


XYZ_MATRIX = np.array(
    [[0.49, 0.31, 0.2], [0.17697, 0.8124, 0.01063], [0.0, 0.01, 0.99]],
    dtype=np.float64,
)


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if np.issubdtype(array.dtype, np.integer):
        values = array.astype(np.float64) / np.iinfo(array.dtype).max
    elif np.issubdtype(array.dtype, np.floating):
        values = array.astype(np.float64, copy=False)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("float image values must be in the range [0, 1]")
    else:
        raise ValueError("image dtype must be an integer or floating-point type")
    return np.clip(values, 0.0, 1.0)


def normalize_red(image: np.ndarray) -> np.ndarray:
    """Return paper RNorm: ``R / sqrt(R**2 + G**2 + B**2)``."""
    rgb = _as_rgb_float(image)
    denominator = np.linalg.norm(rgb, axis=-1)
    return np.divide(rgb[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)


def normalize_x(image: np.ndarray) -> np.ndarray:
    """Return paper XNorm after the printed RGB-to-XYZ transform."""
    rgb = _as_rgb_float(image)
    xyz = rgb @ XYZ_MATRIX.T / 0.17697
    denominator = np.linalg.norm(xyz, axis=-1)
    return np.divide(xyz[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)


def _roi_slice(shape: tuple[int, int], roi: ROI) -> tuple[slice, slice]:
    if roi is None:
        return slice(0, shape[0]), slice(0, shape[1])
    top, left, bottom, right = roi
    if not (0 <= top < bottom <= shape[0] and 0 <= left < right <= shape[1]):
        raise ValueError("roi must be (top, left, bottom, right) inside the image")
    return slice(top, bottom), slice(left, right)


def _resolve_roi(
    rgb: np.ndarray,
    roi: ROI | ROIMode,
    black_threshold: float,
    margin: int,
) -> tuple[np.ndarray, tuple[slice, slice], dict[str, object]]:
    if not 0.0 <= black_threshold <= 1.0:
        raise ValueError("roi_black_threshold must be in the range [0, 1]")
    if margin < 0:
        raise ValueError("roi_margin must be non-negative")
    height, width = rgb.shape[:2]
    if roi is None or roi == "full":
        valid = np.ones((height, width), dtype=bool)
        return valid, (slice(0, height), slice(0, width)), {
            "method": "full",
            "black_threshold": float(black_threshold),
            "margin": int(margin),
            "pixel_count": int(valid.sum()),
        }
    if roi == "circular":
        field = np.max(rgb, axis=-1) > black_threshold
        coordinates = np.argwhere(field)
        if coordinates.size == 0:
            raise ValueError("circular ROI could not detect non-black field pixels")
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0) + 1
        center_row = (top + bottom - 1) / 2.0
        center_col = (left + right - 1) / 2.0
        radius = min(bottom - top, right - left) / 2.0 - margin
        if radius <= 0:
            raise ValueError("roi_margin leaves no circular ROI")
        yy, xx = np.indices((height, width))
        circle = (yy - center_row) ** 2 + (xx - center_col) ** 2 <= radius**2
        valid = field & circle
        valid_coordinates = np.argwhere(valid)
        roi_top, roi_left = valid_coordinates.min(axis=0)
        roi_bottom, roi_right = valid_coordinates.max(axis=0) + 1
        return valid, (slice(roi_top, roi_bottom), slice(roi_left, roi_right)), {
            "method": "circular",
            "black_threshold": float(black_threshold),
            "margin": int(margin),
            "center": [float(center_row), float(center_col)],
            "radius": float(radius),
            "bounds": [int(roi_top), int(roi_left), int(roi_bottom), int(roi_right)],
            "pixel_count": int(valid.sum()),
            "assumption": ROI_ASSUMPTION,
        }
    rows, cols = _roi_slice((height, width), roi)
    valid = np.zeros((height, width), dtype=bool)
    valid[rows, cols] = True
    return valid, (rows, cols), {
        "method": "explicit",
        "bounds": [rows.start, cols.start, rows.stop, cols.stop],
        "pixel_count": int(valid.sum()),
    }


def _normalize_values(values: np.ndarray, method: Method) -> np.ndarray:
    if method == "rnorm":
        transformed = values
    else:
        transformed = values @ XYZ_MATRIX.T / 0.17697
    denominator = np.linalg.norm(transformed, axis=-1)
    numerator = transformed[:, 0]
    return np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def _adjust(channel: np.ndarray, mode: str) -> tuple[np.ndarray, bool]:
    if mode == "none":
        return channel, bool(np.ptp(channel) == 0)
    if mode != "minmax":
        raise ValueError("adjustment must be 'minmax' or 'none'")
    low, high = float(channel.min()), float(channel.max())
    if high <= low:
        return np.zeros_like(channel), True
    return (channel - low) / (high - low), False


def _gaussian_blur(
    channel: np.ndarray,
    sigma: float,
    size: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if sigma < 0 or size < 1 or size % 2 == 0:
        raise ValueError("sigma must be non-negative and kernel_size must be positive odd")
    if valid_mask is not None and valid_mask.shape != channel.shape:
        raise ValueError("valid_mask must have the same shape as channel")
    if sigma == 0:
        return channel
    radius = size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    def blur(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, ((0, 0), (radius, radius)), mode="reflect")
        horizontal = sum(
            kernel[index] * padded[:, index : index + channel.shape[1]] for index in range(size)
        )
        padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
        return sum(
            kernel[index] * padded[index : index + channel.shape[0], :] for index in range(size)
        )

    if valid_mask is None:
        return blur(channel)

    weights = valid_mask.astype(np.float64)
    numerator = blur(channel * weights)
    denominator = blur(weights)
    normalized = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return normalized * weights


def _otsu(channel: np.ndarray) -> tuple[float, bool]:
    if np.ptp(channel) == 0:
        return float(channel.flat[0]), True
    histogram, edges = np.histogram(channel, bins=256, range=(0.0, 1.0))
    weights = histogram.astype(np.float64)
    total = weights.sum()
    cumulative = np.cumsum(weights)
    means = np.cumsum(weights * ((edges[:-1] + edges[1:]) / 2))
    valid = (cumulative > 0) & (total - cumulative > 0)
    background = np.divide(means, cumulative, out=np.zeros_like(means), where=cumulative > 0)
    foreground = np.divide(means[-1] - means, total - cumulative, out=np.zeros_like(means), where=total > cumulative)
    variance = cumulative * (total - cumulative) * (background - foreground) ** 2
    variance[~valid] = -1.0
    return float((edges[:-1] + edges[1:])[int(variance.argmax())] / 2), False


def _components(mask: np.ndarray) -> list[np.ndarray]:
    remaining = mask.copy()
    found: list[np.ndarray] = []
    height, width = mask.shape
    while remaining.any():
        seed = tuple(np.argwhere(remaining)[0])
        stack = [seed]
        remaining[seed] = False
        pixels: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            pixels.append((row, col))
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and remaining[neighbor]:
                    remaining[neighbor] = False
                    stack.append(neighbor)
        component = np.zeros_like(mask)
        rows, cols = zip(*pixels)
        component[rows, cols] = True
        found.append(component)
    return found


def _remove_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return mask
    kept = np.zeros_like(mask)
    for component in _components(mask):
        if int(component.sum()) >= minimum:
            kept |= component
    return kept


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    background = ~mask
    reachable = np.zeros_like(mask)
    height, width = mask.shape
    border = np.argwhere(background & ((np.indices(mask.shape)[0] == 0) | (np.indices(mask.shape)[0] == height - 1) | (np.indices(mask.shape)[1] == 0) | (np.indices(mask.shape)[1] == width - 1)))
    stack = [tuple(pixel) for pixel in border]
    reachable[background & ((np.indices(mask.shape)[0] == 0) | (np.indices(mask.shape)[0] == height - 1) | (np.indices(mask.shape)[1] == 0) | (np.indices(mask.shape)[1] == width - 1))] = True
    while stack:
        row, col = stack.pop()
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and background[neighbor] and not reachable[neighbor]:
                reachable[neighbor] = True
                stack.append(neighbor)
    return mask | (background & ~reachable)


def segment(image: np.ndarray, method: Method, config: SegNormConfig | None = None) -> SegNormResult:
    """Segment one RGB image, returning a boolean ``True = lesion`` mask."""
    settings = config or SegNormConfig()
    if method not in ("rnorm", "xnorm"):
        raise ValueError("method must be 'rnorm' or 'xnorm'")
    if settings.polarity not in ("bright", "dark"):
        raise ValueError("polarity must be 'bright' or 'dark'")
    original = np.asarray(image)
    rgb = _as_rgb_float(image)
    roi_mask, (rows, cols), roi_info = _resolve_roi(
        rgb, settings.roi, settings.roi_black_threshold, settings.roi_margin
    )
    channel = np.zeros(rgb.shape[:2], dtype=np.float64)
    channel[roi_mask] = _normalize_values(rgb[roi_mask], method)
    adjusted_values, degenerate = _adjust(channel[roi_mask], settings.adjustment)
    adjusted_channel = np.zeros_like(channel)
    adjusted_channel[roi_mask] = adjusted_values
    local_roi = roi_mask[rows, cols]
    smoothed = _gaussian_blur(
        adjusted_channel[rows, cols],
        settings.sigma,
        settings.kernel_size,
        valid_mask=local_roi,
    )
    threshold, otsu_degenerate = _otsu(smoothed[local_roi])
    mask_roi = np.zeros_like(channel, dtype=bool)
    if not (degenerate or otsu_degenerate):
        local_mask = smoothed > threshold if settings.polarity == "bright" else smoothed < threshold
        mask_roi[rows, cols] = local_mask & local_roi
    mask_roi = _fill_holes(_remove_small_components(mask_roi, settings.min_component_pixels)) & roi_mask
    configuration_roi: object
    if isinstance(settings.roi, tuple):
        configuration_roi = list(settings.roi)
    else:
        configuration_roi = settings.roi or "full"
    configuration = {
        "roi": configuration_roi,
        "roi_black_threshold": float(settings.roi_black_threshold),
        "roi_margin": int(settings.roi_margin),
        "adjustment": settings.adjustment,
        "sigma": float(settings.sigma),
        "kernel_size": int(settings.kernel_size),
        "min_component_pixels": int(settings.min_component_pixels),
        "polarity": settings.polarity,
    }
    return SegNormResult(
        mask=mask_roi,
        normalized_channel=channel,
        threshold=None if degenerate or otsu_degenerate else threshold,
        diagnostics={
            "method": method,
            "threshold": threshold,
            "degenerate_channel": bool(degenerate or otsu_degenerate),
            "lesion_fraction": float(mask_roi.mean()),
            "polarity": settings.polarity,
            "roi": roi_info,
            "histogram_pixel_count": int(roi_mask.sum()),
            "input_dimensions": list(original.shape),
            "input_hash": hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest(),
            "configuration": configuration,
            "provenance": PROVENANCE,
            "paper_variant": PAPER_VARIANT,
            "polarity_rationale": POLARITY_RATIONALE,
            "paper_polarity_ambiguity": PAPER_POLARITY_AMBIGUITY,
        },
    )
