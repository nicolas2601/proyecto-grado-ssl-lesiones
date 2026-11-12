"""Skin-luminance simulation for /api/simulate-tone (research simulation only).

CIE L* of the surrounding skin is shifted on non-lesion pixels only; a* and b* are untouched.
The shift is feathered: a protected margin around the lesion is dilated by max(3 px, 5% of the lesion's
equivalent diameter), and the weight of the shift is a Gaussian blur (sigma 2 px) of "outside the margin",
so the lesion boundary does not read as a cut. Lesion pixels are restored bit-exact from the source.
The realised shift is measured after the sRGB round-trip, so both L* and gamut clipping count.
Reuses ``dermssl.fitzpatrick.srgb_to_lab``, ``dermssl.circle.lab_to_srgb``, and the e3 geometry dilation.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from dermssl import e3_geometry as geo
from dermssl import fitzpatrick as fz
from dermssl.circle import lab_to_srgb

MAX_ABS_DELTA = 16.0
MAX_SIDE = 768
MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.90
MASK_THRESHOLD = 128
LAB_L_MIN = 0.0
LAB_L_MAX = 100.0
GAMUT_TOLERANCE_L = 0.5
PROTECTION_MIN_PX = 3
PROTECTION_DIAMETER_FRACTION = 0.05
FEATHER_SIGMA_PX = 2.0
_GAUSSIAN_SIGMAS = 3.0
_UINT8_SCALE = 255.0


class ToneSimulationError(ValueError):
    """A simulation input was rejected; ``code`` is the public error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToneResult:
    rgb: np.ndarray
    clipped_fraction: float
    lesion_preserved: bool


def validate_delta(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToneSimulationError("INVALID_DELTA", "delta_l must be a number")
    delta = float(value)
    if not math.isfinite(delta) or abs(delta) > MAX_ABS_DELTA:
        raise ToneSimulationError("INVALID_DELTA", "delta_l must be within [-16, 16]")
    return delta


def fit_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def decode_mask_png(payload: bytes, width: int, height: int) -> np.ndarray:
    """Decode an 8-bit mask PNG (255 = lesion) and nearest-resize it to ``width`` x ``height``."""
    try:
        mask_image = Image.open(io.BytesIO(payload)).convert("L")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ToneSimulationError("INVALID_MASK", "mask_png_base64 is not a decodable PNG") from exc
    if mask_image.size != (width, height):
        mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask_image) >= MASK_THRESHOLD


def check_mask_area(mask: np.ndarray) -> None:
    area = float(np.mean(mask)) if mask.size else 0.0
    if not MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION:
        raise ToneSimulationError("INVALID_MASK", "mask area is outside the plausible 1%-90% range")


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = math.ceil(_GAUSSIAN_SIGMAS * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge replication (no scipy in the image)."""
    kernel = _gaussian_kernel(sigma)
    radius = len(kernel) // 2
    height, width = field.shape
    padded = np.pad(field, radius, mode="edge")
    rows = sum(kernel[i] * padded[:, i : i + width] for i in range(len(kernel)))
    return sum(kernel[i] * rows[i : i + height, :] for i in range(len(kernel)))


def skin_weight(mask: np.ndarray) -> np.ndarray:
    """Per-pixel weight of the luminance shift: 0 inside the lesion, ramping up to 1 beyond the protected margin."""
    diameter = geo.equivalent_diameter(mask)
    margin = max(PROTECTION_MIN_PX, round(PROTECTION_DIAMETER_FRACTION * diameter))
    protected = geo.dilate(mask, margin)
    weight = np.clip(_gaussian_blur((~protected).astype(np.float64), FEATHER_SIGMA_PX), 0.0, 1.0)
    weight[mask] = 0.0
    return weight


def simulate_tone(rgb: np.ndarray, mask: np.ndarray, delta: float) -> ToneResult:
    """Shift L* of the surrounding skin by ``delta`` with a feathered weight. ``rgb`` is uint8 (H, W, 3); ``mask`` is bool (H, W)."""
    if delta == 0.0:
        return ToneResult(rgb=rgb.copy(), clipped_fraction=0.0, lesion_preserved=True)
    lab = fz.srgb_to_lab(rgb.astype(np.float64) / _UINT8_SCALE)
    skin = ~mask
    weight = skin_weight(mask)
    requested = weight * delta
    out_lab = lab.copy()
    out_lab[..., 0] = np.clip(lab[..., 0] + requested, LAB_L_MIN, LAB_L_MAX)
    out = np.rint(lab_to_srgb(out_lab) * _UINT8_SCALE).astype(np.uint8)
    out[mask] = rgb[mask]
    realised = fz.srgb_to_lab(out.astype(np.float64) / _UINT8_SCALE)[..., 0] - lab[..., 0]
    clipped = skin & (np.abs(realised - requested) > GAMUT_TOLERANCE_L)
    skin_count = int(skin.sum())
    return ToneResult(
        rgb=out,
        clipped_fraction=float(clipped.sum()) / skin_count if skin_count else 0.0,
        lesion_preserved=bool(np.array_equal(out[mask], rgb[mask])),
    )


def encode_webp_lossless(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="WEBP", lossless=True)
    return buffer.getvalue()
