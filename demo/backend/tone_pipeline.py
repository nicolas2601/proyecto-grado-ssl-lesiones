"""Input handling and output assembly for /api/simulate-tone (research simulation only)."""

from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np
from PIL import Image

from .photometry import measure_photometry
from .tone_sim import (
    MASK_THRESHOLD,
    ToneSimulationError,
    check_mask_area,
    decode_mask_png,
    encode_webp_lossless,
    fit_size,
    simulate_tone,
    validate_delta,
)

SCHEMA = "dermssl.simulate-tone.v1"
SCOPE = "research-simulation"
CLIPPED_DECIMALS = 4
_MASK_ON = 255


def parse_delta(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ToneSimulationError("INVALID_DELTA", "delta_l must be a number") from exc
    return validate_delta(value)


def decode_mask_field(mask_b64: str, width: int, height: int) -> np.ndarray:
    try:
        payload = base64.b64decode(mask_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ToneSimulationError("INVALID_MASK", "mask_png_base64 is not valid base64") from exc
    mask = decode_mask_png(payload, width=width, height=height)
    check_mask_area(mask)
    return mask


def _resize_pair(image: Image.Image, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width, height = fit_size(image.width, image.height)
    if (width, height) == image.size:
        return np.asarray(image), mask
    resized_image = np.asarray(image.resize((width, height), Image.Resampling.BICUBIC))
    mask_image = Image.fromarray(mask.astype(np.uint8) * _MASK_ON, mode="L")
    resized_mask = np.asarray(mask_image.resize((width, height), Image.Resampling.NEAREST)) >= MASK_THRESHOLD
    return resized_image, resized_mask


def simulate_for_image(image: Image.Image, mask_b64: str, delta_text: str) -> dict[str, Any]:
    """Validate the form fields, run the tone transform and assemble the contract response."""
    delta = parse_delta(delta_text)
    mask = decode_mask_field(mask_b64, image.width, image.height)
    rgb, out_mask = _resize_pair(image, mask)
    result = simulate_tone(rgb, out_mask, delta)
    height, width = result.rgb.shape[:2]
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "delta_l": delta,
        "image_webp_base64": base64.b64encode(encode_webp_lossless(result.rgb)).decode("ascii"),
        "width": width,
        "height": height,
        "lesion_pixels_preserved": result.lesion_preserved,
        "clipped_fraction": round(result.clipped_fraction, CLIPPED_DECIMALS),
        "photometry": measure_photometry(result.rgb, out_mask),
    }
