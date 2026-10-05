"""Colour priors for dermoscopy artefacts: surgical ink and specular highlights.

Label-free by construction: the two rules below only look at the RGB values of
one image, and at nothing a human annotated.

* violet surgical ink is the only skin-coloured thing where blue clearly beats
  red, so ``B - R`` above a margin plus enough saturation isolates it;
* a specular highlight (gel bubble, flash on wet skin) is nearly white: every
  channel is high and the spread between them is small.

The result is dilated a couple of pixels, because both artefacts bleed into
their neighbourhood and the saliency map reacts to the halo as well.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# ``B - R`` margin, as a fraction of the 0..255 range.
DEFAULT_INK_BLUE_MARGIN = 0.08
# Saturation ``(max - min) / max``; skin under a dermoscope stays below this.
DEFAULT_INK_MIN_SATURATION = 0.25
# A highlight has every channel above this fraction of the range (very pale
# skin sits near 0.85 and must not be ignored) ...
DEFAULT_HIGHLIGHT_MIN_VALUE = 0.92
# ... and this little spread between them.
DEFAULT_HIGHLIGHT_MAX_SPREAD = 0.06
DEFAULT_DILATION_ITERATIONS = 2
_FULL_RANGE = 255.0


def _channels(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Red channel, per-pixel maximum and per-pixel minimum, as float arrays."""
    rgb = np.asarray(rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {rgb.shape}")
    rgb = rgb.astype(np.float32)
    return rgb, rgb.max(axis=2), rgb.min(axis=2)


def ink_mask(
    rgb_uint8: np.ndarray,
    *,
    blue_margin: float = DEFAULT_INK_BLUE_MARGIN,
    min_saturation: float = DEFAULT_INK_MIN_SATURATION,
) -> np.ndarray:
    """Pixels where blue beats red by ``blue_margin`` and colour is saturated."""
    rgb, top, bottom = _channels(rgb_uint8)
    bluer = rgb[:, :, 2] - rgb[:, :, 0] >= blue_margin * _FULL_RANGE
    saturation = np.divide(top - bottom, top, out=np.zeros_like(top), where=top > 0)
    return bluer & (saturation >= min_saturation)


def highlight_mask(
    rgb_uint8: np.ndarray,
    *,
    min_value: float = DEFAULT_HIGHLIGHT_MIN_VALUE,
    max_spread: float = DEFAULT_HIGHLIGHT_MAX_SPREAD,
) -> np.ndarray:
    """Pixels that are nearly white: every channel high, little spread."""
    _, top, bottom = _channels(rgb_uint8)
    return (bottom >= min_value * _FULL_RANGE) & (top - bottom <= max_spread * _FULL_RANGE)


def artifact_mask(
    rgb_uint8: np.ndarray,
    *,
    blue_margin: float = DEFAULT_INK_BLUE_MARGIN,
    min_saturation: float = DEFAULT_INK_MIN_SATURATION,
    min_value: float = DEFAULT_HIGHLIGHT_MIN_VALUE,
    max_spread: float = DEFAULT_HIGHLIGHT_MAX_SPREAD,
    dilation_iterations: int = DEFAULT_DILATION_ITERATIONS,
) -> np.ndarray:
    """``True`` on ink or highlight pixels, grown by ``dilation_iterations``."""
    found = ink_mask(rgb_uint8, blue_margin=blue_margin, min_saturation=min_saturation)
    found |= highlight_mask(rgb_uint8, min_value=min_value, max_spread=max_spread)
    if dilation_iterations <= 0 or not found.any():
        return found
    return ndimage.binary_dilation(found, iterations=dilation_iterations)
