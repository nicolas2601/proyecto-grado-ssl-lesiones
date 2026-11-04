"""Dermoscope field of view: which pixels of a cached image are real skin.

A dermoscope projects a bright circular field on a dark frame. Everything
outside that circle is vignette: it is neither skin nor lesion, yet it is the
brightest structural edge in the image, so saliency maps latch onto it and the
pseudo-mask ends up covering the whole field instead of the lesion.

The detector is deliberately geometric: the bright pixels give a bounding box,
a circle is fitted inside that box and shrunk by an inward margin, and the
circle - not the bright set - is the validity mask. Two consequences are
intentional:

* the mask is convex, so a very dark lesion core inside the field can never be
  punched out of it (a hole there would become supervised background);
* a picture whose four corners are all clearly lit (HAM10000 style crops) is
  a full frame and keeps every pixel; a vignette always darkens the corners,
  so one bright corner is not enough to skip the circle fit.

Both decisions are relative to the centre of the image, never to a fixed
brightness: dark skin (Fitzpatrick VI sits near RGB 78/52/44) is darker than a
grey vignette on light skin, so any absolute cut-off trades one tone for
another. The luminance floor survives only as a floor.

The vignette ring is not pure black (grey 0.05-0.20 is common), which is why
the threshold is half of the central brightness rather than the floor.

Nothing here reads masks, labels or diagnoses; the input is the image cache.
"""

from __future__ import annotations

import numpy as np

# Fraction of the 0-255 range on the brightest channel; above it a pixel is field.
DEFAULT_BLACK_THRESHOLD = 0.06
# Inward margin as a fraction of the fitted radius; cuts the vignette ring.
DEFAULT_MARGIN_FRACTION = 0.04
# Below this fraction of valid pixels the fit is degenerate and is discarded.
MIN_VALID_FRACTION = 0.05
# Corner patches (fraction of the side) whose mean brightness decides "no vignette".
CORNER_FRACTION = 0.05
# A corner is lit when it reaches this share of the central luminance...
CORNER_LUMINANCE_RATIO = 0.6
# ...or when it still looks like skin: red above blue (a dermoscope ring is
# achromatic) and not near black. Optical vignetting of the camera darkens the
# corners of a full frame to roughly half of the centre but keeps their colour.
CORNER_SKIN_CHROMA = 0.03
CORNER_SKIN_MIN_LUMINANCE = 0.12
# A dermoscope circle larger than the frame clips one corner; that is still a
# full frame. Two dark corners are a vignette.
MIN_LIT_CORNERS = 3
# Window (fraction of each side) taken as "the centre of the image".
CENTRAL_WINDOW_FRACTION = 0.30
# The field threshold is this share of the central brightness, floored and capped.
ADAPTIVE_THRESHOLD_RATIO = 0.5
MAX_ADAPTIVE_THRESHOLD = 0.6

METHOD_CIRCULAR = "circular"
METHOD_FALLBACK = "fallback_full"
METHOD_FULL_FRAME = "full_frame"


def _central_window(plane: np.ndarray) -> np.ndarray:
    """The central window of a single-channel plane, used as the tone reference."""
    height, width = plane.shape
    half_rows = max(1, int(round(height * CENTRAL_WINDOW_FRACTION / 2)))
    half_columns = max(1, int(round(width * CENTRAL_WINDOW_FRACTION / 2)))
    row, column = height // 2, width // 2
    return plane[
        max(0, row - half_rows) : row + half_rows,
        max(0, column - half_columns) : column + half_columns,
    ]


def _corner_patches(image: np.ndarray) -> tuple[np.ndarray, ...]:
    height, width = image.shape[:2]
    side = max(1, int(round(min(height, width) * CORNER_FRACTION)))
    return (
        image[:side, :side],
        image[:side, width - side :],
        image[height - side :, :side],
        image[height - side :, width - side :],
    )


def _corner_is_lit(patch: np.ndarray, reference: float, ratio: float) -> bool:
    """A corner is lit when it is bright relative to the centre or looks like skin."""
    rgb = patch.reshape(-1, 3).mean(axis=0)
    luminance = float(rgb.mean())
    if luminance >= ratio * reference:
        return True
    chroma = float(rgb[0] - rgb[2]) / 255.0
    return chroma >= CORNER_SKIN_CHROMA and luminance >= CORNER_SKIN_MIN_LUMINANCE * 255.0


def _corners_are_lit(image: np.ndarray, ratio: float = CORNER_LUMINANCE_RATIO) -> bool:
    """True when at least three corners are bright or skin-coloured: no vignette.

    Relative brightness keeps dark skin from being read as a vignette (a
    uniform Fitzpatrick VI frame has dim corners but they match its centre).
    The chroma test keeps camera vignetting from being read as a dermoscope
    ring: the ring is achromatic and near black, the darkened skin is not.
    Known limit: very dark skin whose corners fall below about half of the
    centre is indistinguishable from a brownish ring and keeps the circle.
    """
    reference = float(np.median(_central_window(image.mean(axis=-1))))
    if reference <= 0.0:
        return False
    lit = sum(_corner_is_lit(patch, reference, ratio) for patch in _corner_patches(image))
    return lit >= MIN_LIT_CORNERS


def _adaptive_threshold(max_channel: np.ndarray, black_threshold: float) -> float:
    """Field cut-off as a fraction of 255: half the central brightness, floored.

    Only the outermost bright pixels move the bounding box, so a cut-off above
    a dark lesion core is harmless: the circle is fitted, not carved.
    """
    central = ADAPTIVE_THRESHOLD_RATIO * float(np.median(_central_window(max_channel))) / 255.0
    return min(MAX_ADAPTIVE_THRESHOLD, max(black_threshold, central))


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image.shape}")


def _validate(black_threshold: float, margin_fraction: float) -> None:
    if not 0.0 <= black_threshold <= 1.0:
        raise ValueError(f"black_threshold must be within [0, 1], got {black_threshold}")
    if not 0.0 <= margin_fraction < 1.0:
        raise ValueError(f"margin_fraction must be within [0, 1), got {margin_fraction}")


def _full_frame(
    shape: tuple[int, int],
    field_fraction: float,
    threshold: float,
    method: str = METHOD_FALLBACK,
    **extra,
) -> tuple[np.ndarray, dict]:
    valid = np.ones(shape, dtype=bool)
    return valid, {
        "method": method,
        "field_fraction": field_fraction,
        "valid_fraction": 1.0,
        "threshold": threshold,
        **extra,
    }


def _fit_circle(
    field: np.ndarray, margin_fraction: float
) -> tuple[np.ndarray, float, float, float] | None:
    """Circle inscribed in the bounding box of the field, shrunk by the margin."""
    rows = np.flatnonzero(field.any(axis=1))
    columns = np.flatnonzero(field.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    center_row = (int(rows[0]) + int(rows[-1])) / 2.0
    center_column = (int(columns[0]) + int(columns[-1])) / 2.0
    box = min(int(rows[-1]) - int(rows[0]), int(columns[-1]) - int(columns[0])) + 1
    radius = (box / 2.0) * (1.0 - margin_fraction)
    yy, xx = np.indices(field.shape)
    valid = (yy - center_row) ** 2 + (xx - center_column) ** 2 <= radius**2
    return valid, center_row, center_column, radius


def field_validity_mask(
    rgb_uint8: np.ndarray,
    *,
    black_threshold: float = DEFAULT_BLACK_THRESHOLD,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
) -> tuple[np.ndarray, dict]:
    """Return ``(valid bool (H, W), stats)`` for one ``(H, W, 3)`` uint8 image.

    Never raises on a degenerate image: an all-black frame, or a fit that would
    keep almost nothing, falls back to the full frame with
    ``method="fallback_full"`` so the sample stays usable downstream.
    """
    _validate(black_threshold, margin_fraction)
    image = np.asarray(rgb_uint8)
    _validate_image(image)
    height, width = image.shape[:2]
    max_channel = image.max(axis=-1)
    threshold = _adaptive_threshold(max_channel, black_threshold)
    field = max_channel > threshold * 255.0
    field_fraction = float(field.mean())
    if _corners_are_lit(image):
        return _full_frame((height, width), field_fraction, threshold, METHOD_FULL_FRAME)

    fitted = _fit_circle(field, margin_fraction)
    if fitted is None:
        return _full_frame((height, width), field_fraction, threshold)
    valid, center_row, center_column, radius = fitted
    valid_fraction = float(valid.mean())
    if valid_fraction < MIN_VALID_FRACTION:
        return _full_frame(
            (height, width), field_fraction, threshold, discarded_radius=float(radius)
        )
    return valid, {
        "method": METHOD_CIRCULAR,
        "field_fraction": field_fraction,
        "valid_fraction": valid_fraction,
        "threshold": threshold,
        "center": [center_row, center_column],
        "radius": float(radius),
    }


def compute_field_cache(
    images: np.ndarray,
    out: np.ndarray,
    *,
    black_threshold: float = DEFAULT_BLACK_THRESHOLD,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    chunk: int = 256,
) -> dict:
    """Fill ``out`` ``(N, H, W)`` uint8 with 0/1 validity and report statistics.

    ``images`` is the ``(N, H, W, 3)`` uint8 cache (memmap friendly: it is read
    in chunks and never held whole in memory). ``black_threshold`` is the floor
    of the per-image adaptive threshold, not the threshold itself.
    """
    _validate(black_threshold, margin_fraction)
    total = len(images)
    counts = {METHOD_CIRCULAR: 0, METHOD_FALLBACK: 0, METHOD_FULL_FRAME: 0}
    field_sum = 0.0
    valid_sum = 0.0
    for start in range(0, total, max(1, chunk)):
        block = np.asarray(images[start : start + max(1, chunk)])
        for offset, image in enumerate(block):
            valid, stats = field_validity_mask(
                image, black_threshold=black_threshold, margin_fraction=margin_fraction
            )
            out[start + offset] = valid.astype(np.uint8)
            counts[stats["method"]] += 1
            field_sum += stats["field_fraction"]
            valid_sum += stats["valid_fraction"]
    return {
        "images": total,
        "circular_count": counts[METHOD_CIRCULAR],
        "fallback_count": counts[METHOD_FALLBACK],
        "full_frame_count": counts[METHOD_FULL_FRAME],
        "mean_field_fraction": field_sum / total if total else 0.0,
        "mean_valid_fraction": valid_sum / total if total else 0.0,
        "black_threshold": black_threshold,
        "margin_fraction": margin_fraction,
    }
