"""E5 (docs/33): the global ITA estimators, the canonical decode and the per-dataset L* terciles.

Two estimators read the whole frame, both through the project's own ``dermssl.fitzpatrick``
(the code docs/20 and docs/30 validated against experts, where it did NOT beat the majority baseline):

* ``otsu``: ``estimate_fst(img)``, the lesion taken out by the Otsu fallback, skin = the rest;
* ``whole_image``: ``estimate_fst_whole_image(img)``, every pixel counts.

Neither is a measurement of Fitzpatrick type. The bucket is ``fst_bucket`` (the Groh thresholds
of ``fitzpatrick.FST_THRESHOLDS``, a value on a threshold goes to the darker type) and an image
without a trustworthy estimate ABSTAINS (no bucket, a status), it is never forced into one.

The L* terciles are cut on the skin median L* of ONE dataset, with the convention of the pool
manifest (``fst_pseudo_manifest.assign_tone_terciles``): dark = L* at or below the 1/3 quantile,
light = above the 2/3 quantile. They are luminance, not skin tone, and relative to that dataset.

Canonical decode: JPEG DCT scaling (PIL ``draft``) and then an area-aware bilinear resize bring
any image whose shorter side exceeds ``CANONICAL_SHORT_SIDE`` down to exactly that side, aspect
kept; smaller images are untouched, so HAM10000 (450 x 600) is never resampled and its masks stay
aligned. Large ISIC 2020 files are never decoded at full size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dermssl import fitzpatrick as fz
from dermssl.luminance_dice import TERCILES

ROMAN = ("I", "II", "III", "IV", "V", "VI")
BUCKET_LABELS = ROMAN
ABSTAIN = "abstain"
UNAVAILABLE = "unavailable"
CANONICAL_SHORT_SIDE = 450  # the shorter side of HAM10000: its images are never resampled
MIN_IMAGES_FOR_TERCILES = 3  # same floor as the pool manifest
ESTIMATORS = ("otsu", "whole_image")


class ImageDecodeError(RuntimeError):
    """A file could not be decoded to RGB."""


def bucket_label(bucket: int | None) -> str:
    """Roman numeral of a bucket 1-6, or ``abstain``."""
    return ABSTAIN if bucket is None else ROMAN[bucket - 1]


def _target_size(width: int, height: int) -> tuple[int, int]:
    short = min(width, height)
    if short <= CANONICAL_SHORT_SIDE:
        return width, height
    scale = CANONICAL_SHORT_SIDE / short
    return max(1, round(width * scale)), max(1, round(height * scale))


def decode_canonical(path: Path) -> np.ndarray:
    """RGB float32 in [0, 1], shape (H, W, 3), at the canonical resolution (see the module docstring)."""
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(path) as image:
            target = _target_size(*image.size)
            image.draft("RGB", target)
            rgb = image.convert("RGB")
            if rgb.size != target:
                rgb = rgb.resize(target, Image.BILINEAR)
            return np.asarray(rgb, dtype=np.float32) / 255.0
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageDecodeError(f"cannot decode {path}: {exc}") from exc


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _summary(est: fz.FSTEstimate) -> dict:
    """The JSON-safe fields of one estimate; NaN becomes None."""
    return {"status": est.status, "bucket": None if est.bucket is None else bucket_label(est.bucket),
            "ita": _finite(est.ita), "L_median": _finite(est.L_median), "b_median": _finite(est.b_median),
            "n_skin_px": int(est.n_skin_px)}


def global_estimates(img: np.ndarray) -> dict:
    """Both global estimators on one float sRGB image: ``{"otsu": ..., "whole_image": ...}``."""
    return {"otsu": _summary(fz.estimate_fst(img)), "whole_image": _summary(fz.estimate_fst_whole_image(img))}


def assign_terciles(l_medians: list[float | None]) -> tuple[list[str], dict]:
    """Label every image with its tercile of skin L* inside this list; ``None`` stays unavailable."""
    eligible = [v for v in l_medians if v is not None]
    if len(eligible) < MIN_IMAGES_FOR_TERCILES:
        return [UNAVAILABLE] * len(l_medians), {"low": None, "high": None, "n": len(eligible)}
    low, high = (float(c) for c in np.quantile(np.asarray(eligible, dtype=np.float64), (1 / 3, 2 / 3)))
    dark, medium, light = TERCILES[0], TERCILES[1], TERCILES[2]
    labels = [UNAVAILABLE if v is None else dark if v <= low else light if v > high else medium
              for v in l_medians]
    return labels, {"low": low, "high": high, "n": len(eligible)}
