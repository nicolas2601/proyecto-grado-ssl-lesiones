"""Perilesional-ring skin tone estimator (docs/33 E5, the alternative to the Otsu ITA), descriptive.

The ring is the band of pixels just OUTSIDE a lesion mask (:func:`dermssl.e3_geometry.ring_mask`, the
same ring as the E3 F5 variant). Inside it, every pixel that cannot be skin is dropped before any
colour is read: clipped or specular highlights, a near-black vignette and, when a detector is
given, hair. The skin L* and b* medians of what is left go through the project's ITA path
(``fitzpatrick._skin_ita``, Groh thresholds), so the bucket is comparable with the Otsu estimator.

The estimator ABSTAINS instead of guessing: no mask, a mask that is not one lesion, too few valid
ring pixels, or an ITA the b* floor cannot support each give a status and no bucket. Like the
whole ITA family it is NOT validated against expert phototypes.

Pure numpy; the hair detector is injected (it needs cv2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dermssl import e3_geometry as geo
from dermssl import fitzpatrick as fz

# Not registered anywhere in docs/33: choices of this module, recorded in every result JSON.
SATURATION_LEVEL = 0.98  # any channel at or above this share of full scale is a clipped pixel
SPECULAR_MIN_L = 90.0  # CIELAB L* at or above this ...
SPECULAR_MAX_CHROMA = 20.0  # ... with chroma (sqrt(a*^2 + b*^2)) at or below this is a specular highlight
BORDER_BAND_FRACTION = 0.10  # no-mask fallback: frame width as a share of the shorter image side
MIN_RING_VALID_PX = 300  # fewer valid ring pixels than this: abstain
MIN_RING_VALID_FRACTION = 0.25  # a ring that is mostly excluded is not a skin sample: abstain

OK = "ok"
NO_MASK = "no_mask"
DEGENERATE_MASK = "degenerate_mask"
RING_TOO_SMALL = "ring_too_small"
STATUSES = (OK, NO_MASK, DEGENERATE_MASK, RING_TOO_SMALL, fz.ITA_INDETERMINATE, "ita_out_of_range")

HairFn = Callable[[np.ndarray], np.ndarray]


class RoiShapeError(ValueError):
    """The lesion mask and the image are not on the same grid."""


@dataclass(frozen=True)
class RoiEstimate:
    """One image: a bucket (1-6) or an explicit reason for none, with what was read."""

    status: str
    bucket: int | None = None
    ita: float | None = None
    L_median: float | None = None
    b_median: float | None = None
    n_px: int = 0
    ring_px: int = 0
    valid_fraction: float | None = None


def _abstain(status: str, ring_px: int = 0, n_px: int = 0, fraction: float | None = None) -> RoiEstimate:
    return RoiEstimate(status=status, ring_px=ring_px, n_px=n_px, valid_fraction=fraction)


def exclusion_mask(img: np.ndarray, lab: np.ndarray, hair: np.ndarray | None = None) -> np.ndarray:
    """True where a pixel cannot be skin: clipped, specular, near-black border, hair."""
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    bad = (img.max(axis=-1) >= SATURATION_LEVEL)
    bad |= (lab[..., 0] >= SPECULAR_MIN_L) & (chroma <= SPECULAR_MAX_CHROMA)
    bad |= img.mean(axis=-1) < fz.BORDER_LUMINANCE_THRESHOLD
    if hair is not None:
        bad |= np.asarray(hair, dtype=bool)
    return bad


def _plausible(mask: np.ndarray) -> bool:
    area = float(mask.mean())
    return fz.MIN_MASK_AREA_FRACTION <= area <= fz.MAX_MASK_AREA_FRACTION


def _check_shapes(img: np.ndarray, mask: np.ndarray) -> None:
    if mask.shape != img.shape[:2]:
        raise RoiShapeError(f"mask {mask.shape} and image {img.shape[:2]} are not on the same grid")


def _valid_ring(img: np.ndarray, lab: np.ndarray, ring: np.ndarray, hair_fn: HairFn | None, rgb_u8) -> np.ndarray:
    hair = hair_fn(rgb_u8) if hair_fn is not None else None
    return ring & ~exclusion_mask(img, lab, hair)


def _read_skin(lab: np.ndarray, valid: np.ndarray, region_px: int) -> RoiEstimate:
    """ITA of the valid pixels of a region, or an explicit abstention (shared by ring and border)."""
    n_px = int(valid.sum())
    fraction = n_px / region_px if region_px else 0.0
    if n_px < MIN_RING_VALID_PX or fraction < MIN_RING_VALID_FRACTION:
        return _abstain(RING_TOO_SMALL, region_px, n_px, fraction)
    L_med, b_med, ita, status = fz._skin_ita(lab[valid])  # noqa: SLF001 - the shared ITA path, not a copy
    common = {"L_median": L_med, "b_median": b_med, "n_px": n_px, "ring_px": region_px, "valid_fraction": fraction}
    if status != "ok":
        return RoiEstimate(status=status, ita=None if np.isnan(ita) else ita, **common)
    return RoiEstimate(status=OK, bucket=fz.fst_bucket(ita), ita=ita, **common)


def ring_estimate(img: np.ndarray, lesion: np.ndarray | None, *, hair_fn: HairFn | None = None) -> RoiEstimate:
    """ITA bucket of the perilesional ring of ``lesion`` in ``img`` (float sRGB in [0, 1], (H, W, 3)).

    ``lesion`` is a bool mask (True = lesion) on the same grid; ``None`` means there is no mask and the
    estimator is not applicable (status ``no_mask``, nothing is invented).
    """
    if lesion is None:
        return _abstain(NO_MASK)
    lesion = np.asarray(lesion, dtype=bool)
    _check_shapes(img, lesion)
    if not lesion.any() or not _plausible(lesion):
        return _abstain(DEGENERATE_MASK)
    lab = fz.srgb_to_lab(img)
    ring = geo.ring_mask(lesion)
    rgb_u8 = np.rint(img * 255.0).astype(np.uint8) if hair_fn is not None else None
    valid = _valid_ring(img, lab, ring, hair_fn, rgb_u8)
    return _read_skin(lab, valid, int(ring.sum()))


def border_estimate(img: np.ndarray, *, hair_fn: HairFn | None = None) -> RoiEstimate:
    """No-mask fallback: the frame of the image (``BORDER_BAND_FRACTION`` of the shorter side) is the skin sample.

    Same exclusions, abstention and ITA path as the ring. It assumes the border is skin, which a lesion
    touching the edge or a vignette breaks: it is an estimator of last resort, reported apart.
    """
    height, width = img.shape[:2]
    band = max(1, round(BORDER_BAND_FRACTION * min(height, width)))
    frame = np.ones((height, width), dtype=bool)
    frame[band:-band, band:-band] = False
    lab = fz.srgb_to_lab(img)
    rgb_u8 = np.rint(img * 255.0).astype(np.uint8) if hair_fn is not None else None
    valid = _valid_ring(img, lab, frame, hair_fn, rgb_u8)
    return _read_skin(lab, valid, int(frame.sum()))
