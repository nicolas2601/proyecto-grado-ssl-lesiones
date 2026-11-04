"""Fitzpatrick-anchored tone transform: skin estimation and ITA displacement.

``tone_augment_lab`` samples ``ita_delta`` uniformly in ``[-25, +25]``,
untethered to the image's own phototype. Corbin & Marques (2023, IEEE Access
11:78339) eq. 7 instead derives it, ``ita_delta = ITA - ITA'``, where ``ITA`` is
measured on skin pixels and ``ITA'`` is drawn inside the target FST's range.
Anchoring to the scale is the whole premise of the arm, so this module derives.

``tone_augment_lab`` is NOT modified: its outputs back already-reported
evidence. This is a sibling, the same way ``tone_augment_pairs`` was a sibling
of ``tone_transforms``.

Two skin detectors are provided because the papers disagree on how to find
skin, and which one to use is an empirical question this module exists to
settle:

* ``skin_mask_ycbcr`` -- Corbin & Marques eqs. 2-6, a YCbCr colour rule.
* ``skin_mask_dominant`` -- Bencevic et al. (2024, CMPB 245:108044) sec. 2.1.2:
  Otsu on the HSV value channel, morphological expansion, then k-means over the
  skin pixels' (L*, a*, b*) keeping the most populated cluster.

Both are positive detectors. ``fitzpatrick.estimate_fst`` instead defines skin
by negation (not lesion, not border), which retains hair, specular highlights
and gel bubbles -- all low or negative b* -- and that is the suspected reason
its ITA is indeterminate on ~47% of the pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dermssl.fitzpatrick import FST_THRESHOLDS, ita_degrees, srgb_to_lab
from dermssl.tone_augment_lab import B_PER_ITA_DEGREE, L_PER_ITA_DEGREE

MAX_FST_SHIFT = 2
"""Corbin & Marques sec. III-A2 cap the shift at two FST categories, because
"results beyond this maximum value did not yield satisfactory visual outcomes".
"""

MIN_SKIN_PIXELS = 500
MIN_B_FOR_ITA = 5.0


@dataclass(frozen=True)
class SkinEstimate:
    """Dominant skin colour of one image and how confidently it was found."""

    lab: tuple[float, float, float] | None
    ita: float
    fst: int | None
    skin_fraction: float
    status: str
    method: str


def fst_range(fst: int) -> tuple[float, float]:
    """ITA bounds of one FST bucket under the Groh empirical cut points.

    ``FST_THRESHOLDS`` is descending: (40, 23, 12, 0, -25). Bucket 1 is above
    the first threshold, bucket 6 below the last.
    """
    if not 1 <= fst <= len(FST_THRESHOLDS) + 1:
        raise ValueError(f"fst must be within 1..{len(FST_THRESHOLDS) + 1}, got {fst}")
    high = float("inf") if fst == 1 else FST_THRESHOLDS[fst - 2]
    low = float("-inf") if fst == len(FST_THRESHOLDS) + 1 else FST_THRESHOLDS[fst - 1]
    return low, high


def skin_mask_ycbcr(img: np.ndarray) -> np.ndarray:
    """Corbin & Marques eqs. 2-6: skin pixels by a YCbCr colour rule.

    DISCREPANCY WITH THE SOURCE, deliberate. Corbin & Marques print eqs. 5 and
    6 with ``>=``. Taken literally they reject real skin: a light skin patch
    gives Cr 149.2 against a bound of 176.6, so eq. 5 fails and the rule admits
    nothing. The canonical Chai & Ngan rule those equations come from uses
    ``<=`` for both, which accepts the same patch. ``<=`` is used here; the
    printed form is believed to be a typesetting error.
    """
    rgb = np.asarray(img, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"img must be (H, W, 3), got {rgb.shape}")
    r, g, b = (rgb[..., i] * 255.0 for i in range(3))
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return (
        (cr > 135.0)
        & (cr >= 0.3448 * cb + 76.2069)
        & (cr >= -4.5652 * cb + 234.5652)
        & (cr <= -1.15 * cb + 301.75)
        & (cr <= -2.2857 * cb + 432.85)
    )


def _otsu(values: np.ndarray) -> float:
    counts, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_fg = np.cumsum(counts)
    weight_bg = weight_fg[-1] - weight_fg
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_fg = np.cumsum(counts * centers) / weight_fg
        total = np.sum(counts * centers)
        mean_bg = (total - np.cumsum(counts * centers)) / weight_bg
        variance = weight_fg * weight_bg * (mean_fg - mean_bg) ** 2
    variance[~np.isfinite(variance)] = -1.0
    # Upper edge, not centre: the between-class variance peaks on the last bin
    # of the dark class, whose centre sits BELOW the darkest value it must
    # capture, so a `<= centre` test would exclude that class entirely.
    return float(edges[int(np.argmax(variance)) + 1])


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a square structuring element, numpy only."""
    out = mask.copy()
    for shift in range(1, radius + 1):
        out[shift:, :] |= mask[:-shift, :]
        out[:-shift, :] |= mask[shift:, :]
        out[:, shift:] |= mask[:, :-shift]
        out[:, :-shift] |= mask[:, shift:]
    return out


def _kmeans_dominant(pixels: np.ndarray, k: int, *, seed: int, iters: int = 25) -> np.ndarray:
    """Centroid of the most populated cluster of ``pixels`` shaped (n, 3)."""
    rng = np.random.default_rng(seed)
    centers = pixels[rng.choice(len(pixels), size=k, replace=False)]
    labels = np.zeros(len(pixels), dtype=np.int64)
    for _ in range(iters):
        distances = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            member = pixels[labels == index]
            if len(member):
                centers[index] = member.mean(axis=0)
    counts = np.bincount(labels, minlength=k)
    return centers[int(np.argmax(counts))]


def _clahe_value(img: np.ndarray, *, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE-equalised lightness, used to FIND the mask and never to measure it.

    INTERPRETATION, flagged as such. Bencevic et al. equalise L* and run
    Dullrazor before the Otsu/HSV step, then "convert the image back to the
    CIELAB colorspace" for the k-means. The paper does not say whether the
    k-means sees equalised or original pixels. Equalised lightness would make
    the ITA meaningless, so here the original pixels are measured and CLAHE
    only helps locate skin. ``clip_limit`` and ``tile`` are not given in the
    paper; these are OpenCV's common defaults, not values the source endorses.
    """
    import cv2

    lab = srgb_to_lab(np.asarray(img, dtype=np.float64))
    lightness = np.clip(lab[..., 0] * 255.0 / 100.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(lightness).astype(np.float64) / 255.0


def skin_mask_dominant(
    img: np.ndarray,
    *,
    dilate_radius: int = 3,
    use_clahe: bool = True,
    exclude_hair: bool = True,
) -> np.ndarray:
    """Bencevic et al. sec. 2.1.2: CLAHE, Dullrazor, Otsu on value, expansion.

    Skin is the bright side of the Otsu threshold; the lesion, hair and
    vignette are dark. The expansion grows the *rejected* region so that lesion
    borders and hair haloes do not leak into the skin sample.

    ``exclude_hair`` reuses ``segmentation_methods.common.hair``, whose
    directional black-hat with elongation screening is the same construction
    Dullrazor specifies; only the hair MASK is needed here, not the inpainting,
    because these pixels are dropped rather than repaired.

    Setting both flags False reproduces the reduced variant measured on
    2026-11-04, which is kept callable so the two can be compared.
    """
    rgb = np.asarray(img, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"img must be (H, W, 3), got {rgb.shape}")

    value = _clahe_value(rgb) if use_clahe else rgb.max(axis=-1)
    rejected = value <= _otsu(value)

    if exclude_hair:
        from dermssl.segmentation_methods.common.hair import elongated_hair_mask

        rejected |= elongated_hair_mask(
            np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        ).astype(bool)

    return ~_dilate(rejected, dilate_radius)


def estimate_skin(
    img: np.ndarray,
    *,
    method: str = "dominant",
    lesion_mask: np.ndarray | None = None,
    border_mask: np.ndarray | None = None,
    k: int = 3,
    seed: int = 0,
    use_clahe: bool = True,
    exclude_hair: bool = True,
) -> SkinEstimate:
    """Dominant skin colour, ITA and FST of one sRGB image, float (H, W, 3) in [0, 1].

    ``method`` is ``"ycbcr"`` (Corbin) or ``"dominant"`` (Bencevic). The lesion
    and border masks are intersected out of whichever detector runs: a positive
    skin rule still benefits from dropping a lesion that happens to be skin
    coloured. ``use_clahe`` and ``exclude_hair`` only affect ``"dominant"``.
    """
    if method == "ycbcr":
        skin = skin_mask_ycbcr(img)
    elif method == "dominant":
        skin = skin_mask_dominant(img, use_clahe=use_clahe, exclude_hair=exclude_hair)
    else:
        raise ValueError(f"method must be 'ycbcr' or 'dominant', got {method!r}")

    if lesion_mask is not None:
        skin &= ~np.asarray(lesion_mask, dtype=bool)
    if border_mask is not None:
        skin &= ~np.asarray(border_mask, dtype=bool)

    fraction = float(skin.mean())
    if int(skin.sum()) < MIN_SKIN_PIXELS:
        return SkinEstimate(None, float("nan"), None, fraction, "too_few_skin_pixels", method)

    lab = srgb_to_lab(np.asarray(img, dtype=np.float64))
    pixels = lab[skin]
    if method == "dominant":
        centroid = _kmeans_dominant(pixels, min(k, len(pixels)), seed=seed)
    else:
        centroid = np.median(pixels, axis=0)

    L, a, b = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    if not b >= MIN_B_FOR_ITA:
        return SkinEstimate((L, a, b), float("nan"), None, fraction, "ita_indeterminate", method)

    ita = float(ita_degrees(np.asarray(L), np.asarray(b)))
    if not -90.0 <= ita <= 90.0:
        return SkinEstimate((L, a, b), ita, None, fraction, "ita_out_of_range", method)
    return SkinEstimate((L, a, b), ita, fst_from_ita(ita), fraction, "ok", method)


def fst_from_ita(ita: float) -> int:
    for index, threshold in enumerate(FST_THRESHOLDS, start=1):
        if ita > threshold:
            return index
    return len(FST_THRESHOLDS) + 1


def derive_ita_delta(
    ita: float,
    target_fst: int,
    *,
    rng: np.random.Generator,
    max_abs_bound: float = 60.0,
) -> float:
    """Corbin & Marques eq. 7: ``ita_delta = ITA - ITA'``.

    ``ITA'`` is drawn uniformly inside ``target_fst``'s range. The outermost
    buckets are unbounded on one side, so they are clipped to the adjacent
    threshold plus ``max_abs_bound``: an unbounded draw would be undefined, and
    inventing a wider range than the scale supports is the kind of silent
    assumption this module exists to remove.
    """
    low, high = fst_range(target_fst)
    if not np.isfinite(low):
        low = high - max_abs_bound
    if not np.isfinite(high):
        high = low + max_abs_bound
    return float(ita - rng.uniform(low, high))


def sample_target_fst(
    current_fst: int,
    *,
    rng: np.random.Generator,
    max_shift: int = MAX_FST_SHIFT,
) -> int:
    """A different FST within ``max_shift`` categories of ``current_fst``."""
    n_buckets = len(FST_THRESHOLDS) + 1
    candidates = [
        bucket
        for bucket in range(1, n_buckets + 1)
        if bucket != current_fst and abs(bucket - current_fst) <= max_shift
    ]
    if not candidates:
        return current_fst
    return int(rng.choice(candidates))


def lab_shift(ita_delta: float) -> tuple[float, float]:
    """Corbin & Marques eqs. 8-9, reusing the frozen module's coefficients."""
    return (-ita_delta * L_PER_ITA_DEGREE, ita_delta * B_PER_ITA_DEGREE)
