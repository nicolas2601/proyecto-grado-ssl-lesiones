"""Uncertainty Module (UM): saliency map -> pseudo-mask with an ignore band.

Pure numpy/scipy, in the style of USL-Net (Li et al. 2024, Section 3.4):

* saliency below ``low`` (0.35) is background;
* saliency above ``high`` (0.65) is a lesion candidate;
* the band in between is uncertain and ignored by the loss;
* candidates are restricted to the dermoscope field of view, so the vignette
  can never be voted lesion;
* among candidates, connectivity filtering drops small regions and a centrality
  filter keeps the single region closest to the image center; every other
  candidate region becomes uncertain (ignored), not background;
* a survivor covering more than ``max_area_fraction`` of the field is the field
  itself, not a lesion, and the sample is excluded;
* if nothing survives, if the survivor is too far from the center, or if the
  field is empty, the whole sample is excluded (all pixels ignored);
* pixels outside the field are background: they are neither skin nor lesion,
  and leaving them uncertain would waste most of the loss on the vignette;
* artefact pixels (surgical ink, specular highlights) are removed from the
  candidates and end up ignored, never background: the saliency loves them and
  the student must not learn them as skin either.

Output labels follow ``cache_io``: ``BACKGROUND``, ``FOREGROUND``, ``IGNORE``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from artifacts import artifact_mask
from cache_io import BACKGROUND, FOREGROUND, IGNORE
from saliency_prep import as_field, normalize_saliency, orient_saliency

DEFAULT_LOW = 0.35
DEFAULT_HIGH = 0.65
# USL-Net uses alpha = 500 pixels at 224x224 (about 1 percent of the image).
DEFAULT_MIN_AREA_FRACTION = 500 / (224 * 224)
# Fraction of the half-diagonal; farther regions are treated as border noise.
DEFAULT_MAX_CENTER_DISTANCE = 0.5
# A region covering more than this fraction of the field is the field itself.
DEFAULT_MAX_AREA_FRACTION = 0.6
_EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=int)

REASON_VALID = ""
REASON_NO_REGION = "no_region"
REASON_NO_FIELD = "no_field"
REASON_AREA = "area"


@dataclass(frozen=True)
class PseudoMaskResult:
    mask: np.ndarray
    valid: bool
    reason: str = REASON_VALID
    n_regions: int = 0


def _excluded(shape: tuple[int, int], reason: str) -> PseudoMaskResult:
    """An excluded sample: every pixel ignored, with the reason attached."""
    return PseudoMaskResult(
        mask=np.full(shape, IGNORE, dtype=np.uint8), valid=False, reason=reason
    )


REGION_MODES = ("central", "union")


def select_central_region(
    candidates: np.ndarray,
    min_area: int,
    max_center_distance: float,
    mode: str = "central",
) -> np.ndarray | None:
    """Keep the connected region whose centroid is closest to the image center.

    Regions smaller than ``min_area`` pixels are discarded first. The centroid
    distance is normalized by the half-diagonal; a winner farther than
    ``max_center_distance`` means the sample has no trustworthy lesion.

    ``mode="union"`` (S1, docs/26 §2.4) keeps EVERY region that passes the
    same two filters, so a lesion split into several blobs by hair or
    highlights stays whole, while border shadows are still rejected. It
    returns ``None`` in exactly the same cases as ``"central"``.
    """
    regions = accepted_regions(candidates, min_area, max_center_distance, mode)
    if not regions:
        return None
    return np.logical_or.reduce(regions)


def accepted_regions(
    candidates: np.ndarray, min_area: int, max_center_distance: float, mode: str = "central"
) -> list[np.ndarray]:
    """The region masks that make up the foreground under ``mode``."""
    if mode not in REGION_MODES:
        raise ValueError(f"region mode {mode!r} unknown; expected one of {REGION_MODES}")
    labels, count = ndimage.label(candidates, structure=_EIGHT_CONNECTIVITY)
    if count == 0:
        return []
    height, width = candidates.shape
    center = np.array([(height - 1) / 2.0, (width - 1) / 2.0])
    half_diagonal = math.hypot(height, width) / 2.0

    areas = np.bincount(labels.ravel())[1:]
    centroids = ndimage.center_of_mass(candidates, labels, index=range(1, count + 1))
    scored = []
    for label_index, (area, centroid) in enumerate(zip(areas, centroids, strict=True), start=1):
        if area < min_area:
            continue
        distance = float(np.linalg.norm(np.asarray(centroid) - center)) / half_diagonal
        scored.append(((distance, -area), label_index))
    passing = sorted(item for item in scored if item[0][0] <= max_center_distance)
    if not passing:
        return []
    if mode == "central":
        return [labels == passing[0][1]]
    return [labels == label_index for _, label_index in passing]


def saliency_to_pseudo_mask(
    saliency: np.ndarray,
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    max_center_distance: float = DEFAULT_MAX_CENTER_DISTANCE,
    max_area_fraction: float = DEFAULT_MAX_AREA_FRACTION,
    field: np.ndarray | None = None,
    artifact: np.ndarray | None = None,
    normalize: bool = True,
    region_mode: str = "central",
) -> PseudoMaskResult:
    """Convert one saliency map into a three-valued pseudo-mask.

    ``field`` is the dermoscope field of view (bool, True = usable pixel); both
    area limits are fractions of its area, or of the frame when it is absent.
    ``artifact`` marks ink and highlight pixels: they cannot be voted lesion and
    they end up ignored, so the student never learns them as skin either.
    """
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"thresholds must satisfy 0 <= low < high <= 1, got {low}, {high}")
    saliency = normalize_saliency(saliency) if normalize else np.asarray(saliency, np.float32)
    field = as_field(field, saliency.shape)
    artifact = as_field(artifact, saliency.shape, name="artifact")
    reference = float(saliency.size) if field is None else float(field.sum())
    if reference <= 0.0:
        return _excluded(saliency.shape, REASON_NO_FIELD)

    candidates = saliency > high
    if field is not None:
        candidates = candidates & field
    if artifact is not None:
        candidates = candidates & ~artifact
    min_area = max(1, math.ceil(min_area_fraction * reference))
    regions = accepted_regions(candidates, min_area, max_center_distance, region_mode)
    if not regions:
        return _excluded(saliency.shape, REASON_NO_REGION)
    foreground = np.logical_or.reduce(regions)
    if float(foreground.sum()) / reference > max_area_fraction:
        return _excluded(saliency.shape, REASON_AREA)
    return PseudoMaskResult(
        mask=_paint(saliency, low, foreground, field, artifact), valid=True, n_regions=len(regions)
    )


def _paint(
    saliency: np.ndarray,
    low: float,
    foreground: np.ndarray,
    field: np.ndarray | None,
    artifact: np.ndarray | None,
) -> np.ndarray:
    """Background, then the winning region, then the artefacts on top."""
    mask = np.full(saliency.shape, IGNORE, dtype=np.uint8)
    mask[saliency < low] = BACKGROUND
    if field is not None:
        mask[~field] = BACKGROUND
    mask[foreground] = FOREGROUND
    if artifact is not None:
        mask[artifact] = IGNORE
    return mask


def _convert_one(
    raw: np.ndarray,
    field: np.ndarray | None,
    artifact: np.ndarray | None,
    *,
    orient: bool,
    normalize: bool,
    mask_kwargs: dict,
) -> PseudoMaskResult:
    normalized = normalize_saliency(raw) if normalize else raw
    if orient:
        normalized = orient_saliency(normalized, field=field)
    return saliency_to_pseudo_mask(
        normalized, normalize=False, field=field, artifact=artifact, **mask_kwargs
    )


def _slice_of(cache: np.ndarray | None, start: int, stop: int) -> np.ndarray | None:
    return None if cache is None else np.asarray(cache[start:stop])


def saliency_array_to_pseudo_masks(
    saliency: np.ndarray,
    out: np.ndarray,
    *,
    orient: bool,
    normalize: bool = True,
    chunk_size: int = 256,
    fields: np.ndarray | None = None,
    images: np.ndarray | None = None,
    **mask_kwargs,
) -> dict:
    """Convert an ``(N, H, W)`` saliency array into ``out`` and report statistics.

    ``orient`` applies the border-versus-center sign fix and ``normalize`` the
    per-image min-max; use both for raw CCAM maps. Student predictions are
    already probabilities with a learned sign, so they use neither.
    ``fields`` is the optional ``(N, H, W)`` field of view cache and ``images``
    the optional ``(N, H, W, 3)`` uint8 cache the artefacts are read from.
    """
    total = len(saliency)
    chunk_size = max(1, chunk_size)
    reasons = {REASON_NO_REGION: 0, REASON_NO_FIELD: 0, REASON_AREA: 0}
    fractions = {FOREGROUND: 0.0, IGNORE: 0.0}
    n_multi_region = 0
    artifact_total = 0.0
    for start in range(0, total, chunk_size):
        stop = start + chunk_size
        chunk = np.asarray(saliency[start:stop], dtype=np.float32)
        field_chunk, image_chunk = _slice_of(fields, start, stop), _slice_of(images, start, stop)
        for offset, raw in enumerate(chunk):
            field = None if field_chunk is None else field_chunk[offset]
            artifact = None if image_chunk is None else artifact_mask(image_chunk[offset])
            artifact_total += 0.0 if artifact is None else float(artifact.mean())
            result = _convert_one(
                raw, field, artifact, orient=orient, normalize=normalize, mask_kwargs=mask_kwargs
            )
            out[start + offset] = result.mask
            if not result.valid:
                reasons[result.reason] = reasons.get(result.reason, 0) + 1
            n_multi_region += int(result.n_regions > 1)
            for label in fractions:
                fractions[label] += float((result.mask == label).mean())
    fractions["artifact"] = artifact_total
    stats = _summarize(total, reasons, fractions, field_used=fields is not None)
    # docs/26 §2.5: how many images the union actually changed.
    stats["n_multi_region"] = n_multi_region
    stats["region_mode"] = mask_kwargs.get("region_mode", "central")
    return stats


def _summarize(total: int, reasons: dict, fractions: dict, *, field_used: bool) -> dict:
    """Totals per exclusion reason; unknown reasons still count towards the total."""
    excluded = sum(reasons.values())
    scale = total if total else 1
    return {
        "images": total,
        "excluded": excluded,
        "excluded_no_region": reasons.get(REASON_NO_REGION, 0),
        "excluded_no_field": reasons.get(REASON_NO_FIELD, 0),
        "excluded_by_area": reasons.get(REASON_AREA, 0),
        "excluded_fraction": excluded / scale if total else 0.0,
        "mean_foreground_fraction": fractions[FOREGROUND] / scale if total else 0.0,
        "mean_ignore_fraction": fractions[IGNORE] / scale if total else 0.0,
        "mean_artifact_fraction": fractions.get("artifact", 0.0) / scale if total else 0.0,
        "field_used": field_used,
    }
