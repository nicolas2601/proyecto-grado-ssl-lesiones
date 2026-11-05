"""Traceable lesion-mask records used by FST and tone transformations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.1"
MIN_LESION_AREA_FRACTION = 0.01
MAX_LESION_AREA_FRACTION = 0.90

# U-Net mask predictions below this confidence are quarantined even when
# their area looks plausible: a right-sized mask can still be a confident
# guess on the wrong pixels. Otsu fallback masks report no confidence
# (``confidence=None``) and are judged on area alone, unchanged.
MIN_MASK_CONFIDENCE = 0.50

# A connected component smaller than this fraction of the mask area is
# speckle, not a second lesion region. Measured against ISIC 2018 expert
# masks: despeckling at this threshold moves Dice by +0,0000 (CI95
# [-0,0000; +0,0001]), so it is safe to ignore these components.
MIN_SIGNIFICANT_COMPONENT_FRACTION = 0.005

# Fragmentation is a DETECTOR of a bad prediction, not a repairable defect.
# Measured on ISIC 2018 (2.594 expert masks): keeping only the largest
# component makes Dice WORSE (-0,0056, CI95 [-0,0090; -0,0023], excludes
# zero), and worse still on the fragmented subset (-0,0334). When the
# largest blob holds under 80 % of the area the secondary components are
# real lesion, so discarding them is destructive.
#
# Abstaining is what works: dropping masks with more than one significant
# component lifts retained Dice from 0,8319 [0,8244; 0,8393] to 0,8818
# [0,8763; 0,8871] -- intervals that do not overlap -- at the cost of
# 16,8 % coverage on ISIC 2018 and 41,0 % on PAD-UFES-20.
#
# This is opt-in. ``max_significant_components=None`` reproduces the
# pre-1.1 contract byte for byte, so existing manifests do not move.
DEFAULT_MAX_SIGNIFICANT_COMPONENTS = 1


def mask_sha256(mask: np.ndarray) -> str:
    canonical = np.ascontiguousarray(mask, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(f"{canonical.shape}:true=lesion".encode("utf-8"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def count_significant_components(
    mask: np.ndarray,
    *,
    min_fraction: float = MIN_SIGNIFICANT_COMPONENT_FRACTION,
) -> int:
    """Count connected lesion components holding at least ``min_fraction``.

    Components are 4-connected (``scipy.ndimage.label`` default cross
    structure). The reported fragmentation rates were measured with
    8-connectivity instead, so the two do not match by construction; the
    difference was checked on the teacher-026 masks and is immaterial
    (PAD 41,3 % under 4-connectivity vs 41,0 % under 8, HAM identical at
    9,3 %, ISIC2019/2020 pool 12,7 % vs 12,6 %). 4-connectivity is the
    stricter of the two, since diagonally touching regions split.

    ``min_fraction`` is read against the total lesion area, not the frame,
    so the count does not drift with mask size.

    An all-background mask has zero components.
    """
    from scipy import ndimage  # noqa: PLC0415 -- keeps the module import light

    lesion_mask = np.asarray(mask, dtype=bool)
    total = int(lesion_mask.sum())
    if total == 0:
        return 0
    labelled, found = ndimage.label(lesion_mask)
    if found == 0:
        return 0
    sizes = np.bincount(labelled.ravel())[1:]
    return int((sizes >= min_fraction * total).sum())


def build_mask_record(
    *,
    dataset: str,
    image_id: str,
    source_revision: str,
    mask: np.ndarray | None,
    producer: str,
    version: str,
    fallback_used: bool = False,
    confidence: float | None = None,
    max_significant_components: int | None = None,
) -> dict:
    """Return a JSON-ready QA record; only accepted masks are eligible.

    ``max_significant_components`` enables the fragmentation abstention
    rule. Leave it ``None`` (the default) to reproduce the pre-1.1
    contract exactly: no component counting, no extra field, no change to
    ``eligible``. Pass ``DEFAULT_MAX_SIGNIFICANT_COMPONENTS`` to quarantine
    fragmented predictions, and declare the coverage cost up front -- it is
    41 % of PAD-UFES-20.
    """
    if max_significant_components is not None and max_significant_components < 1:
        raise ValueError("max_significant_components must be at least 1")
    record = {
        "dataset": dataset,
        "image_id": image_id,
        "source_revision": source_revision,
        "producer": producer,
        "version": version,
        "mask_semantics": "true=lesion",
        "fallback_used": fallback_used,
    }
    if mask is None:
        return {
            **record,
            "mask_sha256": None,
            "mask_pixel_count": None,
            "area_fraction": None,
            "quality": "missing",
            "eligible": False,
            "quarantine_reason": "missing_mask",
        }

    lesion_mask = np.asarray(mask, dtype=bool)
    area_fraction = float(lesion_mask.mean())
    valid_area = MIN_LESION_AREA_FRACTION <= area_fraction <= MAX_LESION_AREA_FRACTION
    low_confidence = confidence is not None and confidence < MIN_MASK_CONFIDENCE
    fragmentation: dict[str, int] = {}
    fragmented = False
    if max_significant_components is not None:
        components = count_significant_components(lesion_mask)
        fragmentation = {"significant_component_count": components}
        fragmented = components > max_significant_components
    eligible = valid_area and not low_confidence and not fragmented
    # Precedence keeps the pre-1.1 order -- confidence before area -- so a
    # record quarantined by the old contract reports the same reason under
    # the new one. ``fragmented`` can only appear last, which makes it the
    # reason only for masks the old contract would have accepted.
    if eligible:
        quarantine_reason = None
    elif low_confidence:
        quarantine_reason = "low_confidence"
    elif not valid_area:
        quarantine_reason = "degenerate_area"
    else:
        quarantine_reason = "fragmented"
    return {
        **record,
        "mask_sha256": mask_sha256(lesion_mask),
        "mask_pixel_count": int(lesion_mask.sum()),
        "area_fraction": area_fraction,
        "confidence": confidence,
        **fragmentation,
        "quality": "accepted" if eligible else "degenerate",
        "eligible": eligible,
        "quarantine_reason": quarantine_reason,
    }


def build_mask_manifest(records: Iterable[Mapping[str, object]]) -> dict:
    """Wrap mask QA records in the JSON-first manifest contract."""
    rows = [dict(record) for record in records]
    total = len(rows)
    eligible = sum(bool(row.get("eligible")) for row in rows)
    summary = {
        "total": total,
        "eligible": eligible,
        "quarantined": total - eligible,
    }
    # The coverage cost of abstaining is reported next to the counts so a
    # reader never has to recompute it, and so a run that silently loses
    # 41 % of a cohort cannot look like a run that lost nothing.
    by_reason: dict[str, int] = {}
    for row in rows:
        reason = row.get("quarantine_reason")
        if reason is not None:
            by_reason[str(reason)] = by_reason.get(str(reason), 0) + 1
    if by_reason:
        summary["quarantined_by_reason"] = dict(sorted(by_reason.items()))
    if total:
        summary["coverage_fraction"] = eligible / total
    return {
        "schema_version": SCHEMA_VERSION,
        "masks": rows,
        "summary": summary,
    }


def write_mask_manifest(manifest: Mapping[str, object], path: Path) -> None:
    """Persist the primary mask contract as deterministic UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
