"""Traceable lesion-mask records used by FST and tone transformations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
MIN_LESION_AREA_FRACTION = 0.01
MAX_LESION_AREA_FRACTION = 0.90

# U-Net mask predictions below this confidence are quarantined even when
# their area looks plausible: a right-sized mask can still be a confident
# guess on the wrong pixels. Otsu fallback masks report no confidence
# (``confidence=None``) and are judged on area alone, unchanged.
MIN_MASK_CONFIDENCE = 0.50


def mask_sha256(mask: np.ndarray) -> str:
    canonical = np.ascontiguousarray(mask, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(f"{canonical.shape}:true=lesion".encode("utf-8"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


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
) -> dict:
    """Return a JSON-ready QA record; only accepted masks are eligible."""
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
    eligible = valid_area and not low_confidence
    if eligible:
        quarantine_reason = None
    elif low_confidence:
        quarantine_reason = "low_confidence"
    else:
        quarantine_reason = "degenerate_area"
    return {
        **record,
        "mask_sha256": mask_sha256(lesion_mask),
        "mask_pixel_count": int(lesion_mask.sum()),
        "area_fraction": area_fraction,
        "confidence": confidence,
        "quality": "accepted" if eligible else "degenerate",
        "eligible": eligible,
        "quarantine_reason": quarantine_reason,
    }


def build_mask_manifest(records: Iterable[Mapping[str, object]]) -> dict:
    """Wrap mask QA records in the JSON-first manifest contract."""
    rows = [dict(record) for record in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "masks": rows,
        "summary": {
            "total": len(rows),
            "eligible": sum(bool(row.get("eligible")) for row in rows),
            "quarantined": sum(not bool(row.get("eligible")) for row in rows),
        },
    }


def write_mask_manifest(manifest: Mapping[str, object], path: Path) -> None:
    """Persist the primary mask contract as deterministic UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
