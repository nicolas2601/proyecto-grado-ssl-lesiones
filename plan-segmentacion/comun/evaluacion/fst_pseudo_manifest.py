"""Build a JSON-first ITA/FST manifest from Camino B pseudo-masks.

The adapter is deliberately standalone: it consumes the cache manifest and
the pseudo-mask artifact without changing the training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl import fitzpatrick, mask_manifest

SCHEMA_VERSION = "1.0"
DEFAULT_STATS_NAME = "pseudo_masks_stats.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Cache manifest.csv")
    parser.add_argument("--pseudo-masks", type=Path, required=True, help="pseudo_masks.npy")
    parser.add_argument("--output-json", type=Path, required=True, help="Canonical JSON output")
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional pseudo_masks_stats.json; defaults to the mask directory when present",
    )
    parser.add_argument(
        "--source-revision",
        default="unknown",
        help="Revision recorded in the shared mask provenance contract",
    )
    return parser.parse_args(argv)


def _load_stats(path: Path, explicit_path: Path | None) -> dict[str, Any]:
    candidate = explicit_path or path.parent / DEFAULT_STATS_NAME
    if not candidate.is_file():
        return {}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stats JSON must contain an object: {candidate}")
    return payload


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"index", "dataset", "image_id", "path", "split"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    for row in rows:
        row["index"] = int(row["index"])
    return rows


def _image_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _metadata(stats: dict[str, Any]) -> dict[str, Any]:
    source = stats.get("mask_source", stats.get("source", "pseudo_masks"))
    mask_round = stats.get("mask_round", stats.get("round"))
    return {
        "mask_source": source,
        "mask_round": mask_round,
        "mask_metadata": dict(stats),
    }


def _finite_values(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["ITA"]) for row in rows if isinstance(row["ITA"], (int, float)) and math.isfinite(row["ITA"])]


def _dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(_finite_values(rows), dtype=np.float64)
    states = dict(sorted(Counter(row["status"] for row in rows).items()))
    buckets = Counter(
        str(row["FST_bucket"])
        for row in rows
        if row["FST_bucket"] != "unavailable"
    )
    return {
        "n": len(rows),
        "states": states,
        "ITA": {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
            "n": int(len(values)),
        },
        "FST_distribution": dict(sorted(buckets.items())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    pseudo_masks = np.load(args.pseudo_masks, mmap_mode="r")
    if pseudo_masks.ndim != 3 or pseudo_masks.dtype != np.uint8:
        raise ValueError("pseudo_masks must be uint8 with shape (N, H, W)")
    if rows and max(row["index"] for row in rows) >= len(pseudo_masks):
        raise ValueError("manifest index exceeds pseudo_masks length")

    stats = _load_stats(args.pseudo_masks, args.stats_json)
    metadata = _metadata(stats)
    records: list[dict[str, Any]] = []
    for row in rows:
        index = row["index"]
        pseudo = np.asarray(pseudo_masks[index])
        if not np.isin(pseudo, (0, 1, 255)).all():
            raise ValueError("pseudo_masks may contain only 0, 1 and 255")
        valid = pseudo != 255
        lesion = pseudo == 1
        mask_coverage = float(lesion.mean())
        ignore_fraction = float((~valid).mean())
        image_path = _image_path(row["path"], args.manifest).resolve()
        mask_record = mask_manifest.build_mask_record(
            dataset=row["dataset"],
            image_id=row["image_id"],
            source_revision=args.source_revision,
            mask=lesion,
            producer=str(metadata["mask_source"]),
            version=str(metadata["mask_round"] or "unknown"),
        )

        record: dict[str, Any] = {
            "index": index,
            "dataset": row["dataset"],
            "image_id": row["image_id"],
            "split": row["split"],
            "image_path": str(image_path),
            "mask_source": metadata["mask_source"],
            "mask_round": metadata["mask_round"],
            "mask_coverage": mask_coverage,
            "ignore_fraction": ignore_fraction,
            "ITA": None,
            "FST_bucket": "unavailable",
            "confidence": 0.0,
            "quality": (
                "insufficient_mask_coverage"
                if not mask_record["eligible"]
                else mask_record["quality"]
            ),
            "status": "quarantine",
            "mask": mask_record,
            "mask_metadata": metadata["mask_metadata"],
        }
        if mask_record["eligible"]:
            estimate = fitzpatrick.estimate_fst_from_mask_record(
                _load_image(image_path),
                mask=lesion,
                mask_record=mask_record,
                valid_pixel_mask=valid,
            )
            estimate_valid = estimate.status == "ok"
            record.update(
                {
                    "ITA": None if not math.isfinite(estimate.ita) else float(estimate.ita),
                    "FST_bucket": estimate.bucket if estimate.bucket is not None else "unavailable",
                    "confidence": float(estimate.confidence),
                    "status": estimate.status if estimate_valid else "quarantine",
                    "quality": (
                        mask_record["quality"]
                        if estimate_valid
                        else f"invalid_estimate:{estimate.status}"
                    ),
                }
            )
            if not estimate_valid:
                record["ITA"] = None
                record["FST_bucket"] = "unavailable"
        records.append(record)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dataset"]].append(record)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "records": records,
        "summary": {"total": len(records), "by_dataset": {
            dataset: _dataset_summary(dataset_rows)
            for dataset, dataset_rows in sorted(grouped.items())
        }},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
