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

from dermssl import fitzpatrick, mask_manifest, run_environment
from fst_arrays import cache_pixels, load_cache_images, load_field
from fst_flags import (
    TONE_LABELS,
    TONE_UNAVAILABLE,
    flagged_summary,
    fst_distribution,
    tone_distribution,
)

SCHEMA_VERSION = "1.1"
DEFAULT_STATS_NAME = "pseudo_masks_stats.json"
CSV_COLUMNS = (
    "image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita", "n_skin_px"
)
MIN_IMAGES_FOR_TERCILES = 3
# Statuses whose skin medians describe real skin: the ITA bucket may be
# missing (b* floor) but the L* median is still a valid tone reading.
SKIN_STATUSES = ("ok", fitzpatrick.ITA_INDETERMINATE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Cache manifest.csv")
    parser.add_argument("--pseudo-masks", type=Path, required=True, help="pseudo_masks.npy")
    parser.add_argument("--output-json", type=Path, required=True, help="Canonical JSON output")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional image_id,fst_bucket table for tone-stratified evaluation",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional pseudo_masks_stats.json; defaults to the mask directory when present",
    )
    parser.add_argument(
        "--field",
        type=Path,
        default=None,
        help="Optional field.npy (N, H, W); pixels outside the dermoscope field are not skin",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Optional cache images.npy (N, H, W, 3) uint8; read instead of the source files",
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


def _load_image(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Decode one frame at the pseudo-mask resolution.

    The masks live at the cache resolution while the source files keep their
    native size (600x450 is typical), so the frame is resized with the same
    bilinear transform the cache builder uses. Without it every real image
    fails the mask shape check.
    """
    height, width = shape
    with Image.open(path) as image:
        image.draft("RGB", (width, height))
        rgb = image.convert("RGB").resize((width, height), Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the per-image table the evaluation stratifies by.

    ``fst_bucket`` and ``tone_tercile`` are the two stratification columns;
    the medians, the ITA and the skin pixel count travel with them so the
    strata can be audited without reopening the JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for record in records:
            writer.writerow((
                record["image_id"], record["FST_bucket"], record["tone_tercile"],
                record["status"], _csv_value(record["L_median"]),
                _csv_value(record["b_median"]), _csv_value(record["ITA"]),
                record["n_skin_px"],
            ))


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
    return {
        "n": len(rows),
        "states": states,
        "ITA": {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
            "n": int(len(values)),
        },
        "FST_distribution": fst_distribution(rows),
        "tone_distribution": tone_distribution(rows),
        "flagged": flagged_summary(rows),
    }


def _tone_eligible(record: dict[str, Any]) -> bool:
    L_median = record["L_median"]
    return record["status"] in SKIN_STATUSES and isinstance(L_median, float)


def _tone_label(L_median: float, low: float, high: float) -> str:
    if L_median <= low:
        return TONE_LABELS[0]
    if L_median > high:
        return TONE_LABELS[2]
    return TONE_LABELS[1]


def assign_tone_terciles(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Label every record with a skin-lightness tercile over the whole pool.

    The cuts are the 1/3 and 2/3 quantiles of the skin L* median across all
    datasets, so the label is a transparent proxy for tone that does not
    depend on b* (see ``fitzpatrick.MIN_B_FOR_ITA``). Returns the cuts and
    the number of images behind them; images without skin stay unavailable.
    """
    eligible = [record for record in records if _tone_eligible(record)]
    if len(eligible) < MIN_IMAGES_FOR_TERCILES:
        return {"low": None, "high": None, "n": len(eligible)}
    values = np.asarray([record["L_median"] for record in eligible], dtype=np.float64)
    low, high = (float(cut) for cut in np.quantile(values, (1 / 3, 2 / 3)))
    for record in eligible:
        record["tone_tercile"] = _tone_label(record["L_median"], low, high)
    return {"low": low, "high": high, "n": len(eligible)}


def _border_mask(pseudo: np.ndarray, field: np.ndarray | None, index: int) -> np.ndarray:
    """Pixels that are not skin: the ignore band plus everything outside the field."""
    border = pseudo == 255
    if field is None:
        return border
    return border | ~np.asarray(field[index], dtype=bool)


def _base_record(
    row: dict[str, Any],
    metadata: dict[str, Any],
    image_path: Path,
    pseudo: np.ndarray,
    mask_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "index": row["index"],
        "dataset": row["dataset"],
        "image_id": row["image_id"],
        "split": row["split"],
        "image_path": str(image_path),
        "mask_source": metadata["mask_source"],
        "mask_round": metadata["mask_round"],
        "mask_coverage": float((pseudo == 1).mean()),
        "ignore_fraction": float((pseudo == 255).mean()),
        "ITA": None,
        "FST_bucket": "unavailable",
        "L_median": None,
        "b_median": None,
        "n_skin_px": 0,
        "tone_tercile": TONE_UNAVAILABLE,
        "confidence": 0.0,
        "artifact_detected": False,
        "quality": (
            "insufficient_mask_coverage" if not mask_record["eligible"] else mask_record["quality"]
        ),
        "status": "quarantine",
        "mask": mask_record,
        "mask_metadata": metadata["mask_metadata"],
    }


def _apply_estimate(
    record: dict[str, Any],
    image: np.ndarray,
    lesion: np.ndarray,
    mask_record: dict[str, Any],
    border: np.ndarray,
) -> None:
    estimate = fitzpatrick.estimate_fst_from_mask_record(
        image,
        mask=lesion,
        mask_record=mask_record,
        border_mask=border,
    )
    valid = estimate.status == "ok"
    bucket = estimate.bucket if valid and estimate.bucket is not None else "unavailable"
    record.update(
        {
            "ITA": _finite_or_none(estimate.ita) if valid else None,
            "FST_bucket": bucket,
            "L_median": _finite_or_none(estimate.L_median),
            "b_median": _finite_or_none(estimate.b_median),
            "n_skin_px": int(estimate.n_skin_px),
            "confidence": float(estimate.confidence),
            "artifact_detected": bool(estimate.artifact_detected),
            "status": estimate.status if estimate.status in SKIN_STATUSES else "quarantine",
            "quality": mask_record["quality"] if valid else f"invalid_estimate:{estimate.status}",
        }
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _pixels(images: np.ndarray | None, index: int, path: Path, shape) -> np.ndarray:
    """The cache frame when ``--images`` is given, else the resized source file."""
    return _load_image(path, shape) if images is None else cache_pixels(images, index)


def _records(args: argparse.Namespace, rows, masks, arrays, metadata) -> list[dict[str, Any]]:
    field, images = arrays
    records: list[dict[str, Any]] = []
    for row in rows:
        index = row["index"]
        pseudo = np.asarray(masks[index])
        if not np.isin(pseudo, (0, 1, 255)).all():
            raise ValueError("pseudo_masks may contain only 0, 1 and 255")
        lesion = pseudo == 1
        image_path = _image_path(row["path"], args.manifest).resolve()
        mask_record = mask_manifest.build_mask_record(
            dataset=row["dataset"],
            image_id=row["image_id"],
            source_revision=args.source_revision,
            mask=lesion,
            producer=str(metadata["mask_source"]),
            version=str(metadata["mask_round"] or "unknown"),
        )
        record = _base_record(row, metadata, image_path, pseudo, mask_record)
        if mask_record["eligible"]:
            _apply_estimate(
                record, _pixels(images, index, image_path, lesion.shape), lesion, mask_record,
                _border_mask(pseudo, field, index),
            )
        records.append(record)
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    pseudo_masks = np.load(args.pseudo_masks, mmap_mode="r")
    if pseudo_masks.ndim != 3 or pseudo_masks.dtype != np.uint8:
        raise ValueError("pseudo_masks must be uint8 with shape (N, H, W)")
    if rows and max(row["index"] for row in rows) >= len(pseudo_masks):
        raise ValueError("manifest index exceeds pseudo_masks length")

    field = load_field(args.field, pseudo_masks)
    images = load_cache_images(args.images, pseudo_masks)
    metadata = _metadata(_load_stats(args.pseudo_masks, args.stats_json))
    records = _records(args, rows, pseudo_masks, (field, images), metadata)
    tone_cuts = assign_tone_terciles(records)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dataset"]].append(record)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "field_used": field is not None,
        "images_used": None if args.images is None else str(args.images),
        "records": records,
        "summary": {
            "total": len(records),
            "tone_tercile_cuts": tone_cuts,
            "by_dataset": {
                dataset: _dataset_summary(dataset_rows)
                for dataset, dataset_rows in sorted(grouped.items())
            },
        },
        "environment": run_environment.snapshot(args),
    }
    if args.output_csv is not None:
        _write_csv(args.output_csv, records)
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
