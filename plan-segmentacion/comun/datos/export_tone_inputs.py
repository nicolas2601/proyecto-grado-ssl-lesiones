"""Export the camino B cache and pseudo-masks as inputs of the CIRCLe tone transform.

``dermssl.pretrain.tone_transforms.generate_tone_transformations`` wants an
image root, a mirror of binary PNG masks and a ``mask_manifest.json`` whose
records vouch for every image under the root. The label-free pipeline keeps
its images in a uint8 cache and its masks in ``pseudo_masks.npy``; this
adapter materialises both at the cache resolution and writes only the images
whose mask passes the shared QA (``dermssl.mask_manifest``), so the paired
pretraining sees exactly the images that have a protected lesion.

    python export_tone_inputs.py --cache-dir salidas/final/cache \\
        --pseudo-masks salidas/final/pseudo_r2/pseudo_masks.npy \\
        --output-root salidas/tone/inputs --source-revision pool-final
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cache_io  # noqa: E402
from dermssl import mask_manifest  # noqa: E402

ORIGINALS_DIR = "originals"
MASKS_DIR = "masks"
MANIFEST_NAME = "mask_manifest.json"
SUMMARY_NAME = "export_summary.json"
LESION_LABEL = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache with images.npy and manifest.csv")
    parser.add_argument("--pseudo-masks", type=Path, required=True, help="pseudo_masks.npy of the last round")
    parser.add_argument("--output-root", type=Path, required=True, help="New folder for originals/, masks/ and the manifest")
    parser.add_argument("--source-revision", required=True, help="Data revision recorded in every mask record")
    parser.add_argument("--producer", default="camino-b-student", help="Mask producer recorded in the manifest")
    parser.add_argument("--version", default="unknown", help="Mask version (commit or round) recorded in the manifest")
    parser.add_argument("--limit", type=int, default=0, help="Export only the first N manifest rows (smoke)")
    return parser.parse_args(argv)


def _reject_duplicates(rows: list[dict]) -> None:
    repeated = sorted(name for name, count in Counter(row["image_id"] for row in rows).items() if count > 1)
    if repeated:
        raise ValueError(f"duplicate image_id in the cache manifest: {repeated[:3]}")


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def _export_one(row: dict, image: np.ndarray, lesion: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    record = mask_manifest.build_mask_record(
        dataset=row["dataset"], image_id=row["image_id"], source_revision=args.source_revision,
        mask=lesion, producer=args.producer, version=args.version,
    )
    if not record["eligible"]:
        return record
    relative = Path(row["dataset"]) / f"{row['image_id']}.png"
    _save_png(args.output_root / ORIGINALS_DIR / relative, np.asarray(image, dtype=np.uint8))
    _save_png(args.output_root / MASKS_DIR / relative, (lesion.astype(np.uint8) * 255))
    return record


def _summary(records: list[dict], args: argparse.Namespace, resolution: int) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, int]] = {}
    for record in records:
        entry = by_dataset.setdefault(record["dataset"], {"total": 0, "eligible": 0, "quarantined": 0})
        entry["total"] += 1
        entry["eligible" if record["eligible"] else "quarantined"] += 1
    reasons = Counter(record["quarantine_reason"] for record in records if not record["eligible"])
    return {
        "source_revision": args.source_revision,
        "producer": args.producer,
        "version": args.version,
        "resolution": resolution,
        "total": len(records),
        "eligible": sum(record["eligible"] for record in records),
        "quarantine_reasons": dict(sorted(reasons.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = cache_io.read_manifest(args.cache_dir / "manifest.csv")
    if args.limit > 0:
        rows = rows[: args.limit]
    _reject_duplicates(rows)
    images = cache_io.open_cache(args.cache_dir / "images.npy")
    masks = np.load(args.pseudo_masks, mmap_mode="r")
    if masks.ndim != 3 or masks.shape[0] != images.shape[0]:
        raise ValueError(f"pseudo_masks {masks.shape} does not match the cache {images.shape[:3]}")

    records = []
    for row in rows:
        index = row["index"]
        lesion = np.asarray(masks[index]) == LESION_LABEL
        records.append(_export_one(row, images[index], lesion, args))

    args.output_root.mkdir(parents=True, exist_ok=True)
    mask_manifest.write_mask_manifest(mask_manifest.build_mask_manifest(records), args.output_root / MANIFEST_NAME)
    summary = _summary(records, args, int(images.shape[1]))
    (args.output_root / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
