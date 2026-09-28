"""Provenance shared by the S4-UNet scripts (docs/27): pool and weights identity.

The pool fingerprint is what makes "the same pool as student_r3" checkable:
the cache manifest (row -> image id, path and split) and the image array
shape. Hashing the 8.8 GB ``images.npy`` would add minutes for no extra
guarantee, since the manifest fixes which file fills every row.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src",):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl.pool_manifest import sha256_file  # noqa: E402

PUBLIC = "public"
IMAGES_FILE = "images.npy"
MANIFEST_FILE = "manifest.csv"
TRANSDUCTIVE_NOTE = (
    "transductive without labels: the pool is the student_r3 cache (HAM10000, ISIC 2019 "
    "and ISIC 2020, deduplicated by id), so evaluation images that are also in the pool "
    "(722 ISIC 2018, 9,822 HAM10000) were seen as unlabelled images; no reference mask "
    "and no label is read before the evaluation stage"
)


def pool_fingerprint(cache_dir: Path) -> dict:
    """Manifest digest, row count, split sizes and image array shape."""
    import numpy as np

    cache_dir = Path(cache_dir)
    manifest = cache_dir / MANIFEST_FILE
    with manifest.open(newline="", encoding="utf-8") as handle:
        splits = Counter(row["split"] for row in csv.DictReader(handle))
    images = np.load(cache_dir / IMAGES_FILE, mmap_mode="r")
    return {
        "cache_dir": str(cache_dir),
        "manifest_sha256": sha256_file(manifest),
        "n_images": int(images.shape[0]),
        "image_shape": [int(s) for s in images.shape[1:]],
        "splits": dict(sorted(splits.items())),
    }


def weights_digest(source: str) -> str:
    """``public`` for the hub weights, else the sha256 of the state_dict file."""
    return PUBLIC if source == PUBLIC else sha256_file(Path(source).expanduser())


def parse_backbone(raw: str) -> tuple[str, str]:
    """``ARCH:SOURCE`` -> (arch, source); SOURCE is ``public`` or a file."""
    arch, colon, source = raw.partition(":")
    if not colon or not arch or not source:
        raise ValueError(f"expected ARCH:SOURCE, got {raw!r}")
    return arch, source


__all__ = ["pool_fingerprint", "weights_digest", "parse_backbone", "sha256_file",
           "TRANSDUCTIVE_NOTE", "PUBLIC"]
