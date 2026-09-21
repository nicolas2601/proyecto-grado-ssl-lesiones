"""Cache layout shared by every stage: uint8 image memmap, manifest CSV, hold-out.

The cache never stores labels, diagnoses or masks. Images are resized once to a
square and written to a ``.npy`` memmap; a CSV manifest maps row index to the
source image and to its ``train`` / ``val`` (pseudo-validation) split.

Pseudo-mask arrays use the same layout ``(N, H, W)`` with three label values:
``BACKGROUND``, ``FOREGROUND`` and ``IGNORE`` (excluded from the loss).
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MANIFEST_COLUMNS = ("index", "dataset", "image_id", "path", "split")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "val"

# Pseudo-mask label values (uint8).
BACKGROUND = 0
FOREGROUND = 1
IGNORE = 255


@dataclass(frozen=True)
class ImageRecord:
    """One source image. ``dataset`` is the user-given root name."""

    dataset: str
    image_id: str
    path: str


@dataclass(frozen=True)
class ScanResult:
    records: list[ImageRecord]
    duplicates_dropped: int


def parse_root_spec(spec: str) -> tuple[str, Path]:
    """Parse ``NAME=PATH`` into ``(name, Path)``."""
    name, separator, raw_path = spec.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError(f"expected NAME=PATH, got {spec!r}")
    return name.strip(), Path(raw_path.strip())


def _list_images(root: Path, extensions: Sequence[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {root}")
    allowed = {ext.lower() for ext in extensions}
    matches = (
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )
    return sorted(matches, key=lambda path: (path.stem, str(path)))


def scan_image_roots(
    roots: Mapping[str, Path],
    *,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
    dedupe_by_id: bool = True,
) -> ScanResult:
    """List images of every root in a deterministic order.

    ISIC datasets overlap (ISIC 2019 contains HAM10000 images) and share
    global image ids, so with ``dedupe_by_id`` the first root that provides an
    id wins and later copies are dropped.
    """
    records: list[ImageRecord] = []
    seen_ids: set[str] = set()
    dropped = 0
    for dataset, root in roots.items():
        for path in _list_images(Path(root), extensions):
            if dedupe_by_id:
                if path.stem in seen_ids:
                    dropped += 1
                    continue
                seen_ids.add(path.stem)
            records.append(ImageRecord(dataset, path.stem, str(path)))
    return ScanResult(records=records, duplicates_dropped=dropped)


def _hash_rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def assign_splits(keys: Sequence[str], holdout_fraction: float, seed: int) -> list[str]:
    """Hold out a deterministic fraction, independent of the input order.

    Keys are ranked by a seeded hash and the lowest ranks become ``val``. At
    least one image is held out whenever there are two or more images.
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    total = len(keys)
    if total < 2:
        return [SPLIT_TRAIN] * total
    holdout_count = max(1, round(total * holdout_fraction)) if holdout_fraction > 0 else 0
    ranked = sorted(range(total), key=lambda index: _hash_rank(keys[index], seed))
    validation_positions = set(ranked[:holdout_count])
    return [
        SPLIT_VALIDATION if index in validation_positions else SPLIT_TRAIN
        for index in range(total)
    ]


def write_manifest(path: Path, records: Sequence[ImageRecord], splits: Sequence[str]) -> None:
    if len(records) != len(splits):
        raise ValueError("records and splits must have the same length")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for index, (record, split) in enumerate(zip(records, splits, strict=True)):
            writer.writerow([index, record.dataset, record.image_id, record.path, split])


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["index"] = int(row["index"])
    return rows


def split_indices(rows: Sequence[Mapping[str, object]], split: str) -> np.ndarray:
    """Row indices belonging to ``split``, ascending."""
    return np.array([row["index"] for row in rows if row["split"] == split], dtype=np.int64)


def create_cache(path: Path, shape: Sequence[int], dtype) -> np.memmap:
    """Create a writable ``.npy`` memmap (self-describing shape and dtype)."""
    return np.lib.format.open_memmap(
        Path(path), mode="w+", dtype=dtype, shape=tuple(int(s) for s in shape)
    )


def open_cache(path: Path, mode: str = "r", preload: bool = False) -> np.ndarray:
    """Open a cache as memmap, or copy it into RAM when explicitly requested."""
    return np.load(Path(path), mmap_mode=None if preload else mode)


def load_resized_rgb(path: str | Path, resolution: int) -> np.ndarray:
    """Decode one image and resize it once to ``(resolution, resolution, 3)`` uint8."""
    from PIL import Image  # local: keeps the module importable without Pillow

    with Image.open(path) as image:
        image.draft("RGB", (resolution, resolution))
        rgb = image.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)
