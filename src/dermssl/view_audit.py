"""Dependency-light contracts for the v8 dataloader audit (not training)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_images(root: Path) -> dict[str, Path]:
    """Fail on ambiguous IDs instead of pairing across datasets silently."""
    if not root.is_dir():
        raise ValueError(f"Missing image directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate image ID: {path.stem}")
        result[path.stem] = path
    return result


def select_ids(originals, transformed, masks, exclusions: set[str], count: int, seed: int):
    """Sample eligible originals; missing pairs are errors, never silent filtering."""
    eligible = sorted(set(originals) - exclusions)
    if count < 1 or count > len(eligible):
        raise ValueError(f"Requested {count} samples from {len(eligible)} eligible originals")
    for name, mapping in (("transformed", transformed), ("pseudo-mask", masks)):
        missing = set(eligible) - set(mapping)
        if missing:
            raise ValueError(f"Missing {name} files for {len(missing)} eligible IDs")
    choice = np.random.default_rng(seed).choice(len(eligible), count, replace=False)
    return sorted(eligible[int(i)] for i in choice)


def load_mask(path: Path, size: tuple[int, int], encoding: str) -> np.ndarray:
    """Explicit encodings prevent ignore=255 from becoming foreground."""
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"Mask/image grids differ: {path}")
        values = np.asarray(image.convert("L"))
    schemes = {"binary255": ({0, 255}, 255), "binary01": ({0, 1}, 1),
               "pseudo01255": ({0, 1, 255}, 1)}
    allowed, foreground = schemes[encoding]
    if not set(np.unique(values)).issubset(allowed):
        raise ValueError(f"Unexpected mask values for {encoding}: {path}")
    return values == foreground


def crop_stats(mask: np.ndarray, crop: tuple[int, int, int, int]) -> dict:
    top, left, height, width = crop
    patch = mask[top:top + height, left:left + width]
    total = int(mask.sum())
    return {"lesion_fraction_in_crop": float(patch.mean()),
            "lesion_retained_fraction": float(patch.sum() / total) if total else None,
            "lesion_absent": not bool(patch.any()), "source_mask_empty": total == 0}


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    rgb[mask] = 0.65 * rgb[mask] + 0.35 * np.array([190, 115, 70])
    return Image.fromarray(np.rint(rgb).astype(np.uint8))
