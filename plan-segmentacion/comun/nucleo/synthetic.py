"""Synthetic lesion-like images for the smoke test (no real data involved)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def make_lesion_image(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image uint8 HxWx3, mask bool HxW)``: a dark blob on textured skin."""
    axis = (np.arange(size) + 0.5) / size
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    center_y, center_x = 0.5 + rng.uniform(-0.08, 0.08, size=2)
    radius_y, radius_x = rng.uniform(0.18, 0.30, size=2)
    distance = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2
    mask = distance <= 1.0

    skin = rng.uniform(150, 210, size=3)
    lesion = rng.uniform(40, 100, size=3)
    image = np.where(mask[..., None], lesion, skin) + rng.normal(0, 6, size=(size, size, 3))
    return np.clip(image, 0, 255).astype(np.uint8), mask


def write_pool_images(directory: Path, count: int, size: int, seed: int) -> list[Path]:
    """Write ``count`` synthetic PNG images to emulate an unlabeled pool."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for index in range(count):
        image, _ = make_lesion_image(rng, size)
        path = directory / f"SYN_{index:05d}.png"
        Image.fromarray(image).save(path)
        paths.append(path)
    return paths


def write_eval_set(
    image_dir: Path, mask_dir: Path, count: int, size: int, seed: int
) -> list[str]:
    """Write image/mask pairs in the ISIC 2018 naming (``<id>_segmentation.png``)."""
    from PIL import Image

    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = []
    for index in range(count):
        image_id = f"SYNEVAL_{index:05d}"
        image, mask = make_lesion_image(rng, size)
        Image.fromarray(image).save(image_dir / f"{image_id}.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            mask_dir / f"{image_id}_segmentation.png"
        )
        ids.append(image_id)
    return ids


def main() -> None:
    """Write a synthetic image/mask evaluation set (smoke test only)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from dermssl.cli import build_spanish_parser

    parser = build_spanish_parser(description="Genera un conjunto de evaluacion sintetico (solo smoke test)")
    parser.add_argument("--image-dir", required=True, help="Carpeta de imagenes de salida")
    parser.add_argument("--mask-dir", required=True, help="Carpeta de mascaras de salida")
    parser.add_argument("--count", type=int, default=16, help="Cantidad de pares")
    parser.add_argument("--size", type=int, default=64, help="Lado en pixeles")
    parser.add_argument("--seed", type=int, default=123, help="Semilla")
    args = parser.parse_args()
    ids = write_eval_set(Path(args.image_dir), Path(args.mask_dir), args.count, args.size, args.seed)
    print(f"{len(ids)} pares sinteticos escritos")


if __name__ == "__main__":
    main()
