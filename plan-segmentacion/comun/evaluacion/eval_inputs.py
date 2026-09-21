"""File-system side of the evaluation stage: pairing and decoding.

Split out of ``evaluate_segmentation.py`` so that module stays small. This is
evaluation-only code: together with ``evaluate_segmentation`` it is the only
place allowed to open reference masks, and it runs after training, never
before. Nothing here selects a checkpoint or a threshold.

Predicted masks from a third party (a published baseline, another pipeline)
are decoded exactly like a reference mask: nearest-neighbour resize, then
``> 127``. Any anti-aliasing a producer left in its PNG is therefore resolved
the same way on both sides of the comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import eval_report

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_CUTOFF = 127


def pair_images_and_masks(
    image_dir: Path,
    mask_dir: Path,
    mask_suffix: str = "_segmentation",
    mask_extension: str = ".png",
) -> list[tuple[str, Path, Path]]:
    """Match every image with ``<id><suffix><ext>`` in the reference folder."""
    images = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    pairs, missing = [], []
    for image_path in images:
        mask_path = Path(mask_dir) / f"{image_path.stem}{mask_suffix}{mask_extension}"
        if mask_path.is_file():
            pairs.append((image_path.stem, image_path, mask_path))
        else:
            missing.append(image_path.stem)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images have no mask (first: {missing[:3]}); check --mask-suffix"
        )
    eval_report.reject_duplicate_ids([image_id for image_id, _, _ in pairs])
    return pairs


def pair_predictions(
    pairs: list[tuple[str, Path, Path]],
    prediction_dir: Path,
    suffix: str,
    extension: str,
) -> dict[str, Path]:
    """Locate ``<id><suffix><ext>`` for every already paired image.

    Fails on the first incomplete set instead of silently reporting metrics
    over the subset a baseline happened to cover.
    """
    prediction_dir = Path(prediction_dir)
    found: dict[str, Path] = {}
    missing = []
    for image_id, _, _ in pairs:
        path = prediction_dir / f"{image_id}{suffix}{extension}"
        if path.is_file():
            found[image_id] = path
        else:
            missing.append(image_id)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images have no prediction (first: {missing[:3]}); "
            "check --pred-mask-suffix and --pred-mask-extension"
        )
    return found


def load_binary_mask(path: Path, resolution: int) -> np.ndarray:
    """Read a mask PNG, resize with nearest neighbour and binarize above 127."""
    from PIL import Image

    with Image.open(path) as image:
        resized = image.convert("L").resize((resolution, resolution), Image.NEAREST)
        return np.asarray(resized) > MASK_CUTOFF


def load_image(path: Path, resolution: int) -> np.ndarray:
    """Read an RGB image and resize it bilinearly to the evaluation resolution."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)
