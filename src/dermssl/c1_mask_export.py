"""Provenance for the C1 step-0 pseudo-mask artifact.

C1 runs 24 cells (6 backbones x 4 conditions) and every one of them consumes
the same `.npy` of predicted masks. That file is therefore the single input
that can invalidate the whole grid, and it is produced by a model run rather
than shipped with the dataset -- rerun the segmenter with a different
checkpoint or a changed threshold and you get a file with the same name,
different contents, and no way to tell which one produced the published
numbers.

So the artifact is pinned: the `.npy` carries a sha256, the checkpoint that
produced it carries a sha256, and every C1 result references the first one.
The torch-dependent inference lives in ``scripts/c1_predict_masks.py``; this
module is the part worth testing, and it stays importable without torch.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

LESION_LABEL = 1
BACKGROUND_LABEL = 0

# student_r3 emits a BINARY mask: ``infer_mask`` thresholds the sigmoid and
# returns bool (src/dermssl/torch_unet.py:328). There is no uncertainty band in
# its output -- the 0/1/255 convention belongs to the upstream pseudo-mask
# rounds (CCAM + uncertainty), not here. Recorded as None rather than omitted
# so a later reader sees the question was asked and answered.
UNCERTAIN_LABEL: int | None = None

_SHA256 = re.compile(r"[0-9a-f]{64}")


def is_sha256(value: str) -> bool:
    """The single definition of a valid digest, shared by every C1 module.

    It lived in three places after the module split, which is how two of them
    drift apart and one of them starts accepting uppercase.
    """
    return bool(_SHA256.fullmatch(str(value)))
_CHUNK = 1 << 16


def encode_mask(mask: np.ndarray) -> np.ndarray:
    """Boolean lesion mask -> uint8 in {0, 1}.

    Refuses anything non-boolean: handing this a probability map would truncate
    it to zeros almost everywhere and produce an artifact that looks valid.
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise TypeError(
            f"expected a bool mask, got {array.dtype}; threshold it before encoding "
            "so the decision is explicit rather than implied by a cast"
        )
    return array.astype(np.uint8) * LESION_LABEL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, expected_sha256: str) -> None:
    """Fail loudly when the file on disk is not the one the manifest describes."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {path.name}: manifest says {expected_sha256}, "
            f"file is {actual}. The masks were regenerated; rerun C1 or restore the file."
        )


def summarise_coverage(masks: np.ndarray) -> dict[str, float | int]:
    """Aggregate stats that make a failed segmentation run visible.

    ``n_empty`` matters most: an image with no predicted lesion forces the bbox
    fallback in condition (a) and the CLS fallback in condition (c). If that
    count is high, two of the four conditions have quietly collapsed toward the
    baseline and the comparison no longer measures what its name claims.
    """
    per_image = masks.reshape(len(masks), -1)
    foreground = (per_image == LESION_LABEL).mean(axis=1)
    return {
        "n_images": int(len(masks)),
        "n_empty": int((foreground == 0).sum()),
        "mean_foreground_fraction": float(foreground.mean()),
        "min_foreground_fraction": float(foreground.min()),
        "max_foreground_fraction": float(foreground.max()),
    }


def build_masks_manifest(
    *,
    n_images: int,
    image_ids: list[str],
    masks_sha256: str,
    checkpoint_sha256: str,
    checkpoint_path: str,
    code_revision: str,
    mask_shape: list[int],
    segmentation_run: str,
    segmentation_commit: str,
    partial: bool = False,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The record that lets someone else prove which masks produced C1."""
    if len(image_ids) != n_images:
        raise ValueError(f"image_ids has {len(image_ids)} entries but n_images is {n_images}")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("duplicate entries in image_ids; the mask index would be ambiguous")
    for name, value in (("masks_sha256", masks_sha256), ("checkpoint_sha256", checkpoint_sha256)):
        if not is_sha256(value):
            raise ValueError(f"{name} is not a sha256 hex digest: {value!r}")

    return {
        "artifact": "c1_pseudo_masks",
        "producer": "student_r3",
        "producer_is_label_free": True,
        "producer_note": (
            "USL-Net path B: MoCo v2 -> CCAM -> uncertainty -> student U-Net. Trained "
            "without reference masks or dx labels (docs/23), so feeding its predictions "
            "into the classifier keeps C1 self-supervised."
        ),
        "segmentation_run": segmentation_run,
        "segmentation_commit": segmentation_commit,
        "segmentation_run_note": (
            "The published run: Dice 0,718 ISIC / 0,785 HAM10000 (docs/24). NOT the "
            "'v2' max_area_fraction ablation, which scored 0,716 / 0,783 and was "
            "closed as a default."
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "masks_sha256": masks_sha256,
        "code_revision": code_revision,
        "partial": bool(partial),
        "n_images": n_images,
        "mask_shape": list(mask_shape),
        "image_ids": list(image_ids),
        "label_convention": {
            "lesion": LESION_LABEL,
            "background": BACKGROUND_LABEL,
            "uncertain": UNCERTAIN_LABEL,
            "note": (
                "student_r3 thresholds the sigmoid and returns a binary mask, so no "
                "uncertain band exists in this artifact. Downstream code still treats "
                "255 as non-lesion, matching the exporter convention."
            ),
        },
        "coverage": coverage or {},
    }
