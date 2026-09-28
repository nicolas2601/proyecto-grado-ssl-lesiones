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
from types import MappingProxyType
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


# Segmentation run -> the producer its masks are recorded under. The producer
# is DERIVED from the run rather than passed in, so a caller cannot write A2's
# masks under student_r3's name; the consumer re-derives the run from the
# checkpoint sha (c1_mask_provenance) and checks both agree.
#   final       the published student_r3 (docs/24), the primary C1 arm
#   s4_unet_a2  S4-UNet A2 (docs/27), which met its prediction: C1-v7 SECONDARY
RUN_PRODUCERS = MappingProxyType({"final": "student_r3", "s4_unet_a2": "s4_unet_a2"})

_PRODUCER_NOTES = MappingProxyType({
    "student_r3": (
        "USL-Net path B: MoCo v2 -> CCAM -> uncertainty -> student U-Net. Trained "
        "without reference masks or dx labels (docs/23), so feeding its predictions "
        "into the classifier keeps C1 self-supervised."
    ),
    "s4_unet_a2": (
        "S4-UNet A2 (docs/27): frozen public DINOv2 ViT-B/14 (self-supervised) + DPT "
        "decoder trained on the r0 CCAM pseudo-masks of the student_r3 pool. No "
        "reference mask or dx label enters training, so C1 stays self-supervised."
    ),
})

_RUN_NOTES = MappingProxyType({
    "final": (
        "The published run: Dice 0,718 ISIC / 0,785 HAM10000 (docs/24). NOT the "
        "'v2' max_area_fraction ablation, which scored 0,716 / 0,783 and was "
        "closed as a default."
    ),
    "s4_unet_a2": (
        "SECONDARY C1-v7 arm: A2 met its docs/27 prediction and replaces student_r3 "
        "only in this run. The primary C1 table stays conditioned on student_r3."
    ),
})


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


# Everything that changes a mask without changing the checkpoint. The published
# evaluation and C1 share a checkpoint and still disagree if either of these
# differs, so recording the checkpoint alone does not identify the masks:
#   path        the function that ran the network, so a reader can diff it
#   tta         flip-averaging changes the probabilities
#   resolution  the grid the network saw
#   threshold   where the sigmoid was cut
#   upsample    whether probabilities or labels were resized back
INFERENCE_KEYS = ("path", "tta", "resolution", "threshold", "upsample")


def _validate_inference(inference: dict[str, Any]) -> dict[str, Any]:
    """Every key present and typed, with no defaults.

    A default here would repeat the bug that made ``segmentation_run`` unsafe:
    the producer would describe its own inference, and a run with TTA on would
    record ``tta: false`` because nobody passed it.
    """
    missing = [key for key in INFERENCE_KEYS if key not in inference]
    if missing:
        raise ValueError(
            f"inference is missing {missing}; every field is required because a "
            "default would let a run describe itself as the published one"
        )
    unknown = sorted(set(inference) - set(INFERENCE_KEYS))
    if unknown:
        raise ValueError(f"inference has unexpected keys {unknown}; expected {list(INFERENCE_KEYS)}")
    if not isinstance(inference["tta"], bool):
        raise TypeError(
            f"inference['tta'] must be bool, not {type(inference['tta']).__name__}; "
            "a truthy string would record TTA as enabled either way"
        )
    if not isinstance(inference["resolution"], int) or isinstance(inference["resolution"], bool):
        raise TypeError("inference['resolution'] must be an int")
    if not isinstance(inference["threshold"], float):
        raise TypeError("inference['threshold'] must be a float")
    return {key: inference[key] for key in INFERENCE_KEYS}


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
    inference: dict[str, Any],
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
    if segmentation_run not in RUN_PRODUCERS:
        raise ValueError(
            f"segmentation_run {segmentation_run!r} is not a registered segmenter; "
            f"expected one of {sorted(RUN_PRODUCERS)}"
        )
    producer = RUN_PRODUCERS[segmentation_run]

    return {
        "artifact": "c1_pseudo_masks",
        "producer": producer,
        "producer_is_label_free": True,
        "producer_note": _PRODUCER_NOTES[producer],
        "segmentation_run": segmentation_run,
        "segmentation_commit": segmentation_commit,
        "segmentation_run_note": _RUN_NOTES[segmentation_run],
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "inference": _validate_inference(inference),
        "inference_note": (
            "The same weights produce different masks under a different inference "
            "path. C1 reuses the published one (eval_student.predict_masks, which "
            "applies ImageNet normalisation) instead of reimplementing it, and "
            "upsamples the probabilities to the original grid BEFORE thresholding "
            "so the decision is taken once."
        ),
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
                "The segmenter's sigmoid is thresholded into a binary mask, so no "
                "uncertain band exists in this artifact. Downstream code still treats "
                "255 as non-lesion, matching the exporter convention."
            ),
        },
        "coverage": coverage or {},
    }
