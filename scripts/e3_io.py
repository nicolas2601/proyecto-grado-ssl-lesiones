"""Real-data side of ``e3_run.py``: split and labels, the pinned masks, the B71 extractor.

Everything heavy (torch, torchvision, the v7.1 weight bank) is imported inside
:func:`build_runtime`, so ``e3_run.py`` and its tests import this module without them. A
test passes its own :class:`Runtime` with a fake extractor and synthetic items instead.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HAIR_DIR = REPO / "plan-segmentacion" / "comun" / "nucleo"
B71_WEIGHTS = "B_s42_a1.0"
REQUIRED_PRODUCER = "student_r3"
PROTOCOL = "B_s42_a1.0@224|cls4|balanced|tta"
PART = "cls4"
LESION = 1  # c1_mask_export.LESION_LABEL: 255 (uncertain) counts as NOT lesion


@dataclass
class Runtime:
    """Everything a stage needs from the outside world."""

    ids: dict[str, list[str]]
    labels_of: Callable[[str], np.ndarray]
    groups_of: Callable[[str], np.ndarray]
    codes_of: Callable[[str], np.ndarray]
    n_classes: int
    mel: int
    load_item: Callable[[str], tuple[np.ndarray, np.ndarray]]
    extract_fn: Callable[[np.ndarray, int], np.ndarray]
    resize_crop: Callable[[np.ndarray, int], np.ndarray]
    hair_fn: Callable[[np.ndarray], np.ndarray] | None
    fingerprint: dict


def make_load_item(image_index: dict, masks, positions: dict) -> Callable[[str], tuple[np.ndarray, np.ndarray]]:
    """``id -> (original RGB uint8, lesion bool mask)``; the mask grid must equal the image grid."""
    from PIL import Image  # noqa: PLC0415

    def load(image_id: str):
        image = np.asarray(Image.open(image_index[image_id]).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(masks[positions[image_id]]) == LESION
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mascara {mask.shape} e imagen {image.shape[:2]} no coinciden")
        return image, mask

    return load


def make_extract_fn(model, device, batch_size: int) -> Callable[[np.ndarray, int], np.ndarray]:
    """The B71 feature (cls4) with D4 TTA: ``(n_views, n, 3072)``, identity view first."""
    from dermssl import torch_v71 as tv  # noqa: PLC0415

    def extract(images: np.ndarray, n_views: int) -> np.ndarray:
        return tv.extract_split(model, images, device=device, batch_size=batch_size, n_views=n_views)[PART]

    return extract


def make_resize_crop() -> Callable[[np.ndarray, int], np.ndarray]:
    from PIL import Image  # noqa: PLC0415

    from dermssl import torch_v71 as tv  # noqa: PLC0415

    return lambda array, resolution: tv.resize_crop_uint8(Image.fromarray(array), resolution)


def make_hair_fn() -> Callable[[np.ndarray], np.ndarray] | None:
    """The repo's elongated-hair detector (needs cv2); ``None`` when it cannot be imported."""
    try:
        sys.path.insert(0, str(HAIR_DIR))
        import hair_removal  # noqa: PLC0415
        import cv2  # noqa: F401, PLC0415
    except ImportError:
        return None
    return lambda rgb: hair_removal.elongated_hair_mask(rgb).astype(bool)


def require_student_r3(source) -> None:
    """E3 is registered on the ``student_r3`` masks only (docs/33); the A2 arm of C1 is not allowed here."""
    if source.producer != REQUIRED_PRODUCER:
        raise SystemExit(f"ABORT: E3 usa solo mascaras de {REQUIRED_PRODUCER}, no de {source.producer!r}")


def _lesion_ids(ham_root: Path, ids: dict) -> dict:
    from c1_build_val_index import rebuild_split  # noqa: PLC0415

    _, meta = rebuild_split(ham_root.expanduser())
    lesion = meta.set_index("image_id")["lesion_id"]
    return {split: np.array([lesion[i] for i in members]) for split, members in ids.items()}


def build_runtime(args) -> Runtime:
    """Context from the C1 split and docs/28 holdout, masks verified against their manifest."""
    import torch  # noqa: PLC0415

    import v71_io  # noqa: PLC0415
    from dermssl import c1_mask_export as mx  # noqa: PLC0415
    from dermssl import c1_mask_provenance as mp  # noqa: PLC0415
    from dermssl import e3_selection as sel  # noqa: PLC0415
    from dermssl import run_provenance as rp  # noqa: PLC0415

    ctx = v71_io.load_context(args)
    source = mp.MaskSource.from_manifest(v71_io.read_json(args.masks_manifest.expanduser()))
    require_student_r3(source)
    mx.verify_artifact(args.masks.expanduser(), source.masks_sha256)
    masks = np.load(args.masks.expanduser(), mmap_mode="r")
    weights = v71_io.WeightBank(args.weights_dir.expanduser())
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = weights.model(weights.state(B71_WEIGHTS))
    ids = ctx["ids"]
    groups = _lesion_ids(args.ham_root, ids)
    hair_fn = make_hair_fn()
    return Runtime(
        ids=ids, labels_of=lambda s: v71_io.labels_of(ctx, ids[s]), groups_of=lambda s: groups[s],
        codes_of=lambda s: sel.tercile_codes(ids[s], args.tone_manifest.expanduser()),
        n_classes=len(ctx["classes"]), mel=ctx["mel"],
        load_item=make_load_item(ctx["image_index"], masks, {i: k for k, i in enumerate(source.image_ids)}),
        extract_fn=make_extract_fn(model, device, args.batch_size), resize_crop=make_resize_crop(),
        hair_fn=hair_fn,
        fingerprint={"protocol": PROTOCOL, "masks_sha256": source.masks_sha256,
                     "weights_sha256": rp.file_sha256(weights.final_path("B", "s42")),
                     "manifest_sha256": sel.MANIFEST_SHA256, "holdout_sha256": ctx["holdout_sha"],
                     "limit": args.limit, "hair_flag": hair_fn is not None})
