"""Real-data side of ``e3b_run.py``: split, labels, pinned ``student_r3`` masks, the frozen backbones.

Reuses ``e3_io`` for everything E3 already does (mask loader, lesion groups, hair detector,
``student_r3`` guard) so the masks and the geometry are the ones E3 used. Heavy imports are
lazy: ``e3b_run.py`` and its tests import this module without torch, timm or the images.
"""

from __future__ import annotations

import numpy as np

import e3_io
from dermssl.e3b.runtime import SPLITS, Runtime


def build_runtime(args) -> Runtime:
    """C1/v7 split (checked against its index), masks verified against their manifest, lazy backbones."""
    import torch  # noqa: PLC0415

    import v71_io  # noqa: PLC0415
    from dermssl import c1_mask_export as mx  # noqa: PLC0415
    from dermssl import c1_mask_provenance as mp  # noqa: PLC0415
    from dermssl import e3_selection as sel  # noqa: PLC0415
    from dermssl.e3b import backbones  # noqa: PLC0415

    ctx = v71_io.load_context(args)
    source = mp.MaskSource.from_manifest(v71_io.read_json(args.masks_manifest.expanduser()))
    e3_io.require_student_r3(source)
    mx.verify_artifact(args.masks.expanduser(), source.masks_sha256)
    masks = np.load(args.masks.expanduser(), mmap_mode="r")
    ids = {split: list(ctx["ids"][split]) for split in SPLITS}
    groups = e3_io._lesion_ids(args.ham_root, ids)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hair_fn = e3_io.make_hair_fn()
    positions = {image_id: k for k, image_id in enumerate(source.image_ids)}
    return Runtime(
        ids=ids, labels_of=lambda s: v71_io.labels_of(ctx, ids[s]), groups_of=lambda s: groups[s],
        classes=tuple(ctx["classes"]), mel=ctx["mel"],
        load_item=e3_io.make_load_item(ctx["image_index"], masks, positions), hair_fn=hair_fn,
        build_backbone=lambda name: backbones.build_backbone(name, device),
        fingerprint={"masks_sha256": source.masks_sha256, "holdout_sha256": ctx["holdout_sha"],
                     "manifest_sha256": sel.MANIFEST_SHA256, "hair_flag": hair_fn is not None})
