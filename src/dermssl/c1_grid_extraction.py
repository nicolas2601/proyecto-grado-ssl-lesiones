"""One pending C1 cell, from backbone to saved result.

The torch half (tensor conversion, the forward pass) is imported lazily from
``torch_c1_extraction`` so this module -- and everything that imports it --
stays importable without torch. The backbone builder is injected: it lives in
``scripts/cluster_frozen_bank.py``, which does import torch at module level.
"""

from __future__ import annotations

import numpy as np

from dermssl import c1_cell_gates as cg
from dermssl import c1_extraction as ex
from dermssl import c1_mask_provenance as mp
from dermssl.c1_grid_cells import fit_and_save_cell


def extract_cell(model, image_dir, image_ids, masks, mask_source, condition, device, tally):
    """Features for one (backbone, condition) over the given ids.

    The tally is filled here, where the per-image facts exist. Recomputing the
    degenerate counts afterwards from the features would be measuring a
    different thing: by then the fallback has already happened and looks like
    an ordinary row.
    """
    from PIL import Image  # noqa: PLC0415

    from dermssl import torch_c1_extraction as tex  # noqa: PLC0415

    rows = []
    for index, image_id in enumerate(image_ids):
        path = image_dir / f"{image_id}.jpg"
        if not path.exists():
            raise SystemExit(f"ABORT: falta la imagen {image_id} en {image_dir}")

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        tracked = mp.TrackedMask(
            array=masks[index].astype(bool), source=mask_source, image_id=image_id
        )
        prepared = ex.build_eval_input(image, tracked, condition)
        tensor = tex.to_normalized_tensor(prepared.image).to(device)
        feature, used_fallback = tex.extract_feature(
            model, tensor, condition, prepared.mask
        )
        tally.record(
            used_cls_fallback=bool(used_fallback),
            empty_after_geometry=prepared.empty_after_geometry,
            used_bbox_fallback=bool(prepared.used_bbox_fallback),
        )
        rows.append(feature)
    return np.stack(rows)


def run_pending_cell(store, inputs, backbone, condition, *, build_model, image_dir, device,
                     probe, provenance) -> dict:
    """Build the backbone, extract train and val, then fit, gate and save the cell."""
    arch, source = inputs.spec_of(backbone)
    print(f"\n=== {backbone} / {condition} ===", flush=True)
    built = build_model(arch, source, ex.RESOLUTION, device)
    model = built["model"] if isinstance(built, dict) else built

    train_tally = cg.CellTally(backbone=backbone, condition=condition)
    val_tally = cg.CellTally(backbone=backbone, condition=condition)
    train_features = extract_cell(
        model, image_dir, inputs.train_ids, inputs.train_masks,
        inputs.mask_source, condition, device, train_tally,
    )
    val_features = extract_cell(
        model, image_dir, inputs.val_ids, inputs.val_masks,
        inputs.mask_source, condition, device, val_tally,
    )
    return fit_and_save_cell(
        store, backbone, condition, train_features=train_features,
        train_labels=inputs.train_labels, val_features=val_features,
        val_labels=inputs.val_labels, train_tally=train_tally, val_tally=val_tally,
        probe=probe, n_classes=inputs.n_classes, provenance=provenance,
    )
