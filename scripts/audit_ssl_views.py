"""Export real SSL views and pseudo-mask overlays; no model or labels are loaded."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermssl.view_audit import index_images, load_mask, overlay, select_ids, sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image-root", "transformed-image-root", "pseudo-mask-root",
                 "exclude-ids-file", "recipe", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mask-encoding", choices=("binary255", "binary01", "pseudo01255"),
                        required=True)
    parser.add_argument("--count", type=int, default=100)
    return parser.parse_args()


def read_recipe(path):
    recipe = json.loads(path.read_text())
    required = {"seed", "resolution", "hue_jitter", "cc_probs", "tone_view"}
    if set(recipe) != required or not isinstance(recipe["tone_view"], bool):
        raise ValueError(f"Recipe must contain exactly {sorted(required)}")
    if recipe["resolution"] < 1 or recipe["seed"] < 0:
        raise ValueError("Invalid resolution or seed")
    weights = np.asarray([p for _, p in recipe["cc_probs"]], dtype=float)
    if not len(weights) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid cc_probs")
    return recipe


def make_dataset(ids, originals, transformed, recipe):
    from dermssl.pretrain.torch_data import SimSiamPairDataset
    return SimSiamPairDataset([(originals[i], transformed[i]) for i in ids],
                             resolution=recipe["resolution"], seed=recipe["seed"],
                             cc_probs=tuple(tuple(x) for x in recipe["cc_probs"]),
                             hue_jitter=recipe["hue_jitter"], tone_view=recipe["tone_view"])


def export_case(args, dataset, index, image_id, originals, transformed, masks):
    from dermssl.pretrain.torch_view_audit import audit_item, tensor_image
    with Image.open(originals[image_id]) as image:
        mask = load_mask(masks[image_id], image.size, args.mask_encoding)
        with Image.open(transformed[image_id]) as shifted:
            if image.size != shifted.size:
                raise ValueError(f"Original/transformed grids differ: {image_id}")
            delta = np.abs(np.asarray(image.convert("RGB"), dtype=float)
                           - np.asarray(shifted.convert("RGB"), dtype=float))
            raw_lesion_delta = float(delta[mask].mean()) if mask.any() else None
    views, records = audit_item(dataset, index, mask)
    case = args.out / image_id
    case.mkdir()
    resolution = views[0].shape[-1]
    sheet = Image.new("RGB", (resolution * len(views), resolution + 28), "white")
    names = ("v1_original", "v2_transformed", "tone_aligned")[:len(views)]
    exported = []
    for column, (name, tensor, trace) in enumerate(zip(names, views, records)):
        rgb, warped = tensor_image(tensor), trace.pop("mask")
        rgb.save(case / f"{name}.png")
        Image.fromarray(warped.astype(np.uint8) * 255).save(case / f"{name}_mask.png")
        visual = overlay(rgb, warped)
        visual.save(case / f"{name}_overlay.png")
        sheet.paste(visual, (resolution * column, 28))
        ImageDraw.Draw(sheet).text((resolution * column + 4, 5), name, fill="black")
        exported.append({"name": name, **trace})
    sheet.save(case / "views.png")
    return {"image_id": image_id, "dataset_index": index, "views": exported,
            "raw_lesion_mean_abs_rgb_delta_255": raw_lesion_delta,
            "inputs": {key: {"path": str(mapping[image_id]), "sha256": sha256(mapping[image_id])}
                       for key, mapping in (("original", originals), ("transformed", transformed),
                                            ("pseudo_mask", masks))},
            "outputs": {p.name: sha256(p) for p in sorted(case.iterdir())}}


def main():
    args = parse_args()
    if args.out.exists():
        raise ValueError("Output already exists; use a new run directory")
    recipe = read_recipe(args.recipe)
    originals = index_images(args.image_root)
    transformed = index_images(args.transformed_image_root)
    masks = index_images(args.pseudo_mask_root)
    excluded = set(args.exclude_ids_file.read_text().split())
    ids = select_ids(originals, transformed, masks, excluded, args.count, recipe["seed"])
    dataset = make_dataset(ids, originals, transformed, recipe)
    args.out.mkdir(parents=True)
    write_metadata(args, recipe, ids)
    for index, image_id in enumerate(ids):
        record = export_case(args, dataset, index, image_id, originals, transformed, masks)
        with (args.out / "cases.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    (args.out / "status.json").write_text(json.dumps({"status": "complete", "n": len(ids)}) + "\n")


def write_metadata(args, recipe, ids):
    import torch
    import torchvision
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = None
    root = Path(__file__).resolve().parents[1]
    report = {"status": "audit_only", "recipe": recipe, "ids": ids, "code_revision": revision,
              "mask_encoding": args.mask_encoding, "mask_source": "user_supplied_pseudo_masks",
              "sampling": "uniform_without_diagnoses; selected-list dataset indices",
              "rng": "CPU single-process; torch seed=recipe.seed+selected-list index",
              "reproduction_limit": "new deterministic audit draws, not historical worker draws",
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "torchvision": torchvision.__version__, "numpy": np.__version__},
              "exclude_sha256": sha256(args.exclude_ids_file), "recipe_sha256": sha256(args.recipe),
              "code_sha256": {p: sha256(root / p) for p in
                              ("src/dermssl/pretrain/torch_data.py",
                               "src/dermssl/pretrain/torch_view_audit.py", "src/dermssl/view_audit.py",
                               "scripts/audit_ssl_views.py")}}
    (args.out / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
