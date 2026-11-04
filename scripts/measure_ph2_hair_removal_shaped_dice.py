"""C3(c): Dice of the pinned teacher 026 recipe on PH2, shaped=False vs
shaped=True hair removal.

Same pinned checkpoint/loader/recipe as ~/work/ita-ladder-20261104/arm3_isic.py
(maestro's reference), adapted to PH2 (local images/masks directory with
expert masks, not a zip) and run twice per image - once with the production
default (shaped=False) and once with the unused elongation filter
(shaped=True) - to isolate the effect of C3(a)/(b)'s finding (hair_mask
overlaps the expert lesion, median 0.5171 of its area) on actual Dice.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

AUDIT = Path("/home/jovyan/work/dermlip_slic_v8/audits/paired-raw-morph-20261027-001")
CHECKPOINT = Path(
    "/home/jovyan/work/dermlip_slic_v8/runs/"
    "dermlip-semi-supervised-v8-isic2018-2594-abl026-resnet34-ph2cp-ima0-s42/"
    "selection/best_teacher_model.pt"
)
CHECKPOINT_SHA256 = "f4fcee05cd4e4f884d31bb6a9bc08fc7225817d2529c8fa22eb1b2ab261462f4"
THRESHOLD = 0.5
PH2 = Path("/home/jovyan/work/dermlip_slic_v8/data/ph2")
OUT = Path("/home/jovyan/work/hair-removal-shaped-20261104/ph2_shaped_dice_rows.jsonl")
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"

sys.path.insert(0, str(AUDIT / "src"))
from dermssl.segmentation_methods.common.hair import remove_hair  # noqa: E402


def load_pinned_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_pinned_eval_026", AUDIT / "scripts/evaluate_dermlip_semi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(model, img_pil, device, shaped):
    import torch

    orig_w, orig_h = img_pil.size
    arr_u8, hair_fraction = remove_hair(
        np.asarray(img_pil.resize((224, 224), Image.Resampling.BILINEAR)), shaped=shaped
    )
    img_t = torch.from_numpy(
        np.transpose(arr_u8.astype(np.float32) / 255.0, (2, 0, 1))
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_224 = torch.sigmoid(model(img_t)).squeeze().cpu().numpy()
    # Round trip through uint8 PNG-equivalent, pinned: quantizes before the
    # resize back to native size, same as arm3_isic.py. Removing this step
    # moves the binarization boundary.
    prob_pil = Image.fromarray((prob_224 * 255).astype(np.uint8), mode="L")
    prob = np.array(
        prob_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    mask = prob >= THRESHOLD
    return mask, float(hair_fraction)


def dice(a, b):
    denom = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / denom) if denom else float("nan")


def main() -> int:
    model = load_pinned_evaluator().load_model(CHECKPOINT, DEVICE, "teacher")
    manifest = json.loads((PH2 / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    print(f"PH2: {len(records)} imagenes | device {DEVICE}", flush=True)

    with OUT.open("w") as fh:
        for i, record in enumerate(records, 1):
            img_path = PH2 / record["image_path"]
            mask_path = PH2 / record["mask_path"]
            img_pil = Image.open(img_path).convert("RGB")
            gt = np.asarray(Image.open(mask_path).convert("L")) > 127

            mask_unshaped, hair_frac_unshaped = predict(model, img_pil, DEVICE, shaped=False)
            mask_shaped, hair_frac_shaped = predict(model, img_pil, DEVICE, shaped=True)

            fh.write(json.dumps({
                "image_id": record["image_id"],
                "dice_unshaped": dice(mask_unshaped, gt),
                "dice_shaped": dice(mask_shaped, gt),
                "hair_fraction_unshaped": hair_frac_unshaped,
                "hair_fraction_shaped": hair_frac_shaped,
                "pred_area_fraction_unshaped": float(mask_unshaped.mean()),
                "pred_area_fraction_shaped": float(mask_shaped.mean()),
                "gt_area_fraction": float(gt.mean()),
            }) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(records)}", flush=True)

    print(f"escrito: {OUT}", flush=True)
    print(f"checkpoint_sha256_used={CHECKPOINT_SHA256}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
