"""Kaggle GPU script: train the U-Net once, run mask inference, emit
``fst_manifest.csv`` - the GPU/CPU seam for Fitzpatrick fairness evaluation
(D4 in design.md).

Two datasets, two roles:
  - ISIC 2018 Task 1 (Kaggle handle confirmed for task 0.1:
    ``tschandl/isic2018-challenge-task1-data-segmentation``) - image/mask
    pairs, TRAINS the U-Net once (D3: the ISIC-2018 U-Net is the primary
    mask source, the Otsu fallback inside ``fitzpatrick.estimate_fst``
    stays the fallback).
  - The FST target pool (e.g. HAM10000/ISIC2019, the same pool docs/16
    evaluates) - INFERENCE only, one manifest row per image.

Output ``fst_manifest.csv`` columns: ``image_id, ita, fst_bucket,
skin_fraction, mask_source, status`` (D4's exact schema). Everything
downstream (``metrics.stratified_summary``, ``gates.fairness_gate``,
``scripts/notebook_code_v5.py``'s ``FST_MANIFEST_PATH`` block) reads this
CSV; no torch import ever happens off Kaggle.

    python scripts/kaggle_fst_manifest.py \\
        --unet-train-images /kaggle/input/isic2018.../train/images \\
        --unet-train-masks  /kaggle/input/isic2018.../train/masks \\
        --target-image-root /kaggle/input/skin-cancer-mnist-ham10000/images \\
        --output-csv fst_manifest.csv --epochs 20

Smoke (validates the full wiring in minutes, one epoch, a handful of target
images):
    python scripts/kaggle_fst_manifest.py ... --smoke

This is a MANUAL step - it needs a real GPU session and the two datasets
above and is not run by CI (see tests/test_torch_unet_static.py and
design.md's Testing Strategy "Deferred (GPU)" row).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# fst_manifest.csv column order (D4 in design.md - the GPU/CPU seam schema).
MANIFEST_COLUMNS = (
    "image_id",
    "ita",
    "fst_bucket",
    "skin_fraction",
    "mask_source",
    "status",
)

SMOKE_EPOCHS = 1
SMOKE_MAX_TARGET_IMAGES = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--unet-train-images", required=True,
                         help="ISIC 2018 Task 1 training images (jpg)")
    parser.add_argument("--unet-train-masks", required=True,
                         help="ISIC 2018 Task 1 training masks (png)")
    parser.add_argument("--target-image-root", required=True,
                         help="FST target pool to run inference on (jpg)")
    parser.add_argument("--output-csv", default="fst_manifest.csv")
    parser.add_argument("--checkpoint-out", default="",
                         help="optional path to save the trained U-Net weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"tiny run ({SMOKE_EPOCHS} epoch, <= {SMOKE_MAX_TARGET_IMAGES} "
        "target images) to validate wiring before a real run",
    )
    return parser.parse_args(argv)


def image_id(path: Path) -> str:
    return path.stem


def write_manifest(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    """The torch half - imported lazily so ``parse_args`` alone never
    touches torch (mirrors ``scripts/pretrain.py``'s validate-before-import
    order)."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    from dermssl import fitzpatrick, torch_unet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = torch_unet.ISIC2018SegmentationDataset(
        args.unet_train_images, args.unet_train_masks
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    epochs = SMOKE_EPOCHS if args.smoke else args.epochs
    model = torch_unet.train_unet(train_loader, epochs=epochs, device=device, lr=args.lr)
    model.eval()

    if args.checkpoint_out:
        torch_unet.save_checkpoint(model, args.checkpoint_out)

    target_paths = sorted(Path(args.target_image_root).glob("*.jpg"))
    if args.smoke:
        target_paths = target_paths[:SMOKE_MAX_TARGET_IMAGES]

    rows = []
    for path in target_paths:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        mask = torch_unet.infer_mask(model, img, device=device)

        # Same degenerate-area check estimate_fst applies internally (D3) -
        # recomputed here only to LABEL which mask source it will actually
        # use, not to change the decision.
        area = float(mask.mean())
        mask_source = (
            "unet"
            if fitzpatrick.MIN_MASK_AREA_FRACTION <= area <= fitzpatrick.MAX_MASK_AREA_FRACTION
            else "otsu_fallback"
        )

        estimate = fitzpatrick.estimate_fst(img, mask=mask)
        rows.append(
            {
                "image_id": image_id(path),
                "ita": estimate.ita,
                "fst_bucket": estimate.bucket if estimate.bucket is not None else "",
                "skin_fraction": estimate.skin_fraction,
                "mask_source": mask_source,
                "status": estimate.status,
            }
        )

    write_manifest(rows, Path(args.output_csv))
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    return len(rows)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
