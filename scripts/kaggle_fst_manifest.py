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
import random
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
    "mask_confidence",
    "fallback_used",
    "confidence",
    "artifact_detected",
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
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"tiny run ({SMOKE_EPOCHS} epoch, <= {SMOKE_MAX_TARGET_IMAGES} "
        "target images) to validate wiring before a real run",
    )
    return parser.parse_args(argv)


def image_id(path: Path) -> str:
    return path.stem


def deterministic_split_indices(
    n_items: int, *, val_fraction: float = 0.20, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Return a deterministic, disjoint train/validation split."""
    if n_items < 2:
        raise ValueError("held-out validation requires at least two image/mask pairs")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("validation fraction must be strictly between 0 and 1")
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    n_val = min(n_items - 1, max(1, int(round(n_items * val_fraction))))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return train_indices, val_indices


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
    from torch.utils.data import DataLoader, Subset

    from dermssl import fitzpatrick, torch_unet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_ds = torch_unet.ISIC2018SegmentationDataset(
        args.unet_train_images, args.unet_train_masks
    )
    train_indices, val_indices = deterministic_split_indices(
        len(full_ds), val_fraction=args.validation_fraction, seed=args.seed
    )
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    epochs = SMOKE_EPOCHS if args.smoke else args.epochs
    training = torch_unet.train_unet(
        train_loader, val_loader, epochs=epochs, device=device, lr=args.lr
    )
    model = training.model
    model.eval()
    print(
        f"best U-Net checkpoint: epoch={training.best_epoch} "
        f"val_dice={training.best_val_dice:.5f} val_iou={training.best_val_iou:.5f}"
    )

    if args.checkpoint_out:
        torch_unet.save_checkpoint(training, args.checkpoint_out)

    target_paths = sorted(Path(args.target_image_root).glob("*.jpg"))
    if args.smoke:
        target_paths = target_paths[:SMOKE_MAX_TARGET_IMAGES]

    rows = []
    for path in target_paths:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        prediction = torch_unet.infer_mask_with_confidence(model, img, device=device)
        estimate = fitzpatrick.estimate_fst(
            img,
            mask=prediction.mask,
            mask_confidence=prediction.confidence,
        )
        rows.append(
            {
                "image_id": image_id(path),
                "ita": estimate.ita,
                "fst_bucket": (
                    estimate.bucket if estimate.bucket is not None else "unavailable"
                ),
                "skin_fraction": estimate.skin_fraction,
                "mask_source": "otsu_fallback" if estimate.fallback_used else "unet",
                "mask_confidence": prediction.confidence,
                "fallback_used": estimate.fallback_used,
                "confidence": estimate.confidence,
                "artifact_detected": estimate.artifact_detected,
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
