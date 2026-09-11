"""Script Kaggle con GPU: entrena el U-Net una vez, corre inferencia de
mascaras, y emite ``fst_manifest.csv`` - la costura entre GPU y CPU para la
evaluacion de fairness por fototipo Fitzpatrick.

Dos datasets, dos roles:
  - ISIC 2018 Task 1 (``tschandl/isic2018-challenge-task1-data-segmentation``)
    - pares imagen/mascara, ENTRENA el U-Net una sola vez (el U-Net de
    ISIC-2018 es la fuente primaria de mascara, el fallback Otsu dentro de
    ``fitzpatrick.estimate_fst`` se mantiene como fallback).
  - El pool objetivo de FST (por ejemplo HAM10000/ISIC2019, el mismo pool
    que evalua docs/16) - solo INFERENCIA, una fila de manifiesto por imagen.

Columnas de salida en ``fst_manifest.csv``: ``image_id, ita, fst_bucket,
skin_fraction, mask_source, status``. Todo lo que viene despues
(``metrics.stratified_summary``, ``gates.fairness_gate``, el bloque
``FST_MANIFEST_PATH`` de ``scripts/notebook_code_v5.py``) lee este CSV;
nunca se importa torch fuera de Kaggle.

    python scripts/kaggle_fst_manifest.py \\
        --unet-train-images /kaggle/input/isic2018.../train/images \\
        --unet-train-masks  /kaggle/input/isic2018.../train/masks \\
        --target-image-root /kaggle/input/skin-cancer-mnist-ham10000/images \\
        --output-csv fst_manifest.csv --epochs 20

Smoke (valida todo el cableado en minutos, un epoch, pocas imagenes
objetivo):
    python scripts/kaggle_fst_manifest.py ... --smoke

Este es un paso MANUAL - necesita una sesion real de GPU y los dos datasets
de arriba, y no corre en CI (ver tests/test_torch_unet_static.py, que solo
verifica que el codigo compila y su AST).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Orden de columnas de fst_manifest.csv (el esquema de la costura GPU/CPU).
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
                         help="imagenes de entrenamiento de ISIC 2018 Task 1 (jpg)")
    parser.add_argument("--unet-train-masks", required=True,
                         help="mascaras de entrenamiento de ISIC 2018 Task 1 (png)")
    parser.add_argument("--target-image-root", required=True,
                         help="pool objetivo de FST sobre el que correr inferencia (jpg)")
    parser.add_argument("--output-csv", default="fst_manifest.csv")
    parser.add_argument("--checkpoint-out", default="",
                         help="ruta opcional para guardar los pesos del U-Net entrenado")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="semilla para torch_unet.train_unet y el shuffling del DataLoader, "
        "asi correr este script de nuevo sobre los mismos datos reproduce el "
        "mismo U-Net y el mismo fst_manifest.csv (fairness_gate lee este archivo)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"corrida chica ({SMOKE_EPOCHS} epoch, <= {SMOKE_MAX_TARGET_IMAGES} "
        "imagenes objetivo) para validar el cableado antes de una corrida real",
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
    """La mitad que usa torch - se importa de forma perezosa para que
    ``parse_args`` sola nunca toque torch (mismo orden de validar-antes-de-
    importar que ``scripts/pretrain.py``)."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    from dermssl import fitzpatrick, torch_unet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = torch_unet.ISIC2018SegmentationDataset(
        args.unet_train_images, args.unet_train_masks
    )
    # Con semilla para que el shuffling de batches sea reproducible entre
    # corridas sobre los mismos datos - fst_manifest.csv alimenta
    # gates.fairness_gate, una decision PASS/FAIL que no puede cambiar en
    # silencio de una corrida a otra.
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=shuffle_generator,
    )

    epochs = SMOKE_EPOCHS if args.smoke else args.epochs
    model = torch_unet.train_unet(
        train_loader, epochs=epochs, device=device, lr=args.lr, seed=args.seed
    )
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

        # Mismo chequeo de area degenerada que estimate_fst aplica
        # internamente - se llama aca (no se re-deriva) solo para ETIQUETAR
        # que fuente de mascara va a usar realmente, no para cambiar la
        # decision. is_degenerate_mask es la unica fuente de verdad para
        # los limites y operadores.
        mask_source = (
            "otsu_fallback" if fitzpatrick.is_degenerate_mask(mask) else "unet"
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
