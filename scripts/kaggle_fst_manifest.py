"""Script GPU de Kaggle: entrena la U-Net una vez, infiere máscaras y emite
``fst_manifest.json``, la interfaz GPU/CPU para la evaluación de equidad de
Fitzpatrick (D4 en design.md).

Dos conjuntos, dos roles:
  - ISIC 2018 Task 1 (identificador Kaggle confirmado para la tarea 0.1:
    ``tschandl/isic2018-challenge-task1-data-segmentation``), pares de
    imagen/máscara que ENTRENAN la U-Net una vez (D3: la U-Net de ISIC-2018
    es la fuente principal; Otsu en ``fitzpatrick.estimate_fst`` es respaldo).
  - El pool objetivo FST (p. ej., HAM10000/ISIC2019, el mismo que evalúa
    docs/16): solo INFERENCIA, una fila de manifiesto por imagen.

La salida ``fst_manifest.json`` contiene estimaciones de imagen completa y
guiadas por máscara, un bucket ``unavailable`` explícito y procedencia de QA.
El CSV solo se solicita como exportación heredada; fuera de Kaggle no importa
torch.

    python scripts/kaggle_fst_manifest.py \\
        --unet-train-images /kaggle/input/isic2018.../train/images \\
        --unet-train-masks  /kaggle/input/isic2018.../train/masks \\
        --target-image-root /kaggle/input/skin-cancer-mnist-ham10000/images \\
        --output-json fst_manifest.json --epochs 20

Smoke (valida toda la integración en minutos: una época y pocas imágenes):
    python scripts/kaggle_fst_manifest.py ... --smoke

Este es un paso MANUAL: requiere una sesión GPU real y los dos conjuntos
anteriores; CI no lo ejecuta (ver tests/test_torch_unet_static.py y la fila
"Deferred (GPU)" de la estrategia de pruebas en design.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl import run_environment  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

# Orden de columnas de fst_manifest.csv (D4 en design.md: esquema GPU/CPU).
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

FST_SCHEMA_VERSION = "1.0"

SMOKE_EPOCHS = 1
SMOKE_MAX_TARGET_IMAGES = 8


def find_target_images(root) -> list[Path]:
    """Recursivo (mismo criterio que ``data_resolver.index_images`` y
    ``diagnostico_fst_comparativa.find_images``): HAM10000 reparte las
    imagenes en subcarpetas (part_1/part_2), un glob no recursivo sobre
    ``--target-image-root`` encuentra cero imagenes."""
    from dermssl import data_resolver

    found = data_resolver.index_images([Path(root)])
    return sorted(found.values())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--unet-train-images", required=True,
                          help="Imágenes de entrenamiento de ISIC 2018 Task 1 (jpg)")
    parser.add_argument("--unet-train-masks", required=True,
                          help="Máscaras de entrenamiento de ISIC 2018 Task 1 (png)")
    parser.add_argument("--target-image-root", required=True,
                          help="Pool objetivo FST sobre el que se ejecuta la inferencia (jpg)")
    parser.add_argument("--dataset-id", default="target_pool", help="Identificador del conjunto objetivo")
    parser.add_argument("--source-revision", default="unknown", help="Revisión de los datos de origen")
    parser.add_argument("--output-json", default="fst_manifest.json", help="Ruta del manifiesto JSON principal")
    parser.add_argument("--output-csv", default="fst_manifest.csv",
                         help="exportación CSV heredada (por defecto: fst_manifest.csv)")
    parser.add_argument("--mask-manifest-json", default="",
                         help="exportación independiente opcional de mask_manifest.json")
    parser.add_argument("--checkpoint-out", default="",
                          help="ruta opcional para guardar los pesos U-Net entrenados")
    parser.add_argument("--epochs", type=int, default=20, help="Épocas de entrenamiento de la U-Net")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    parser.add_argument("--num-workers", type=int, default=4, help="Procesos de carga de datos")
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Fracción reservada para validación")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Ejecución reducida ({SMOKE_EPOCHS} época, hasta {SMOKE_MAX_TARGET_IMAGES} "
        "imágenes objetivo) para validar el cableado antes de una corrida real",
    )
    return parser.parse_args(argv)


def image_id(path: Path) -> str:
    return path.stem


def deterministic_split_indices(
    n_items: int, *, val_fraction: float = 0.20, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Return a deterministic, disjoint train/validation split."""
    if n_items < 2:
        raise ValueError("la validación retenida requiere al menos dos pares imagen/máscara")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("la fracción de validación debe estar estrictamente entre 0 y 1")
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    n_val = min(n_items - 1, max(1, int(round(n_items * val_fraction))))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return train_indices, val_indices


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write the optional legacy CSV export."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _fst_payload(estimate) -> dict:
    return {
        "ita": estimate.ita,
        "bucket": estimate.bucket if estimate.bucket is not None else "unavailable",
        "status": estimate.status,
        "skin_fraction": estimate.skin_fraction,
        "fallback_used": estimate.fallback_used,
        "confidence": estimate.confidence,
        "artifact_detected": estimate.artifact_detected,
    }


def build_fst_manifest_record(
    *,
    dataset: str,
    path: Path,
    source_revision: str,
    whole,
    masked,
    mask_record: dict,
) -> dict:
    """Build one traceable JSON-first FST record with both estimators."""
    masked_payload = _fst_payload(masked)
    masked_payload["mask"] = {
        key: mask_record.get(key)
        for key in (
            "producer",
            "version",
            "mask_sha256",
            "mask_pixel_count",
            "mask_semantics",
            "area_fraction",
            "quality",
            "eligible",
            "fallback_used",
            "quarantine_reason",
        )
    }
    return {
        "dataset": dataset,
        "image_id": image_id(path),
        "source_revision": source_revision,
        "whole_image": _fst_payload(whole),
        "masked": masked_payload,
    }


def build_fst_manifest(records: list[dict]) -> dict:
    """Wrap FST records in the JSON-primary manifest contract."""
    unavailable = sum(record["masked"]["bucket"] == "unavailable" for record in records)
    return {
        "schema_version": FST_SCHEMA_VERSION,
        "records": records,
        "summary": {"total": len(records), "masked_unavailable": unavailable},
    }


def write_fst_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """The torch half - imported lazily so ``parse_args`` alone never
    touches torch (mirrors ``scripts/pretrain.py``'s validate-before-import
    order)."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Subset

    from dermssl import fitzpatrick, mask_manifest, torch_unet

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
        f"mejor punto de control de U-Net: época={training.best_epoch} "
        f"dice_validación={training.best_val_dice:.5f} iou_validación={training.best_val_iou:.5f}"
    )

    if args.checkpoint_out:
        torch_unet.save_checkpoint(training, args.checkpoint_out)

    target_paths = find_target_images(args.target_image_root)
    if args.smoke:
        target_paths = target_paths[:SMOKE_MAX_TARGET_IMAGES]

    rows = []
    fst_records = []
    mask_records = []
    for path in target_paths:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        prediction = torch_unet.infer_mask_with_confidence(model, img, device=device)
        mask_record = mask_manifest.build_mask_record(
            dataset=args.dataset_id,
            image_id=image_id(path),
            source_revision=args.source_revision,
            mask=prediction.mask,
            producer="unet",
            version=Path(args.checkpoint_out).name if args.checkpoint_out else "in_memory_unet",
            confidence=prediction.confidence,
        )
        whole = fitzpatrick.estimate_fst_whole_image(img)
        estimate = fitzpatrick.estimate_fst_from_mask_record(
            img, mask=prediction.mask, mask_record=mask_record
        )
        mask_records.append(mask_record)
        fst_records.append(
            build_fst_manifest_record(
                dataset=args.dataset_id,
                path=path,
                source_revision=args.source_revision,
                whole=whole,
                masked=estimate,
                mask_record=mask_record,
            )
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

    fst_manifest = {**build_fst_manifest(fst_records), "environment": run_environment.snapshot(args)}
    write_fst_manifest(fst_manifest, Path(args.output_json))
    if args.output_csv:
        write_manifest(rows, Path(args.output_csv))
        print(f"se escribieron {len(rows)} filas CSV heredadas en {args.output_csv}")
    if args.mask_manifest_json:
        mask_manifest.write_mask_manifest(
            mask_manifest.build_mask_manifest(mask_records), Path(args.mask_manifest_json)
        )
        print(f"se escribieron {len(mask_records)} registros de máscara en {args.mask_manifest_json}")
    print(f"se escribieron {len(fst_records)} registros JSON FST en {args.output_json}")
    return len(rows)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
