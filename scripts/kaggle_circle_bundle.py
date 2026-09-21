"""Arma el paquete NPZ de caracteristicas de entrada para
``scripts/kaggle_circle_stage2.py`` a partir de un ``fst_manifest.json`` real
(Fase 2), la particion congelada de ``pool.json`` (Fase 1) y un backbone
congelado de Fase 3 (DINOv2 o ResNet50).

``mask_manifest.json`` no persiste la mascara en si, solo su ``mask_sha256``
(ver ``dermssl.mask_manifest.build_mask_record``) - por eso este script vuelve
a correr el U-Net para recuperar la mascara real de cada imagen y verifica el
hash antes de usarla, exactamente lo que
``dermssl.circle.shift_skin_tone_from_mask_record`` ya exige.

Las imagenes con mascara en cuarentena (``masked.mask.eligible=False``) se
excluyen del paquete: no existe una version desplazada valida sin mascara de
lesion protegida. La cuenta de cuantas se excluyeron, por bucket FST de la
imagen completa, se reporta explicitamente en la salida - si la cuarentena se
concentra en un tono de piel, es un limite real del resultado y tiene que
quedar documentado, no perdido en el descarte (ver docs/21).

Manual, no CI: requiere una sesion GPU real con el U-Net y el backbone ya
entrenados (ver docs/20, docs/21). Smoke valida el cableado en minutos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl import circle  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

REQUIRED_BUNDLE_KEYS = (
    "x_train_original",
    "x_train_shifted",
    "y_train",
    "x_val",
    "y_val",
    "fst_buckets_val",
    "backbone_frozen",
)

BUNDLE_SPLITS = ("train", "validation")

SMOKE_MAX_IMAGES = 32


def is_bundle_split(split: str) -> bool:
    """El split ``test`` (Fase 1) queda congelado - nunca entra a esta fase
    experimental (design.md: "Validacion y prueba se congelan antes de
    ajustar umbrales o checkpoints")."""
    return split in BUNDLE_SPLITS


def partition_by_eligibility(records: list[dict]) -> tuple[list[dict], dict]:
    """Separa los registros con mascara elegible de los cuarentenados.

    Devuelve ``(elegibles, conteo_cuarentena_por_bucket)`` donde el conteo
    usa ``whole_image.bucket`` - el unico bucket FST que existe tambien para
    las imagenes excluidas, ya que no dependen de una mascara."""
    kept: list[dict] = []
    quarantine_counts: dict = {}
    for record in records:
        if record["masked"]["mask"].get("eligible"):
            kept.append(record)
        else:
            bucket = record["whole_image"]["bucket"]
            quarantine_counts[bucket] = quarantine_counts.get(bucket, 0) + 1
    return kept, quarantine_counts


def sample_target_bucket(
    source_bucket: int,
    rng: np.random.RandomState,
    *,
    max_shift: int = circle.MAX_FST_SHIFT,
) -> int:
    """Sortea un bucket FST destino distinto de ``source_bucket`` dentro de
    ``max_shift`` niveles, siguiendo a Pakzad et al.: "we randomly select
    another skin type s_j != s_i" - nunca un shift fijo para todas las
    imagenes."""
    candidates = sorted(
        {
            circle.target_fst_bucket(source_bucket, shift, max_shift=max_shift)
            for shift in range(-max_shift, max_shift + 1)
            if shift != 0
        }
        - {source_bucket}
    )
    if not candidates:
        raise ValueError(
            f"no hay bucket FST distinto de {source_bucket} dentro de max_shift={max_shift}"
        )
    return int(rng.choice(candidates))


def build_bundle_report(
    *,
    backbone: str,
    split_hash: str,
    n_train: int,
    n_val: int,
    quarantine_counts: dict,
    seed: int,
) -> dict:
    """Resumen JSON del paquete: procedencia del split y cuarentena por
    bucket FST, para que un sesgo de cuarentena por tono de piel quede
    documentado en vez de escondido en un descarte silencioso."""
    return {
        "backbone": backbone,
        "split_hash": split_hash,
        "n_train": n_train,
        "n_val": n_val,
        "quarantine_by_whole_image_bucket": {
            str(bucket): count for bucket, count in quarantine_counts.items()
        },
        "quarantine_total": sum(quarantine_counts.values()),
        "seed": seed,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--fst-manifest", required=True, help="fst_manifest.json real (Fase 2)")
    parser.add_argument("--pool-manifest", required=True, help="pool.json con split congelado (Fase 1)")
    parser.add_argument("--unet-checkpoint", required=True, help="Checkpoint del U-Net (Fase 2)")
    parser.add_argument("--backbone", required=True, choices=("dino", "resnet50"),
                         help="Backbone congelado a usar para extraer caracteristicas")
    parser.add_argument("--backbone-checkpoint", default="",
                         help="Checkpoint del backbone SSL continuo (obligatorio solo con --backbone dino; "
                              "resnet50 usa pesos ImageNet estandar, D3 en design.md)")
    parser.add_argument("--backbone-arch", default="dinov2_vits14",
                         help="Nombre de arquitectura de torch.hub facebookresearch/dinov2 "
                              "(solo aplica con --backbone dino; default = DEFAULT_BACKBONES"
                              "['dino_continual'] en dermssl/pretrain/config.py, cambiar solo "
                              "si scripts/pretrain.py corrio con --backbone distinto)")
    parser.add_argument("--image-root", required=True, help="Raiz de imagenes del pool objetivo")
    parser.add_argument("--ham-metadata", required=True, help="HAM10000_metadata.csv (etiquetas reales de diagnostico)")
    parser.add_argument("--output-npz", default="circle_features.npz", help="Ruta del paquete NPZ de salida")
    parser.add_argument("--output-json", default="circle_bundle_report.json", help="Ruta del resumen JSON")
    parser.add_argument("--max-shift", type=int, default=circle.MAX_FST_SHIFT, help="Maximo desplazamiento FST sintetico")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Ejecucion reducida (hasta {SMOKE_MAX_IMAGES} imagenes) para validar el cableado",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for path_arg in ("fst_manifest", "pool_manifest", "unet_checkpoint", "ham_metadata"):
        if not Path(getattr(args, path_arg)).exists():
            raise ValueError(f"no se encontro {path_arg.replace('_', '-')}: {getattr(args, path_arg)}")
    if not Path(args.image_root).is_dir():
        raise ValueError(f"no se encontro --image-root: {args.image_root}")
    if args.max_shift <= 0:
        raise ValueError("max-shift debe ser positivo")
    if args.backbone == "dino":
        if not args.backbone_checkpoint or not Path(args.backbone_checkpoint).exists():
            raise ValueError("--backbone dino requiere --backbone-checkpoint existente")
        if not args.backbone_arch:
            raise ValueError("--backbone dino requiere --backbone-arch (tag timm usado en scripts/pretrain.py)")


def _load_ham_labels(ham_metadata_path: Path) -> dict[str, int]:
    """Etiquetas reales de diagnostico por ``image_id``. ``fst_manifest.json``
    solo trae ITA/bucket, nunca diagnostico - mismo parseo que
    ``dermssl.finetune.torch_run._load_ham``, sin acoplarse a ``FinetuneConfig``."""
    import pandas as pd

    meta = pd.read_csv(ham_metadata_path)
    classes = sorted(meta.dx.unique())
    cls2idx = {c: i for i, c in enumerate(classes)}
    return {str(row.image_id): cls2idx[row.dx] for row in meta.itertuples()}


def run(args: argparse.Namespace) -> dict:
    """La mitad con torch - importada perezosamente para que ``parse_args``
    nunca toque torch ni una sesion GPU (mismo orden que
    ``scripts/kaggle_fst_manifest.py``)."""
    import torch
    from PIL import Image

    from dermssl import data_resolver, mask_manifest, torch_unet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    fst_manifest = json.loads(Path(args.fst_manifest).read_text(encoding="utf-8"))
    pool_manifest = json.loads(Path(args.pool_manifest).read_text(encoding="utf-8"))
    split_by_image_id = {row["image_id"]: row["split"] for row in pool_manifest["images"]}
    labels_by_image_id = _load_ham_labels(Path(args.ham_metadata))
    image_paths = data_resolver.index_images([Path(args.image_root)])

    bundle_records = [
        record
        for record in fst_manifest["records"]
        if is_bundle_split(split_by_image_id.get(record["image_id"]))
        and record["image_id"] in labels_by_image_id
        and record["image_id"] in image_paths
    ]
    if args.smoke:
        bundle_records = bundle_records[:SMOKE_MAX_IMAGES]

    eligible_records, quarantine_counts = partition_by_eligibility(bundle_records)

    unet = torch_unet.load_checkpoint(args.unet_checkpoint, device)
    backbone = _load_frozen_backbone(args, device)

    def extract(image_id: str) -> tuple[np.ndarray, np.ndarray]:
        img = np.asarray(
            Image.open(image_paths[image_id]).convert("RGB"), dtype=np.float32
        ) / 255.0
        prediction = torch_unet.infer_mask_with_confidence(unet, img, device=device)
        mask_record = mask_manifest.build_mask_record(
            dataset="target_pool",
            image_id=image_id,
            source_revision="unknown",
            mask=prediction.mask,
            producer="unet",
            version=Path(args.unet_checkpoint).name,
            confidence=prediction.confidence,
        )
        source_bucket = None
        for record in eligible_records:
            if record["image_id"] == image_id:
                source_bucket = record["masked"]["bucket"]
                break
        target_bucket = sample_target_bucket(source_bucket, rng, max_shift=args.max_shift)
        shifted = circle.shift_skin_tone_from_mask_record(
            img, prediction.mask, mask_record, target_bucket - source_bucket
        )

        def features(image_array: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                return backbone(tensor).cpu().numpy()[0]

        return features(img), features(shifted)

    train_ids = [
        r["image_id"] for r in eligible_records
        if split_by_image_id[r["image_id"]] == "train"
    ]
    val_ids = [
        r["image_id"] for r in eligible_records
        if split_by_image_id[r["image_id"]] == "validation"
    ]

    x_train_original, x_train_shifted = [], []
    for image_id in train_ids:
        original, shifted = extract(image_id)
        x_train_original.append(original)
        x_train_shifted.append(shifted)

    x_val = []
    fst_buckets_val = []
    for image_id in val_ids:
        original, _ = extract(image_id)
        x_val.append(original)
        record = next(r for r in eligible_records if r["image_id"] == image_id)
        fst_buckets_val.append(record["masked"]["bucket"])

    payload = {
        "x_train_original": np.asarray(x_train_original, dtype=np.float32),
        "x_train_shifted": np.asarray(x_train_shifted, dtype=np.float32),
        "y_train": np.asarray([labels_by_image_id[i] for i in train_ids], dtype=np.int64),
        "x_val": np.asarray(x_val, dtype=np.float32),
        "y_val": np.asarray([labels_by_image_id[i] for i in val_ids], dtype=np.int64),
        "fst_buckets_val": np.asarray(fst_buckets_val, dtype=object),
        "backbone_frozen": True,
    }
    assert set(payload) == set(REQUIRED_BUNDLE_KEYS), (
        "el paquete debe declarar exactamente las claves que kaggle_circle_stage2.py exige"
    )
    np.savez(args.output_npz, **payload)

    report = build_bundle_report(
        backbone=args.backbone,
        split_hash=pool_manifest.get("split_hash", "unknown"),
        n_train=len(train_ids),
        n_val=len(val_ids),
        quarantine_counts=quarantine_counts,
        seed=args.seed,
    )
    report["smoke"] = args.smoke
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"se escribio {args.output_npz} | train={len(train_ids)} val={len(val_ids)} "
          f"cuarentena={report['quarantine_total']}")
    return report


def _load_frozen_backbone(args, device):
    """Devuelve el modelo en modo evaluacion, sin gradientes. ResNet50 usa
    pesos ImageNet estandar (D3: control separado, sin continuar su
    preentrenamiento). DINOv2 se construye igual que
    ``pretrain/torch_dino.py::DinoContinual`` - ``torch.hub.load(
    "facebookresearch/dinov2", backbone_name, pretrained=False)``, NO timm -
    y carga el ``state_dict`` plano de solo el backbone que
    ``pretrain/torch_loop.py::_export_backbone`` exporta. Solo se llama desde
    ``run()``, nunca antes de validar argumentos."""
    import torch

    if args.backbone == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
    else:
        model = torch.hub.load("facebookresearch/dinov2", args.backbone_arch, pretrained=False)
        state = torch.load(args.backbone_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)

    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
