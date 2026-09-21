"""Ejecuta un preentrenamiento SSL v6 sin etiquetas en GPU.

Una GPU:
    python scripts/pretrain.py --method dino_continual \\
        --image-root /scratch/derm/pool --output-dir /scratch/derm/runs/dino \\
        --split-hash <REEMPLAZAR_CON_SHA256_DE_64_HEX> --budget-gpu-hours 8 \\
        --tracking-uri file:/scratch/derm/mlruns

Varias GPU (un nodo, 4 GPU):
    torchrun --nproc_per_node=4 scripts/pretrain.py --method dino_continual ...

Smoke (valida el cableado completo en minutos y sigue sin usar etiquetas):
    python scripts/pretrain.py --method simsiam --smoke \\
        --image-root /path/to/a/few/images --output-dir /tmp/smoke \\
        --split-hash <REEMPLAZAR_CON_SHA256_DE_64_HEX>

Smoke con transformaciones CIRCLe ya materializadas (usa pares con la misma
ruta relativa y registra la pérdida de consistencia en ``run.json``):
    python scripts/pretrain.py --method simsiam --smoke \\
        --image-root /path/to/original --transformed-image-root /path/to/transformed \\
        --transformed-manifest-path /path/to/transformed/tone_transform_manifest.json \\
        --data-revision pool-v1 --transformed-data-revision <REVISION_DEL_MANIFIESTO> \\
        --output-dir /tmp/smoke --split-hash <REEMPLAZAR_CON_SHA256_DE_64_HEX> \\
        --batch-size-per-gpu 2

La interfaz de MLflow sobre las ejecuciones registradas:
    mlflow ui --backend-store-uri file:/scratch/derm/mlruns
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl.pretrain import config as config_mod  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

SMOKE_BUDGET_GPU_HOURS = 0.05  # 3 minutos de reloj en 1 GPU
SMOKE_MAX_STEPS = 1
LEGACY_SPLIT_HASH = "0" * 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--method", required=True, choices=config_mod.METHODS, help="Método SSL que se ejecutará")
    parser.add_argument("--image-root", required=True, help="Carpeta raíz de imágenes sin etiquetas")
    parser.add_argument("--output-dir", required=True, help="Carpeta donde se guardan los artefactos")
    parser.add_argument(
        "--split-hash",
        default="",
        help="SHA-256 de la partición congelada; obligatorio para adaptación transformada",
    )
    parser.add_argument("--backbone", default="", help="Arquitectura base; se resuelve desde el método si se omite")
    parser.add_argument("--initial-weights", default="", help="Pesos iniciales; se resuelven desde el método si se omiten")
    parser.add_argument("--resume-checkpoint", default="", help="Punto de control desde el cual reanudar")
    parser.add_argument("--data-revision", default="", help="Revisión de los datos de entrada")
    parser.add_argument("--transformed-image-root", default="", help="Carpeta de imágenes transformadas para adaptación pareada")
    parser.add_argument("--transformed-manifest-path", default="", help="Ruta al manifiesto auditable de las imágenes transformadas")
    parser.add_argument("--transformed-data-revision", default="", help="Revisión de las imágenes transformadas")
    parser.add_argument("--consistency-weight", type=float, default=1.0, help="Peso de la consistencia entre imágenes original y transformada")
    parser.add_argument("--max-steps", type=int, default=0, help="Máximo de pasos; cero mantiene el presupuesto por GPU")
    parser.add_argument("--resolution", type=int, default=224, help="Resolución global de entrada")
    parser.add_argument("--batch-size-per-gpu", type=int, default=64, help="Tamaño de lote por GPU")
    parser.add_argument("--budget-gpu-hours", type=float, default=8.0, help="Presupuesto máximo en horas-GPU")
    parser.add_argument("--num-workers", type=int, default=8, help="Procesos de carga de datos")
    parser.add_argument("--precision", default="bf16", choices=config_mod.PRECISIONS, help="Precisión numérica")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    parser.add_argument("--mlflow-experiment", default="obj2-ssl-pretraining", help="Nombre del experimento de MLflow")
    parser.add_argument("--tracking-uri", default="", help="URI del backend de seguimiento de MLflow")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""), help="Proyecto de Weights & Biases (vacío = solo MLflow)")
    parser.add_argument("--wandb-group", default="", help="Grupo de la corrida en Weights & Biases")
    parser.add_argument(
        "--no-lesion-guided",
        action="store_true",
        help="Ablación: crops locales uniformes en vez de guiados por el borde",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Presupuesto reducido ({SMOKE_BUDGET_GPU_HOURS} horas-GPU) para validar el cableado",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> config_mod.PretrainConfig:
    """Construye la configuración validable, incluida la modalidad smoke."""
    if args.transformed_image_root and not args.split_hash:
        raise ValueError("split_hash es obligatorio para adaptación transformada")
    return config_mod.PretrainConfig(
        method=args.method,
        image_root=args.image_root,
        output_dir=args.output_dir,
        split_hash=args.split_hash or LEGACY_SPLIT_HASH,
        backbone=args.backbone,
        initial_weights=args.initial_weights,
        resume_checkpoint=args.resume_checkpoint,
        data_revision=args.data_revision,
        transformed_image_root=args.transformed_image_root,
        transformed_manifest_path=args.transformed_manifest_path,
        transformed_data_revision=args.transformed_data_revision,
        adaptation_mode=(
            "original_transformed_consistency"
            if args.transformed_image_root
            else "two_view_ssl"
        ),
        consistency_weight=args.consistency_weight,
        max_steps=SMOKE_MAX_STEPS if args.smoke else args.max_steps,
        resolution=args.resolution,
        batch_size_per_gpu=args.batch_size_per_gpu,
        budget_gpu_hours=(
            SMOKE_BUDGET_GPU_HOURS if args.smoke else args.budget_gpu_hours
        ),
        num_workers=args.num_workers,
        precision=args.precision,
        seed=args.seed,
        mlflow_experiment=args.mlflow_experiment,
        tracking_uri=args.tracking_uri,
        lesion_guided_crops=not args.no_lesion_guided,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    cfg = config_mod.validate(cfg)  # Falla antes de usar torch o la GPU.

    try:
        import mlflow
    except ImportError:
        mlflow = None
    wandb = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError:
            print("ADVERTENCIA: wandb no está instalado; la corrida se registra solo en MLflow.")

    from dermssl.pretrain import torch_loop

    final = torch_loop.train(
        cfg, mlflow_module=mlflow, wandb_module=wandb,
        wandb_project=args.wandb_project, wandb_group=args.wandb_group,
    )
    print(f"Modelo base final: {final}")


if __name__ == "__main__":
    main()
