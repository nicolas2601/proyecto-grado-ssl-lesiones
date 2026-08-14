"""Launch a v6 label-free SSL pretraining run on GPU.

Single GPU:
    python scripts/pretrain.py --method dino_continual \
        --image-root /scratch/derm/pool --output-dir /scratch/derm/runs/dino \
        --budget-gpu-hours 8 --tracking-uri file:/scratch/derm/mlruns

Multi GPU (one node, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/pretrain.py --method dino_continual ...

Smoke (validates the full wiring in minutes, still label-free):
    python scripts/pretrain.py --method simsiam --smoke \
        --image-root /path/to/a/few/images --output-dir /tmp/smoke

The MLflow UI over the recorded runs:
    mlflow ui --backend-store-uri file:/scratch/derm/mlruns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl.pretrain import config as config_mod  # noqa: E402

SMOKE_BUDGET_GPU_HOURS = 0.05  # 3 wall-clock minutes on 1 GPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=config_mod.METHODS)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--budget-gpu-hours", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16", choices=config_mod.PRECISIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-experiment", default="obj2-ssl-pretraining")
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument(
        "--no-lesion-guided",
        action="store_true",
        help="ablation switch: uniform local crops instead of border-ring guided",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"tiny budget ({SMOKE_BUDGET_GPU_HOURS} GPU-hours) to validate wiring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_mod.PretrainConfig(
        method=args.method,
        image_root=args.image_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
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
    cfg = config_mod.validate(cfg)  # fail before touching torch or the GPU

    try:
        import mlflow
    except ImportError:
        mlflow = None

    from dermssl.pretrain import torch_loop

    final = torch_loop.train(cfg, mlflow_module=mlflow)
    print(f"final backbone: {final}")


if __name__ == "__main__":
    main()
