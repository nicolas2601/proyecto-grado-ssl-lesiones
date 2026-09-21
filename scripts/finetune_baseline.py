"""Ajusta un backbone base sobre un conjunto médico (ejecutor de clúster).

Usa el mismo protocolo que los notebooks Kaggle en notebooks/finetune*; ver
src/dermssl/finetune/protocol.py. Está pensado para el clúster de Dr. Karen:
los pesos provienen de una caché precargada de Hugging Face / torch (sin
internet) y los resultados van a un JSON, checkpoint, Weights & Biases y MLflow.

Una tarea por nombre:
    python scripts/finetune_baseline.py --model resnet50 --dataset ham10000 \
        --data-root $SCRATCH/dermssl/data --output-dir $SCRATCH/dermssl/finetune/runs/x

Una tarea por índice del manifiesto (lo que usa el arreglo SLURM, 0..15):
    python scripts/finetune_baseline.py --job-index 3 --data-root ... --output-dir ...

Smoke (1 época sobre un subconjunto estratificado; valida la integración en minutos):
    python scripts/finetune_baseline.py --model resnet50 --dataset ham10000 --smoke ...

Lista las 16 tareas del banco:
    python scripts/finetune_baseline.py --list-jobs
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl.finetune import protocol  # noqa: E402
from dermssl.finetune.config import ConfigError, FinetuneConfig, validate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=protocol.MODELS)
    p.add_argument("--dataset", choices=tuple(protocol.DATASETS))
    p.add_argument("--job-index", type=int,
                    help="0..15; reemplaza --model/--dataset desde el manifiesto")
    p.add_argument("--list-jobs", action="store_true")
    # Underscore spellings mirror Dr. Karen's CT-Encoders runner so the bank
    # can be launched on her cluster without learning a second CLI.
    p.add_argument("--data-root", "--data_path", dest="data_root",
                   help="carpeta con ham10000/ e isic2019/")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir",
                   help="donde se escriben el JSON y el checkpoint")
    p.add_argument("--results-dir", default="",
                    help="carpeta opcional del repositorio a la que copiar el JSON (para git push)")
    p.add_argument("--epochs", type=int, default=protocol.EPOCHS)
    p.add_argument("--batch", type=int, default=protocol.BATCH)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--seed", type=int, default=protocol.SEED)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb-project", default=protocol.WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"),
                   choices=("offline", "online", "disabled"))
    p.add_argument("--mlflow-tracking-uri", default="")
    p.add_argument("--tensorboard-dir", "--tensorboard_dir", dest="tensorboard_dir",
                   default="", help="root for TensorBoard event files (one subfolder "
                   "per run). Default: <output-dir>/tb, like CT-Encoders. "
                   "Pass 'none' to disable.")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    model, dataset = args.model, args.dataset
    if args.job_index is not None:
        job = protocol.job_from_index(args.job_index)
        model, dataset = job.model, job.dataset
    if not model or not dataset:
        raise ConfigError("indique --model y --dataset, o --job-index")
    if not args.data_root or not args.output_dir:
        raise ConfigError("--data-root y --output-dir son obligatorios")
    # TensorBoard is on by default and lands inside the run's own output dir:
    # the UNAB platform shares one MLflow server across every student, so a
    # private event-file directory is the only dashboard that is ours.
    tb = args.tensorboard_dir
    tensorboard_dir = "" if tb == "none" else (tb or str(Path(args.output_dir) / "tb"))
    return FinetuneConfig(
        model=model, dataset=dataset, data_root=args.data_root,
        output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch,
        seed=args.seed, num_workers=args.num_workers, smoke=args.smoke,
        precision=args.precision, wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity, wandb_mode=args.wandb_mode,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tensorboard_dir=tensorboard_dir,
        extra_tags=["smoke"] if args.smoke else [],
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_jobs:
        for job in protocol.manifest():
            print(f"{job.index:2d}  {job.model:<16} {job.dataset:<9} -> {job.results_file}")
        return
    try:
        cfg = validate(build_config(args))  # falla antes de importar torch
    except ConfigError as exc:
        sys.exit(f"config error: {exc}")

    try:
        import wandb
    except ImportError:
        wandb = None
    try:
        import mlflow
    except ImportError:
        mlflow = None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None

    from dermssl.finetune.tracking import build_trackers
    from dermssl.finetune.torch_run import run

    tracker = build_trackers(
        wandb_module=wandb, mlflow_module=mlflow, run=cfg.job_name,
        wandb_project=cfg.wandb_project, wandb_mode=cfg.wandb_mode,
        wandb_entity=cfg.wandb_entity,
        wandb_tags=[cfg.model, cfg.dataset, *cfg.extra_tags],
        wandb_group=cfg.dataset,
        mlflow_experiment=f"{protocol.MLFLOW_EXPERIMENT_PREFIX}{cfg.dataset}",
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        summary_writer_cls=SummaryWriter, tensorboard_dir=cfg.tensorboard_dir,
    )
    if tracker.is_null:
        print("AVISO: sin wandb, mlflow ni tensorboard - la corrida NO queda registrada "
              "en ningun dashboard (el JSON de resultados si se escribe).")

    json_path = run(cfg, tracker)
    if args.results_dir:
        dest = Path(args.results_dir) / cfg.dataset
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, dest / json_path.name)
        print("copiado a", dest / json_path.name)


if __name__ == "__main__":
    main()
