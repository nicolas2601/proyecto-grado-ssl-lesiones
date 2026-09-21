"""Encadena las etapas del camino B: cache, MoCo, CCAM, pseudo-mascaras, estudiante y evaluacion.

Cada etapa es un script independiente; este driver solo construye y ejecuta sus
comandos en orden, con las rondas de re-etiquetado:

    cache -> contrastive -> ccam -> pseudo_r0 -> student_r1 -> pseudo_r1 ... -> fst -> evaluate

Solo las etapas `evaluate` reciben mascaras de referencia, y solo despues del
entrenamiento. Los argumentos vienen de un preset JSON (`comun/configs`); los
valores dados por linea de comandos (`--rounds`, `--epochs-per-round`) los
sobrescriben.

    # verificar la instalacion en menos de dos minutos (64 imagenes sinteticas)
    python run_pipeline.py --smoke

    # piloto en una GPU, con dos conjuntos de evaluacion
    python run_pipeline.py --config plan-segmentacion/comun/configs/cluster-piloto.json \\
        --workdir salidas/piloto --root ham10000=/datos/ham --root isic2019=/datos/i19 \\
        --root isic2020=/datos/i20 \\
        --eval-set isic2018=/datos/isic2018/images:/datos/isic2018/masks \\
        --eval-set ham10000=/datos/ham/images:/datos/ham/masks

    # dos GPU, con seguimiento en Weights & Biases (la clave va en WANDB_API_KEY)
    python run_pipeline.py ... --nproc-per-node 2 --wandb-project dermssl-segmentacion
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (Path(__file__).resolve().parent, _PLAN_ROOT.parent / "src",
               _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import presets  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402
from pipeline_stages import (  # noqa: E402
    ALL_STAGES,
    SMOKE_RESOLUTION,
    SMOKE_SYNTHETIC_COUNT,
    PipelineConfig,
    Stage,
    build_stage_commands,
    child_env,
    parse_eval_set,
)

PLACEHOLDER_PREFIX = "<"
DEFAULT_CONFIG = _PLAN_ROOT / "comun" / "configs" / "piloto.json"
DEFAULT_WORKDIR = _PLAN_ROOT / "camino-b-cnn" / "salidas" / "piloto"

SMOKE_PRESET = {
    "pipeline": {"rounds": 2, "epochs_per_round": 4, "seed": 0},
    "cache": {"resolution": SMOKE_RESOLUTION, "threads": 4, "synthetic": SMOKE_SYNTHETIC_COUNT,
              "field_black_threshold": 0.06, "field_margin_fraction": 0.04},
    "contrastive": {"epochs": 1, "batch_size": 8, "queue_size": 64, "max_steps_per_epoch": 4,
                    "bn_splits": 2, "log_every": 2, "warmup_epochs": 0},
    "ccam": {"epochs": 1, "batch_size": 8, "max_steps_per_epoch": 4, "log_every": 2,
             "predict_batch_size": 16},
    # Same cap as the cluster presets on purpose. The synthetic frames have lit
    # corners, so the field detector reports "full_frame" and the oriented region
    # covers about 59 % of it, just under the cap. If this ever starts excluding
    # every sample, read excluded_by_area in pseudo_masks_stats.json before
    # loosening it: the orientation flip is the usual culprit, not the cap.
    "pseudo": {"max_area_fraction": 0.6},
    "student": {"batch_size": 8, "lr": 0.003, "log_every": 4, "warmup_epochs": 0},
    "evaluate": {"batch_size": 8},
}
# The FST manifest needs the original files on disk, which the synthetic cache does not have.
SMOKE_STAGES = tuple(name for name in ALL_STAGES if name not in ("fst", "evaluate"))


def find_placeholders(preset: dict) -> list[str]:
    """Names (``section.key``) whose value is still a ``<...>`` placeholder."""
    found = []
    for section, values in preset.items():
        if section.startswith("_") or not isinstance(values, dict):
            continue
        found.extend(
            f"{section}.{key}"
            for key, value in values.items()
            if isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX)
        )
    return found


def smoke_config(workdir: Path, python: str, nproc_per_node: int) -> PipelineConfig:
    return PipelineConfig(
        workdir=Path(workdir), preset=SMOKE_PRESET, nproc_per_node=nproc_per_node,
        python=python, smoke=True, stages=list(SMOKE_STAGES))


def clean_smoke_workdir(workdir: Path, default_workdir: Path) -> None:
    """Remove only the canonical smoke directory, never a user-selected path."""
    if Path(workdir).resolve() == Path(default_workdir).resolve():
        shutil.rmtree(Path(default_workdir), ignore_errors=True)


def run_stages(stages: list[Stage], dry_run: bool, env: dict[str, str] | None = None) -> None:
    extra = dict(env or {})
    if extra:
        print("entorno heredado: " + " ".join(f"{k}={v}" for k, v in sorted(extra.items())),
              flush=True)
    for stage in stages:
        print(f"\n=== {stage.name} ===\n{' '.join(stage.argv)}", flush=True)
        if dry_run:
            continue
        started = time.perf_counter()
        completed = subprocess.run(stage.argv, check=False, env={**os.environ, **extra})
        elapsed = time.perf_counter() - started
        print(f"--- {stage.name}: {elapsed:.1f} s (codigo {completed.returncode})", flush=True)
        if completed.returncode != 0:
            sys.exit(f"error: la etapa {stage.name} fallo con codigo {completed.returncode}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--config", default=None, help=f"Preset JSON de argumentos (por defecto {DEFAULT_CONFIG.name})")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="Carpeta de salidas (ignorada por git)")
    parser.add_argument("--root", action="append", default=[], metavar="NOMBRE=RUTA", help="Raiz de imagenes sin etiquetas; se repite por dataset")
    parser.add_argument("--rounds", type=int, default=None, help="Rondas de re-etiquetado (sobrescribe el preset)")
    parser.add_argument("--epochs-per-round", type=int, default=None, help="Epocas del estudiante por ronda (sobrescribe el preset)")
    parser.add_argument("--stages", nargs="+", default=None, choices=ALL_STAGES, help="Etapas a ejecutar, en el orden fijo del pipeline (por defecto todas)")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="GPU por nodo; con mas de 1 usa torch.distributed.run")
    parser.add_argument("--eval-image-dir", default="", help="Imagenes de prueba (solo evaluacion)")
    parser.add_argument("--eval-mask-dir", default="", help="Mascaras de referencia (solo evaluacion)")
    parser.add_argument("--eval-name", default="referencia", help="Nombre del conjunto de evaluacion")
    parser.add_argument("--eval-set", action="append", default=[], metavar="NOMBRE=IMAGENES:MASCARAS", help="Conjunto de evaluacion adicional; se repite por conjunto")
    parser.add_argument("--fst-manifest", default="", help="CSV image_id,fst_bucket para estratificar la evaluacion por tono")
    parser.add_argument("--wandb-project", default="", help="Proyecto de Weights & Biases; la clave se lee del entorno, nunca de esta linea")
    parser.add_argument("--smoke", action="store_true", help="Corre todas las etapas con 64 imagenes sinteticas y 1 epoca")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime los comandos")
    preload = parser.add_mutually_exclusive_group()
    preload.add_argument("--preload", action="store_true", help="Precarga images.npy en RAM")
    preload.add_argument("--no-preload", action="store_true", help="Conserva images.npy como memmap")
    return _resolve_defaults(parser, parser.parse_args(argv))


def _resolve_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Fill in the defaults, after rejecting the flags that ``--smoke`` would ignore."""
    ignored = [name for name in ("--config", "--stages")
               if getattr(args, name.lstrip("-").replace("-", "_")) is not None]
    if args.smoke and ignored:
        parser.error(f"--smoke trae su propio preset y sus propias etapas: quite {', '.join(ignored)}")
    args.config = args.config or str(DEFAULT_CONFIG)
    args.stages = args.stages or list(ALL_STAGES)
    return args


def _smoke_pipeline(args: argparse.Namespace) -> PipelineConfig:
    canonical = DEFAULT_WORKDIR.with_name("smoke")
    workdir = canonical if args.workdir == str(DEFAULT_WORKDIR) else Path(args.workdir)
    if args.workdir == str(DEFAULT_WORKDIR):
        clean_smoke_workdir(workdir, canonical)
    config = smoke_config(workdir, sys.executable, args.nproc_per_node)
    config.preload, config.no_preload = args.preload, args.no_preload
    return config


def _preset_pipeline(args: argparse.Namespace) -> PipelineConfig:
    preset = presets.load_preset(Path(args.config))
    pending = find_placeholders(preset)
    if pending:
        sys.exit(f"error: el preset tiene valores sin completar: {', '.join(pending)}")
    return PipelineConfig(
        workdir=Path(args.workdir), preset=preset, roots=args.root, rounds=args.rounds,
        epochs_per_round=args.epochs_per_round, nproc_per_node=args.nproc_per_node,
        stages=[s for s in ALL_STAGES if s in args.stages],
        eval_image_dir=args.eval_image_dir, eval_mask_dir=args.eval_mask_dir,
        eval_name=args.eval_name, eval_sets=[parse_eval_set(spec) for spec in args.eval_set],
        fst_manifest=args.fst_manifest, wandb_project=args.wandb_project,
        preload=args.preload, no_preload=args.no_preload)


def main() -> None:
    args = parse_args()
    config = _smoke_pipeline(args) if args.smoke else _preset_pipeline(args)
    run_stages(build_stage_commands(config), args.dry_run, child_env(config))


if __name__ == "__main__":
    main()
