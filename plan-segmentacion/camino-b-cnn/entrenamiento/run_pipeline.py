"""Encadena las etapas del camino B: cache, MoCo, CCAM, pseudo-mascaras, estudiante y evaluacion.

Cada etapa es un script independiente; este driver solo construye y ejecuta sus
comandos en orden, con las rondas de re-etiquetado:

    cache -> contrastive -> ccam -> pseudo_r0 -> student_r1 -> pseudo_r1 -> student_r2 ... -> evaluate

Solo la etapa `evaluate` recibe mascaras de referencia, y solo despues del
entrenamiento. Los argumentos vienen de un preset JSON (`comun/configs`); los
valores dados por linea de comandos (`--rounds`, `--epochs-per-round`) los
sobrescriben.

    # verificar la instalacion en menos de dos minutos (64 imagenes sinteticas)
    python run_pipeline.py --smoke

    # piloto en una GPU
    python run_pipeline.py --config plan-segmentacion/comun/configs/piloto.json \\
        --workdir salidas/piloto --root ham10000=/datos/ham --root isic2019=/datos/i19 \\
        --root isic2020=/datos/i20 --eval-image-dir /datos/isic2018/images \\
        --eval-mask-dir /datos/isic2018/masks

    # dos GPU
    python run_pipeline.py ... --nproc-per-node 2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import presets  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

ALL_STAGES = ("env", "cache", "contrastive", "ccam", "pseudo", "student", "evaluate")
PLACEHOLDER_PREFIX = "<"
DEFAULT_CONFIG = _PLAN_ROOT / "comun" / "configs" / "piloto.json"
DEFAULT_WORKDIR = _PLAN_ROOT / "camino-b-cnn" / "salidas" / "piloto"

SCRIPTS = {
    "env": _PLAN_ROOT / "comun" / "nucleo" / "check_env.py",
    "synthetic": _PLAN_ROOT / "comun" / "nucleo" / "synthetic.py",
    "cache": _PLAN_ROOT / "comun" / "datos" / "prepare_cache.py",
    "contrastive": _PLAN_ROOT / "camino-b-cnn" / "preentrenamiento-contrastivo" / "train_moco.py",
    "ccam": _PLAN_ROOT / "camino-b-cnn" / "pseudo-mascaras" / "train_ccam.py",
    "pseudo": _PLAN_ROOT / "camino-b-cnn" / "pseudo-mascaras" / "make_pseudo_masks.py",
    "student": _PLAN_ROOT / "comun" / "estudiante" / "train_student.py",
    "evaluate": _PLAN_ROOT / "comun" / "evaluacion" / "evaluate_segmentation.py",
}
DISTRIBUTED_STAGES = ("contrastive", "ccam", "student")
SEEDED_STAGES = ("cache", "contrastive", "ccam", "student")

SMOKE_PRESET = {
    "pipeline": {"rounds": 2, "epochs_per_round": 4, "seed": 0},
    "cache": {"resolution": 64, "threads": 4, "synthetic": 64},
    "contrastive": {"epochs": 1, "batch_size": 8, "queue_size": 64, "max_steps_per_epoch": 4,
                    "bn_splits": 2, "log_every": 2, "warmup_epochs": 0},
    "ccam": {"epochs": 1, "batch_size": 8, "max_steps_per_epoch": 4, "log_every": 2,
             "predict_batch_size": 16},
    "pseudo": {},
    "student": {"batch_size": 8, "lr": 0.003, "log_every": 4, "warmup_epochs": 0},
    "evaluate": {"batch_size": 8},
}
SMOKE_EVAL_COUNT = 16


@dataclass(frozen=True)
class Stage:
    name: str
    argv: list[str]


@dataclass
class PipelineConfig:
    workdir: Path
    preset: dict
    roots: list[str] = field(default_factory=list)
    rounds: int | None = None
    epochs_per_round: int | None = None
    nproc_per_node: int = 1
    python: str = sys.executable
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    eval_image_dir: str = ""
    eval_mask_dir: str = ""
    eval_name: str = "referencia"
    smoke: bool = False
    preload: bool = False
    no_preload: bool = False


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


def _section(preset: dict, name: str) -> dict:
    return {k: v for k, v in preset.get(name, {}).items() if not k.startswith("_")}


def _launcher(config: PipelineConfig, stage_key: str) -> list[str]:
    script = str(SCRIPTS[stage_key])
    if stage_key in DISTRIBUTED_STAGES and config.nproc_per_node > 1:
        return [config.python, "-m", "torch.distributed.run",
                f"--nproc_per_node={config.nproc_per_node}", script]
    return [config.python, script]


def _stage(config: PipelineConfig, name: str, stage_key: str, fixed: list[str],
           section: dict, extra: dict | None = None) -> Stage:
    """Fixed paths first, then preset flags, then explicit overrides (which replace preset keys)."""
    seed = config.preset.get("pipeline", {}).get("seed")
    overrides = dict(extra or {})
    if seed is not None and stage_key in SEEDED_STAGES:
        overrides.setdefault("seed", seed)
    kept = {key: value for key, value in section.items() if key not in overrides}
    argv = [*_launcher(config, stage_key), *fixed, *presets.flags_from_mapping(kept),
            *presets.flags_from_mapping(overrides)]
    return Stage(name, argv)


def _resolve_rounds(config: PipelineConfig) -> tuple[int, int]:
    pipeline = config.preset.get("pipeline", {})
    rounds = config.rounds if config.rounds is not None else pipeline.get("rounds", 1)
    epochs = (
        config.epochs_per_round
        if config.epochs_per_round is not None
        else pipeline.get("epochs_per_round", 10)
    )
    return int(rounds), int(epochs)


def build_stage_commands(config: PipelineConfig) -> list[Stage]:
    """Return the ordered, fully explicit commands for the selected stages."""
    work, preset = Path(config.workdir), config.preset
    cache, moco, ccam = work / "cache", work / "moco", work / "ccam"
    rounds, epochs_per_round = _resolve_rounds(config)
    wanted = set(config.stages)
    stages: list[Stage] = []

    if "env" in wanted:
        stages.append(Stage("env", [config.python, str(SCRIPTS["env"])]))
    if "cache" in wanted:
        source = ["--synthetic", str(_section(preset, "cache")["synthetic"])] if config.smoke else []
        roots = [flag for spec in config.roots for flag in ("--root", spec)]
        section = {k: v for k, v in _section(preset, "cache").items() if k != "synthetic"}
        stage = _stage(config, "cache", "cache", ["--output-dir", str(cache), *roots, *source], section)
        stages.append(stage)
    if "contrastive" in wanted:
        stages.append(_stage(config, "contrastive", "contrastive",
                             ["--cache-dir", str(cache), "--output-dir", str(moco)],
                             _section(preset, "contrastive")))
    if "ccam" in wanted:
        stages.append(_stage(config, "ccam", "ccam",
                             ["--cache-dir", str(cache), "--backbone", str(moco / "backbone.pt"),
                              "--output-dir", str(ccam)], _section(preset, "ccam")))
    if "pseudo" in wanted:
        stages.append(_stage(config, "pseudo_r0", "pseudo",
                             ["--saliency", str(ccam / "saliency.npy"),
                              "--output-dir", str(work / "pseudo_r0"), "--source", "ccam"],
                             _section(preset, "pseudo")))
    if "student" in wanted:
        stages.extend(_student_rounds(config, work, cache, rounds, epochs_per_round))
    if "evaluate" in wanted and config.eval_image_dir and config.eval_mask_dir:
        stages.append(_stage(config, "evaluate", "evaluate",
                             ["--checkpoint", str(work / f"student_r{rounds}" / "best.pt"),
                              "--image-dir", config.eval_image_dir,
                              "--mask-dir", config.eval_mask_dir,
                              "--output-dir", str(work / f"evaluacion_{config.eval_name}")],
                             _section(preset, "evaluate")))
    stages = _with_synthetic_eval(config, stages, work)
    runtime_flags = ["--no-preload"] if config.no_preload else ["--preload"] if config.preload else []
    return [
        Stage(stage.name, [*stage.argv, *runtime_flags])
        if stage.name.split("_", 1)[0] in {"contrastive", "ccam", "student"}
        else stage
        for stage in stages
    ]


def _student_rounds(config, work, cache, rounds, epochs_per_round) -> list[Stage]:
    stages = []
    for index in range(1, rounds + 1):
        is_last = index == rounds
        extra = {"epochs": epochs_per_round, "no_predict": is_last}
        stages.append(_stage(
            config, f"student_r{index}", "student",
            ["--cache-dir", str(cache),
             "--pseudo-masks", str(work / f"pseudo_r{index - 1}" / "pseudo_masks.npy"),
             "--output-dir", str(work / f"student_r{index}")],
            _section(config.preset, "student"), extra))
        if not is_last:
            stages.append(_stage(
                config, f"pseudo_r{index}", "pseudo",
                ["--saliency", str(work / f"student_r{index}" / "predictions.npy"),
                 "--output-dir", str(work / f"pseudo_r{index}"), "--source", "student"],
                _section(config.preset, "pseudo")))
    return stages


def _with_synthetic_eval(config: PipelineConfig, stages: list[Stage], work: Path) -> list[Stage]:
    """Smoke only: generate a synthetic reference set right before the evaluation."""
    if not config.smoke:
        return stages
    size = _section(config.preset, "cache")["resolution"]
    image_dir, mask_dir = work / "synthetic_eval" / "images", work / "synthetic_eval" / "masks"
    generate = Stage("synthetic_eval", [
        config.python, str(SCRIPTS["synthetic"]), "--image-dir", str(image_dir),
        "--mask-dir", str(mask_dir), "--count", str(SMOKE_EVAL_COUNT), "--size", str(size)])
    evaluate = Stage("evaluate", [
        config.python, str(SCRIPTS["evaluate"]),
        "--checkpoint", str(work / f"student_r{_resolve_rounds(config)[0]}" / "best.pt"),
        "--image-dir", str(image_dir), "--mask-dir", str(mask_dir),
        "--output-dir", str(work / "evaluacion_synthetic"),
        *presets.flags_from_mapping(_section(config.preset, "evaluate"))])
    return [*stages, generate, evaluate]


def smoke_config(workdir: Path, python: str, nproc_per_node: int) -> PipelineConfig:
    return PipelineConfig(
        workdir=Path(workdir), preset=SMOKE_PRESET, nproc_per_node=nproc_per_node,
        python=python, smoke=True,
        stages=[s for s in ALL_STAGES if s != "evaluate"])


def clean_smoke_workdir(workdir: Path, default_workdir: Path) -> None:
    """Remove only the canonical smoke directory, never a user-selected path."""
    if Path(workdir).resolve() == Path(default_workdir).resolve():
        shutil.rmtree(Path(default_workdir), ignore_errors=True)


def run_stages(stages: list[Stage], dry_run: bool) -> None:
    for stage in stages:
        printable = " ".join(stage.argv)
        print(f"\n=== {stage.name} ===\n{printable}", flush=True)
        if dry_run:
            continue
        started = time.perf_counter()
        completed = subprocess.run(stage.argv, check=False)
        elapsed = time.perf_counter() - started
        print(f"--- {stage.name}: {elapsed:.1f} s (codigo {completed.returncode})", flush=True)
        if completed.returncode != 0:
            sys.exit(f"error: la etapa {stage.name} fallo con codigo {completed.returncode}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Preset JSON de argumentos")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="Carpeta de salidas (ignorada por git)")
    parser.add_argument("--root", action="append", default=[], metavar="NOMBRE=RUTA", help="Raiz de imagenes sin etiquetas; se repite por dataset")
    parser.add_argument("--rounds", type=int, default=None, help="Rondas de re-etiquetado (sobrescribe el preset)")
    parser.add_argument("--epochs-per-round", type=int, default=None, help="Epocas del estudiante por ronda (sobrescribe el preset)")
    parser.add_argument("--stages", nargs="+", default=list(ALL_STAGES), choices=ALL_STAGES, help="Etapas a ejecutar, en el orden fijo del pipeline")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="GPU por nodo; con mas de 1 usa torch.distributed.run")
    parser.add_argument("--eval-image-dir", default="", help="Imagenes de prueba (solo evaluacion)")
    parser.add_argument("--eval-mask-dir", default="", help="Mascaras de referencia (solo evaluacion)")
    parser.add_argument("--eval-name", default="referencia", help="Nombre del conjunto de evaluacion")
    parser.add_argument("--smoke", action="store_true", help="Corre todas las etapas con 64 imagenes sinteticas y 1 epoca")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime los comandos")
    preload = parser.add_mutually_exclusive_group()
    preload.add_argument("--preload", action="store_true", help="Precarga images.npy en RAM")
    preload.add_argument("--no-preload", action="store_true", help="Conserva images.npy como memmap")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_workdir = (Path(args.workdir) if args.workdir != str(DEFAULT_WORKDIR)
                         else DEFAULT_WORKDIR.with_name("smoke"))
        if args.workdir == str(DEFAULT_WORKDIR):
            clean_smoke_workdir(smoke_workdir, DEFAULT_WORKDIR.with_name("smoke"))
        config = smoke_config(smoke_workdir, sys.executable, args.nproc_per_node)
        config.preload = args.preload
        config.no_preload = args.no_preload
    else:
        preset = presets.load_preset(Path(args.config))
        pending = find_placeholders(preset)
        if pending:
            sys.exit(f"error: el preset tiene valores sin completar: {', '.join(pending)}")
        config = PipelineConfig(
            workdir=Path(args.workdir), preset=preset, roots=args.root, rounds=args.rounds,
            epochs_per_round=args.epochs_per_round, nproc_per_node=args.nproc_per_node,
            stages=[s for s in ALL_STAGES if s in args.stages],
            eval_image_dir=args.eval_image_dir, eval_mask_dir=args.eval_mask_dir,
            eval_name=args.eval_name, preload=args.preload, no_preload=args.no_preload)
    run_stages(build_stage_commands(config), args.dry_run)


if __name__ == "__main__":
    main()
