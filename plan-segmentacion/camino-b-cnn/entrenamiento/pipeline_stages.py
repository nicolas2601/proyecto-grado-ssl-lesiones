"""Stage builders for the Camino B driver: one explicit command per step.

Kept apart from ``run_pipeline.py`` so the driver stays a thin CLI. Nothing here
runs a process: a preset plus a handful of overrides become ``argv`` lists that
the driver prints or executes.

Reference masks appear only in the evaluation stages, never in training ones.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import presets  # noqa: E402

ALL_STAGES = ("env", "cache", "contrastive", "ccam", "pseudo", "student", "fst", "evaluate")

SCRIPTS = {
    "env": _PLAN_ROOT / "comun" / "nucleo" / "check_env.py",
    "synthetic": _PLAN_ROOT / "comun" / "nucleo" / "synthetic.py",
    "cache": _PLAN_ROOT / "comun" / "datos" / "prepare_cache.py",
    "contrastive": _PLAN_ROOT / "camino-b-cnn" / "preentrenamiento-contrastivo" / "train_moco.py",
    "ccam": _PLAN_ROOT / "camino-b-cnn" / "pseudo-mascaras" / "train_ccam.py",
    "pseudo": _PLAN_ROOT / "camino-b-cnn" / "pseudo-mascaras" / "make_pseudo_masks.py",
    "student": _PLAN_ROOT / "comun" / "estudiante" / "train_student.py",
    "fst": _PLAN_ROOT / "comun" / "evaluacion" / "fst_pseudo_manifest.py",
    "evaluate": _PLAN_ROOT / "comun" / "evaluacion" / "evaluate_segmentation.py",
}
DISTRIBUTED_STAGES = ("contrastive", "ccam", "student")
SEEDED_STAGES = ("cache", "contrastive", "ccam", "student")
TRACKED_STAGES = ("contrastive", "ccam", "student", "evaluate")
RUNTIME_FLAG_STAGES = frozenset({"contrastive", "ccam", "student"})

FIELD_FILE = "field.npy"
IMAGES_FILE = "images.npy"
MANIFEST_FILE = "manifest.csv"
FST_DIR = "fst"
FST_JSON_FILE = "fst_pseudo_manifest.json"
FST_CSV_FILE = "fst_pseudo_manifest.csv"
SMOKE_EVAL_COUNT = 16
SMOKE_RESOLUTION = 64
SMOKE_SYNTHETIC_COUNT = 64
# The smoke is a check, not a benchmark: on a loaded machine a child with one
# torch thread per core crawls (the smoke went from ~34 s to ~300 s under load).
SMOKE_THREADS = {name: "2" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}


@dataclass(frozen=True)
class Stage:
    name: str
    argv: list[str]


@dataclass(frozen=True)
class EvalSet:
    """One reference set: a name, its images and its masks."""

    name: str
    image_dir: str
    mask_dir: str


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
    eval_sets: list[EvalSet] = field(default_factory=list)
    fst_manifest: str = ""
    wandb_project: str = ""
    smoke: bool = False
    preload: bool = False
    no_preload: bool = False


def parse_eval_set(spec: str) -> EvalSet:
    """Read ``NOMBRE=IMAGENES:MASCARAS`` into an :class:`EvalSet`."""
    name, assigned, rest = spec.partition("=")
    image_dir, separated, mask_dir = rest.rpartition(":")
    if not (name and assigned and image_dir and separated and mask_dir):
        raise ValueError(f"se esperaba NOMBRE=IMAGENES:MASCARAS, no {spec!r}")
    return EvalSet(name, image_dir, mask_dir)


def child_env(config: PipelineConfig) -> dict[str, str]:
    """Extra environment for the child processes; never carries the API key."""
    env = dict(SMOKE_THREADS) if config.smoke else {}
    if config.wandb_project:
        env.update(WANDB_PROJECT=config.wandb_project, WANDB_RUN_GROUP=Path(config.workdir).name)
    return env


def evaluation_sets(config: PipelineConfig) -> list[EvalSet]:
    """The legacy single set first, then every ``--eval-set``."""
    sets: list[EvalSet] = []
    if config.eval_image_dir and config.eval_mask_dir:
        sets.append(EvalSet(config.eval_name, config.eval_image_dir, config.eval_mask_dir))
    sets.extend(config.eval_sets)
    return sets


def resolve_rounds(config: PipelineConfig) -> tuple[int, int]:
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
    rounds, epochs_per_round = resolve_rounds(config)
    wanted = set(config.stages)
    stages: list[Stage] = []

    if "env" in wanted:
        stages.append(Stage("env", [config.python, str(SCRIPTS["env"])]))
    if "cache" in wanted:
        stages.append(_cache_stage(config, cache))
    if "contrastive" in wanted:
        stages.append(_stage(config, "contrastive", "contrastive",
                             ["--cache-dir", str(cache), "--output-dir", str(moco)],
                             _section(preset, "contrastive")))
    if "ccam" in wanted:
        stages.append(_stage(config, "ccam", "ccam",
                             ["--cache-dir", str(cache), "--backbone", str(moco / "backbone.pt"),
                              "--output-dir", str(ccam)], _section(preset, "ccam")))
    if "pseudo" in wanted:
        stages.append(_pseudo_stage(config, "pseudo_r0", cache,
                                    ccam / "saliency.npy", work / "pseudo_r0", "ccam"))
    if "student" in wanted:
        stages.extend(_student_rounds(config, work, cache, rounds, epochs_per_round))
    if "fst" in wanted:
        stages.append(_fst_stage(config, work, cache, rounds))
    if "evaluate" in wanted:
        stages.extend(_evaluate_stage(config, work, rounds, item)
                      for item in evaluation_sets(config))
    stages = _with_synthetic_eval(config, stages, work, rounds)
    return [_with_runtime_flags(config, stage) for stage in stages]


def _section(preset: dict, name: str) -> dict:
    return {k: v for k, v in preset.get(name, {}).items() if not k.startswith("_")}


def _launcher(config: PipelineConfig, stage_key: str) -> list[str]:
    script = str(SCRIPTS[stage_key])
    if stage_key in DISTRIBUTED_STAGES and config.nproc_per_node > 1:
        return [config.python, "-m", "torch.distributed.run",
                f"--nproc_per_node={config.nproc_per_node}", script]
    return [config.python, script]


def _tracking_flags(config: PipelineConfig, stage_key: str) -> list[str]:
    """Only the project name travels on the command line; the key stays in the env."""
    if not config.wandb_project or stage_key not in TRACKED_STAGES:
        return []
    return ["--wandb-project", config.wandb_project]


def _stage(config: PipelineConfig, name: str, stage_key: str, fixed: list[str],
           section: dict, extra: dict | None = None) -> Stage:
    """Fixed paths first, then preset flags, then explicit overrides (which replace preset keys)."""
    seed = config.preset.get("pipeline", {}).get("seed")
    overrides = dict(extra or {})
    if seed is not None and stage_key in SEEDED_STAGES:
        overrides.setdefault("seed", seed)
    kept = {key: value for key, value in section.items() if key not in overrides}
    argv = [*_launcher(config, stage_key), *fixed, *presets.flags_from_mapping(kept),
            *presets.flags_from_mapping(overrides), *_tracking_flags(config, stage_key)]
    return Stage(name, argv)


def _field_flags(config: PipelineConfig, cache: Path) -> list[str]:
    """The dermoscope field mask, unless the preset turned it off in the cache stage."""
    if _section(config.preset, "cache").get("no_field"):
        return []
    return ["--field", str(cache / FIELD_FILE)]


def _cache_stage(config: PipelineConfig, cache: Path) -> Stage:
    section = _section(config.preset, "cache")
    count = section.get("synthetic", SMOKE_SYNTHETIC_COUNT)
    source = ["--synthetic", str(count)] if config.smoke else []
    roots = [flag for spec in config.roots for flag in ("--root", spec)]
    kept = {key: value for key, value in section.items() if key != "synthetic"}
    return _stage(config, "cache", "cache", ["--output-dir", str(cache), *roots, *source], kept)


def _pseudo_stage(config: PipelineConfig, name: str, cache: Path, saliency: Path,
                  output: Path, source: str) -> Stage:
    """``artifact_suppression`` is a switch, not a flag: it selects ``--images``."""
    section = dict(_section(config.preset, "pseudo"))
    suppress = section.pop("artifact_suppression", True)
    images = ["--images", str(cache / IMAGES_FILE)] if suppress else []
    fixed = ["--saliency", str(saliency), "--output-dir", str(output), "--source", source,
             *_field_flags(config, cache), *images]
    return _stage(config, name, "pseudo", fixed, section)


def _student_rounds(config: PipelineConfig, work: Path, cache: Path,
                    rounds: int, epochs_per_round: int) -> list[Stage]:
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
            stages.append(_pseudo_stage(
                config, f"pseudo_r{index}", cache,
                work / f"student_r{index}" / "predictions.npy",
                work / f"pseudo_r{index}", "student"))
    return stages


def _fst_stage(config: PipelineConfig, work: Path, cache: Path, rounds: int) -> Stage:
    """ITA and FST manifest built from the pseudo-masks of the last round."""
    last = work / f"pseudo_r{max(rounds - 1, 0)}" / "pseudo_masks.npy"
    argv = [config.python, str(SCRIPTS["fst"]),
            "--manifest", str(cache / MANIFEST_FILE),
            "--pseudo-masks", str(last),
            "--output-json", str(work / FST_DIR / FST_JSON_FILE),
            "--output-csv", str(fst_manifest_csv(work)),
            *_field_flags(config, cache)]
    return Stage("fst", argv)


def fst_manifest_csv(work: Path) -> Path:
    """The ``image_id,fst_bucket`` table the evaluation stratifies with."""
    return Path(work) / FST_DIR / FST_CSV_FILE


def _fst_manifest_flags(config: PipelineConfig, work: Path) -> list[str]:
    """An explicit manifest wins; otherwise the one the ``fst`` stage writes.

    Without the ``fst`` stage the CSV of a previous run still counts when it is
    on disk. When the preset asks for strata and no manifest exists anywhere,
    fail here: an evaluation that silently drops the strata block is worse
    than one that does not start.
    """
    if config.fst_manifest:
        return ["--fst-manifest", config.fst_manifest]
    deduced = fst_manifest_csv(work)
    if "fst" in config.stages or deduced.is_file():
        return ["--fst-manifest", str(deduced)]
    if _section(config.preset, "evaluate").get("strata_column"):
        raise ValueError(
            "the preset sets evaluate.strata_column but no FST manifest exists: pass "
            f"--fst-manifest, include the fst stage or produce {deduced}"
        )
    return []


def _evaluate_stage(config: PipelineConfig, work: Path, rounds: int, item: EvalSet) -> Stage:
    output = work / f"evaluacion_{item.name}"
    fixed = ["--checkpoint", str(work / f"student_r{rounds}" / "best.pt"),
             "--image-dir", item.image_dir, "--mask-dir", item.mask_dir,
             "--output-dir", str(output), "--eval-name", item.name,
             "--overlay-dir", str(output / "overlays"),
             *_fst_manifest_flags(config, work)]
    return _stage(config, f"evaluate_{item.name}", "evaluate", fixed,
                  _section(config.preset, "evaluate"))


def _with_synthetic_eval(config: PipelineConfig, stages: list[Stage],
                         work: Path, rounds: int) -> list[Stage]:
    """Smoke only: generate a synthetic reference set right before the evaluation."""
    if not config.smoke:
        return stages
    size = _section(config.preset, "cache").get("resolution", SMOKE_RESOLUTION)
    base = work / "synthetic_eval"
    images, masks = base / "images", base / "masks"
    generate = Stage("synthetic_eval", [
        config.python, str(SCRIPTS["synthetic"]), "--image-dir", str(images),
        "--mask-dir", str(masks), "--count", str(SMOKE_EVAL_COUNT), "--size", str(size)])
    item = EvalSet("synthetic", str(images), str(masks))
    return [*stages, generate, _evaluate_stage(config, work, rounds, item)]


def _with_runtime_flags(config: PipelineConfig, stage: Stage) -> Stage:
    flags = ["--no-preload"] if config.no_preload else ["--preload"] if config.preload else []
    if not flags or stage.name.split("_", 1)[0] not in RUNTIME_FLAG_STAGES:
        return stage
    return Stage(stage.name, [*stage.argv, *flags])
