"""Run the whole tone pipeline with one command, and survive a pod restart.

The UNABIA pod has died four times mid-run. Recovery used to mean a human
reading the runbook, working out which stage had finished and pasting the
next command by hand - which is how an arm ended up relaunched with the
wrong flags. Here every stage declares the artefact that proves it finished:
a stage whose artefact exists is skipped, an interrupted one is relaunched
exactly as it was, and the training loop itself resumes from its own
``resume.pt``. After the container dies the whole recovery is:

    python scripts/run_stage2.py --repo . --outputs $HOME/work/proyectonp/salidas \\
        --ham-root /home/jovyan/work/datos/ham10000

Run it again as many times as the pod needs; it only ever does what is left.

``--outputs`` must already exist: the script will not create it. "The
artefact is not there" is how both a fresh run and a wrong path look from
the inside, and the wrong path costs forty-eight GPU-hours of silent
retraining. Creating the tree once, by hand, is what separates them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Every 5 percent of the budget. The pod loses at most that much wall clock
# when it dies, and the 2026-09-21 crash proved a coarse cadence is the
# difference between losing twelve minutes and losing half a run.
CHECKPOINT_FRACTIONS = tuple(round(0.05 * i, 2) for i in range(1, 21))
SOURCE_REVISION = "pool-final"
POOL_HASH_FILE = "pool_split_hash.json"
# Replaced with the real hash when the stage launches; see resolve_argv.
SPLIT_HASH_TOKEN = "@POOL_SPLIT_HASH@"
# Likewise for the tone-transform manifest revision; config.py rejects a value
# that does not match the manifest, so it can only be read from the manifest.
SHIFT_REVISION_TOKEN = "@SHIFT_MANIFEST_REVISION@"
STATE_FILE = "stage2_state.json"
BACKBONE_FILE = "simsiam_dinov2_backbone_final.pt"
METHOD = "simsiam_dinov2"
ARCH = "dinov2_vits14"
# Both DINOv2 sizes, compared at EQUAL COMPUTE rather than equal epochs: the
# budget is in GPU-hours, so the larger backbone simply sees fewer samples in
# the same time. That is the comparison the thesis claims to make.
DEFAULT_ARCHS = (ARCH,)


def _arch_suffix(arch: str) -> str:
    """``dinov2_vitb14`` -> ``vitb14``; the tail that names an arm."""
    return arch.split("_")[-1]


def _arm_name(kind: str, arch: str, archs: list[str]) -> str:
    """``ssl_plain`` with one architecture, ``ssl_plain_vitb14`` with several."""
    if len(archs) == 1:
        return f"ssl_{kind}"
    return f"ssl_{kind}_{_arch_suffix(arch)}"


@dataclass
class Settings:
    repo: Path
    outputs: Path
    python: str
    ham_root: Path
    budget_gpu_hours: float = 12.0
    nproc_per_node: int = 2
    batch_size_per_gpu: int = 64
    num_workers: int = 0
    wandb_project: str = "dermssl-segmentacion"
    wandb_group: str = "etapa2"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    archs: list[str] = field(default_factory=lambda: list(DEFAULT_ARCHS))
    only: str | None = None
    dry_run: bool = False
    max_retries: int = 2
    code_revision: str = ""


@dataclass
class Stage:
    name: str
    argv: list[str]
    done_when: Path
    note: str = ""


def _script(settings: Settings, *parts: str) -> str:
    return str(Path(settings.repo, *parts))


def _pool_dir(settings: Settings) -> Path:
    return settings.outputs / "tone" / "inputs" / "originals"


def compute_pool_split_hash(pool: Path) -> str:
    """SHA-256 of the sorted pool listing: which images the run trained on.

    ``split_hash`` is the field that answers "over what data". Derived from
    the pool itself it is recomputable by anyone holding the folder, and it
    changes if and only if the pool changes - which a hardcoded constant
    cannot do, and which is why the constant would lie exactly like the
    zeroed placeholder did.
    """
    names = sorted(str(p.relative_to(pool)) for p in pool.rglob("*.png"))
    if not names:
        raise ValueError(f"el pool esta vacio: {pool}")
    digest = hashlib.sha256()
    digest.update(f"{len(names)}\n".encode())
    for name in names:
        digest.update(name.encode() + b"\n")
    return digest.hexdigest()


def pool_split_hash(settings: Settings) -> str:
    """The pool hash, written once and checked on every later launch.

    If a later launch derives a different hash the pool moved under us, and
    the arms already trained are no longer comparable with the ones about to
    start. That must stop the run, not be silently recorded as a second
    value: a paired comparison across two different pools is not a
    comparison.
    """
    receipt = settings.outputs / "tone" / "inputs" / POOL_HASH_FILE
    current = compute_pool_split_hash(_pool_dir(settings))
    if receipt.is_file():
        previous = json.loads(receipt.read_text(encoding="utf-8"))["split_hash"]
        if previous != current:
            raise ValueError(
                f"el pool cambio desde la primera corrida: {previous[:12]}... -> "
                f"{current[:12]}...; los brazos ya entrenados no son comparables "
                f"con los que faltan. Revisa {_pool_dir(settings)} antes de seguir."
            )
        return previous
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps({"split_hash": current, "pool": str(_pool_dir(settings))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return current


def shift_manifest_revision(settings: Settings) -> str:
    """The revision recorded inside the tone-transform manifest.

    config.py refuses a transformed_data_revision that disagrees with the
    manifest, so this value has exactly one honest source. Regenerating the
    transforms changes it, which a constant here could not follow.
    """
    manifest = settings.outputs / "tone" / "shift2" / "tone_transform_manifest.json"
    revision = json.loads(manifest.read_text(encoding="utf-8")).get("revision")
    if not revision:
        raise ValueError(f"{manifest} no declara 'revision'")
    return revision


def resolve_argv(argv: list[str], settings: Settings) -> list[str]:
    """Substitute the placeholders, now that their sources exist.

    Stages are built before the export and tone_pairs stages have run, so
    neither the pool hash nor the manifest revision can be computed at
    construction time. They are replaced when the stage is actually launched.
    """
    resolvers = {
        SPLIT_HASH_TOKEN: pool_split_hash,
        SHIFT_REVISION_TOKEN: shift_manifest_revision,
    }
    pending = {token for token in resolvers if token in argv}
    if not pending:
        return argv
    values = {token: resolvers[token](settings) for token in pending}
    return [values.get(item, item) for item in argv]


def _reproducible_command(argv: list[str], settings: Settings) -> str:
    """The command with the environment that makes it runnable by hand."""
    base = dict(os.environ)
    overrides = {
        key: value for key, value in stage_environment(settings).items()
        if base.get(key) != value
    }
    prefix = " ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
    return f"env {prefix} {' '.join(argv)}" if prefix else " ".join(argv)


def _preview_argv(stage: Stage, settings: Settings) -> list[str]:
    """The command a dry run shows, with the hash resolved when it can be.

    A preview that still shows the placeholder is fine before the pool
    exists; pretending it failed would be worse than showing the token.
    """
    try:
        return resolve_argv(stage.argv, settings)
    except (OSError, ValueError):
        return stage.argv


def _fractions_argv() -> list[str]:
    return ["--checkpoint-fractions", *[str(f) for f in CHECKPOINT_FRACTIONS]]


def _export_stage(settings: Settings) -> Stage:
    inputs = settings.outputs / "tone" / "inputs"
    return Stage(
        name="export",
        argv=[
            settings.python,
            _script(settings, "plan-segmentacion", "comun", "datos", "export_tone_inputs.py"),
            "--cache-dir", str(settings.outputs / "final" / "cache"),
            "--pseudo-masks", str(settings.outputs / "final" / "pseudo_r2" / "pseudo_masks.npy"),
            "--output-root", str(inputs),
            "--source-revision", SOURCE_REVISION,
            "--version", "d81a492",
        ],
        done_when=inputs / "export_summary.json",
        note="masks and their images, at the cache resolution",
    )


def _tone_pairs_stage(settings: Settings) -> Stage:
    shift = settings.outputs / "tone" / "shift2"
    return Stage(
        name="tone_pairs",
        argv=[
            settings.python,
            _script(settings, "scripts", "generate_tone_transforms.py"),
            "--image-root", str(settings.outputs / "tone" / "inputs" / "originals"),
            "--mask-root", str(settings.outputs / "tone" / "inputs" / "masks"),
            "--mask-manifest", str(settings.outputs / "tone" / "inputs" / "mask_manifest.json"),
            "--output-root", str(shift),
            "--source-revision", SOURCE_REVISION,
            "--shift-buckets", "2",
        ],
        done_when=shift / "tone_transform_manifest.json",
        note="the darker-skin counterpart of every eligible image",
    )


def _warmup_stage(settings: Settings) -> Stage:
    receipt = settings.outputs / "scratch" / "hub_warmup.json"
    return Stage(
        name="warmup",
        argv=[
            settings.python,
            _script(settings, "scripts", "warm_hub_cache.py"),
            "--archs", *settings.archs,
            "--receipt", str(receipt),
        ],
        done_when=receipt,
        note="fill the torch.hub cache from one process, before any distributed arm",
    )


def _pretrain_argv(settings: Settings, output_dir: Path, arch: str) -> list[str]:
    return [
        settings.python, "-m", "torch.distributed.run",
        f"--nproc_per_node={settings.nproc_per_node}",
        _script(settings, "scripts", "pretrain.py"),
        "--method", METHOD,
        "--backbone", arch,
        "--image-root", str(settings.outputs / "tone" / "inputs" / "originals"),
        "--output-dir", str(output_dir),
        "--budget-gpu-hours", str(settings.budget_gpu_hours),
        "--batch-size-per-gpu", str(settings.batch_size_per_gpu),
        "--num-workers", str(settings.num_workers),
        "--precision", "bf16",
        "--split-hash", SPLIT_HASH_TOKEN,
        "--data-revision", SOURCE_REVISION,
        *_fractions_argv(),
        "--wandb-project", settings.wandb_project,
        "--wandb-group", settings.wandb_group,
    ]


def _plain_stage(settings: Settings, arch: str) -> Stage:
    name = _arm_name("plain", arch, settings.archs)
    output_dir = settings.outputs / "tone" / name
    return Stage(
        name=name,
        argv=_pretrain_argv(settings, output_dir, arch),
        done_when=output_dir / BACKBONE_FILE,
        note=f"control arm on {arch}: two views of the same image",
    )


def _proposed_stage(settings: Settings, arch: str) -> Stage:
    name = _arm_name("proposed", arch, settings.archs)
    output_dir = settings.outputs / "tone" / name
    shift = settings.outputs / "tone" / "shift2"
    return Stage(
        name=name,
        argv=[
            *_pretrain_argv(settings, output_dir, arch),
            "--transformed-image-root", str(shift),
            "--transformed-manifest-path", str(shift / "tone_transform_manifest.json"),
            "--transformed-data-revision", SHIFT_REVISION_TOKEN,
        ],
        done_when=output_dir / BACKBONE_FILE,
        note=f"proposed arm on {arch}: the original against its tone-shifted copy",
    )


def _bank_backbones(settings: Settings) -> list[str]:
    """Every adapted arm plus the unadapted public weights of ITS OWN size.

    Comparing an adapted ViT-S against a public ViT-B would confound the
    adaptation with the capacity, so each architecture brings its own
    baseline into the same table.
    """
    tone = settings.outputs / "tone"
    specs: list[str] = []
    for arch in settings.archs:
        label = "dinov2_public" if len(settings.archs) == 1 else f"dinov2_public_{_arch_suffix(arch)}"
        specs += ["--backbone", f"{label}={arch}:public"]
    for arch in settings.archs:
        for kind in ("plain", "proposed"):
            name = _arm_name(kind, arch, settings.archs)
            specs += ["--backbone", f"{name}={arch}:{tone / name / BACKBONE_FILE}"]
    return specs


def _bank_stage(settings: Settings) -> Stage:
    bank = settings.outputs / "tone" / "bank"
    return Stage(
        name="bank",
        argv=[
            settings.python,
            _script(settings, "scripts", "cluster_frozen_bank.py"),
            "--ham-root", str(settings.ham_root),
            *_bank_backbones(settings),
            "--tone-manifest", str(settings.outputs / "final" / "fst" / "fst_pseudo_manifest.csv"),
            "--strata-column", "tone_tercile",
            "--probes", "logreg", "logreg_balanced", "knn",
            "--fractions", "1.0", "0.1",
            "--seeds", *[str(s) for s in settings.seeds],
            "--output-dir", str(bank),
            "--cache-dir", str(bank / "cache"),
            "--wandb-project", settings.wandb_project,
        ],
        done_when=bank / "bank.json",
        note="frozen backbones on HAM10000, stratified by skin tone",
    )


def all_stages(settings: Settings) -> list[Stage]:
    """Export, the tone pairs, both arms of every architecture, and the bank."""
    stages = [_export_stage(settings), _tone_pairs_stage(settings), _warmup_stage(settings)]
    for arch in settings.archs:
        stages.append(_plain_stage(settings, arch))
        stages.append(_proposed_stage(settings, arch))
    stages.append(_bank_stage(settings))
    return stages


def build_stages(settings: Settings) -> list[Stage]:
    """Every stage in order, or the single one named by ``--only``."""
    stages = all_stages(settings)
    if settings.only is None:
        return stages
    chosen = [stage for stage in stages if stage.name == settings.only]
    if not chosen:
        raise ValueError(
            f"etapa desconocida: {settings.only!r}; use {[s.name for s in stages]}"
        )
    return chosen


def is_done(stage: Stage) -> bool:
    """A stage is finished when the artefact it promises exists."""
    return stage.done_when.is_file()


def pending_stages(stages: list[Stage]) -> list[Stage]:
    return [stage for stage in stages if not is_done(stage)]


def stage_environment(settings: Settings, base_env: dict | None = None) -> dict:
    """Environment for every stage, with all scratch pinned inside ``outputs``.

    The pod's overlay filesystem had 25 GB free while the NFS share had 40 TB:
    a single resume checkpoint is 400 MB and torch.hub weights are 88 MB, so a
    long run that writes to the default locations fills the container's disk
    and takes the run down with it. Every cache and temporary directory is
    therefore redirected into the outputs tree, which lives on the share.
    """
    env = dict(os.environ if base_env is None else base_env)
    scratch = settings.outputs / "scratch"
    env.update({
        "TMPDIR": str(scratch / "tmp"),
        "TORCH_HOME": str(scratch / "torch"),
        "WANDB_DIR": str(scratch / "wandb"),
        "WANDB_CACHE_DIR": str(scratch / "wandb"),
        "HF_HOME": str(scratch / "huggingface"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
    })
    if settings.code_revision and not env.get("GIT_COMMIT"):
        env["GIT_COMMIT"] = settings.code_revision
    return env


def detect_code_revision(repo: Path) -> str:
    """The commit the arms are trained from; the loop reads it as GIT_COMMIT.

    Without it every run.json records ``code_revision: null`` and nobody can
    tell which version of the training code produced a checkpoint. Resolved
    once per run, not per attempt: the environment is rebuilt on every retry
    and shelling out to git there would add a subprocess to each one. An
    empty string when git is unavailable is honest; a made-up value is not.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def _prepare_scratch(settings: Settings) -> None:
    for name in ("tmp", "torch", "wandb", "huggingface", "cache"):
        (settings.outputs / "scratch" / name).mkdir(parents=True, exist_ok=True)


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _launch(stage: Stage, settings: Settings, state: dict, state_path: Path) -> bool:
    """Run one stage, retrying it; True when it ended with code zero."""
    argv = resolve_argv(stage.argv, settings)
    for attempt in range(1, settings.max_retries + 1):
        entry = {
            "status": "running",
            "attempt": attempt,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": " ".join(argv),
        }
        state["stages"][stage.name] = entry
        _save_state(state_path, state)
        # With the env prefix, because the bare command is NOT reproducible:
        # copied without TORCH_HOME it reads an empty hub cache and both ranks
        # race to download the same zip - the failure the warmup stage exists
        # to prevent, reintroduced by whoever pastes the log line to debug.
        print(f"[{stage.name}] intento {attempt}/{settings.max_retries}: "
              f"{_reproducible_command(argv, settings)}")

        completed = subprocess.run(
            argv, cwd=str(settings.repo), env=stage_environment(settings)
        )
        entry["returncode"] = completed.returncode
        entry["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry["status"] = "ok" if completed.returncode == 0 else "error"
        _save_state(state_path, state)

        if completed.returncode == 0:
            return True
        print(f"[{stage.name}] salio con codigo {completed.returncode}")
        print(f"[{stage.name}] el motivo esta en el stderr del subproceso; si el log "
              f"no lo trae, relanza con 2>&1. El comando de arriba incluye su "
              f"entorno: copialo entero, sin las variables no es reproducible")
    return False


# What a real outputs root holds. Passing salidas/tone instead of salidas is
# the mistake this catches: that directory exists, so an existence check waves
# it through, and every path becomes salidas/tone/tone/... - every stage reads
# as pending and the run retrains everything with an air of normality.
PIPELINE_ENTRIES = ("tone", "final", "scratch", STATE_FILE)


def _check_looks_like_a_pipeline_root(settings: Settings) -> None:
    """Existence is not correspondence.

    A wrong path that happens to exist passes an existence check. This asks
    the stronger question - does this directory look like the root of this
    pipeline - which a subdirectory of it does not.
    """
    if not settings.outputs.is_dir():
        return
    present = {entry.name for entry in settings.outputs.iterdir()}
    if not present:
        return
    if present & set(PIPELINE_ENTRIES):
        return
    raise SystemExit(
        f"{settings.outputs} no parece la raiz de salidas del pipeline: no "
        f"contiene ninguno de {list(PIPELINE_ENTRIES)}, y si contiene "
        f"{sorted(present)[:4]}.\n"
        "Si apuntaste a un subdirectorio (por ejemplo .../salidas/tone en vez "
        "de .../salidas), todas las etapas van a leerse como pendientes y la "
        "corrida reentrenaria todo."
    )


# A stage marked running for longer than any stage can take is not running.
STALE_RUNNING_HOURS = 24.0


def _hours_since(stamp: str) -> float | None:
    """Age of a state timestamp, or None when it cannot be read.

    An unparseable stamp must not silence the warning: unknown age is not
    young. Returning None makes the caller treat it as stale.
    """
    try:
        started = time.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None
    return (time.time() - time.mktime(started)) / 3600.0


def _warn_about_stale_running(settings: Settings, state: dict) -> None:
    """Say out loud when the state file records a corpse.

    The other diagonal of the artefact check: `ok` with no artefact means a
    wrong tree, and `running` from twelve hours ago means an orchestrator
    that died mid-stage. Nothing notices the second one, because the entry
    is never rewritten - the process that would have closed it is gone. The
    run continues, since the artefact decides completion and not this file,
    but a reader deserves to know the history is lying.
    """
    stages = state.get("stages") or {}
    by_name = {stage.name: stage for stage in all_stages(settings)}
    for name, entry in sorted(stages.items()):
        if entry.get("status") != "running":
            continue
        started = entry.get("started_at", "")
        age = _hours_since(started)
        stage = by_name.get(name)
        if stage is not None and stage.done_when.is_file():
            print(f"[{name}] el estado dice 'running' desde {started} pero su "
                  f"artefacto existe: la etapa termino y nadie cerro el registro")
            continue
        if age is not None and age < STALE_RUNNING_HOURS:
            continue          # puede ser una corrida viva de verdad
        print(f"[{name}] ADVERTENCIA: el estado dice 'running' desde {started} y "
              f"no hay artefacto. Ninguna etapa tarda tanto: no es una corrida "
              f"en curso, es un orquestador que murio a mitad sin cerrar el registro")


def check_outputs_tree(settings: Settings, state: dict) -> None:
    """Refuse to run when the outputs tree is not the one that holds the work.

    Completion is decided by "does this artefact exist", so a wrong or
    unmounted --outputs makes every stage look pending and the orchestrator
    cheerfully retrains forty-eight GPU-hours. Absence and never-ran are
    indistinguishable from the inside; the state file is what tells them
    apart, because it lists stages that did finish.
    """
    _check_looks_like_a_pipeline_root(settings)
    if not settings.outputs.exists():
        raise SystemExit(
            f"no existe el directorio de salidas: {settings.outputs}\n"
            "Si es una corrida nueva, crealo explicitamente. Si no, la ruta esta "
            "mal o el volumen no esta montado: seguir reentrenaria todo desde cero."
        )
    _warn_about_stale_running(settings, state)
    done_before = [
        name for name, entry in (state.get("stages") or {}).items()
        if entry.get("status") == "ok"
    ]
    if not done_before:
        return
    by_name = {stage.name: stage for stage in all_stages(settings)}
    missing = [
        name for name in done_before
        if name in by_name and not by_name[name].done_when.is_file()
    ]
    if missing:
        raise SystemExit(
            f"el estado dice que estas etapas terminaron, pero su artefacto no esta: "
            f"{sorted(missing)}\n"
            f"Revisa {settings.outputs}: o es la ruta equivocada, o se borraron. "
            "No sigo, porque reentrenar lo que ya corrio cuesta horas de GPU."
        )


def run(settings: Settings) -> dict:
    """Run what is left of the pipeline; returns what was launched and what failed."""
    stages = build_stages(settings)
    if not settings.code_revision:
        settings.code_revision = detect_code_revision(settings.repo)
    check_outputs_tree(settings, _load_state(settings.outputs / STATE_FILE))
    if not settings.dry_run:
        _prepare_scratch(settings)
    state_path = settings.outputs / STATE_FILE
    state = _load_state(state_path)
    launched: list[str] = []
    failed: list[str] = []

    for stage in stages:
        if is_done(stage):
            print(f"[{stage.name}] ya esta: {stage.done_when}")
            continue
        if settings.dry_run:
            preview = " ".join(_preview_argv(stage, settings))
            print(f"[{stage.name}] pendiente ({stage.note})\n    {preview}")
            continue
        if _launch(stage, settings, state, state_path):
            launched.append(stage.name)
        else:
            failed.append(stage.name)
            print(f"[{stage.name}] se detiene la cadena; las etapas siguientes quedan pendientes")
            break

    return {"dry_run": settings.dry_run, "launched": launched, "failed": failed}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", type=Path, required=True, help="Checkout root (holds scripts/ and plan-segmentacion/)")
    parser.add_argument("--outputs", type=Path, required=True, help="Outputs root, outside the checkout")
    parser.add_argument("--ham-root", type=Path, required=True, help="HAM10000 directory with its metadata CSV")
    parser.add_argument("--python", default=sys.executable, help="Interpreter that runs every stage")
    parser.add_argument("--budget-gpu-hours", type=float, default=12.0, help="GPU-hour budget per pretraining arm")
    parser.add_argument("--nproc-per-node", type=int, default=2, help="GPUs per arm")
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", default="dermssl-segmentacion")
    parser.add_argument("--wandb-group", default="etapa2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--archs", nargs="+", default=list(DEFAULT_ARCHS),
        help="torch.hub DINOv2 names; one arm pair per architecture, compared at equal compute",
    )
    parser.add_argument("--only", default=None, help="Run a single stage by name")
    parser.add_argument("--max-retries", type=int, default=2, help="Attempts per stage before stopping the chain")
    parser.add_argument("--dry-run", action="store_true", help="Print what is pending and exit")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    settings = Settings(
        repo=args.repo, outputs=args.outputs, python=args.python, ham_root=args.ham_root,
        budget_gpu_hours=args.budget_gpu_hours, nproc_per_node=args.nproc_per_node,
        batch_size_per_gpu=args.batch_size_per_gpu, num_workers=args.num_workers,
        wandb_project=args.wandb_project, wandb_group=args.wandb_group, seeds=args.seeds,
        archs=args.archs, only=args.only, dry_run=args.dry_run, max_retries=args.max_retries,
    )
    result = run(settings)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
