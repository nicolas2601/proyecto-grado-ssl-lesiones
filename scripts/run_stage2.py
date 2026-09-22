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
"""

from __future__ import annotations

import argparse
import json
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
STATE_FILE = "stage2_state.json"
BACKBONE_FILE = "simsiam_dinov2_backbone_final.pt"
METHOD = "simsiam_dinov2"
ARCH = "dinov2_vits14"


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
    only: str | None = None
    dry_run: bool = False
    max_retries: int = 2


@dataclass
class Stage:
    name: str
    argv: list[str]
    done_when: Path
    note: str = ""


def _script(settings: Settings, *parts: str) -> str:
    return str(Path(settings.repo, *parts))


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


def _pretrain_argv(settings: Settings, output_dir: Path) -> list[str]:
    return [
        settings.python, "-m", "torch.distributed.run",
        f"--nproc_per_node={settings.nproc_per_node}",
        _script(settings, "scripts", "pretrain.py"),
        "--method", METHOD,
        "--image-root", str(settings.outputs / "tone" / "inputs" / "originals"),
        "--output-dir", str(output_dir),
        "--budget-gpu-hours", str(settings.budget_gpu_hours),
        "--batch-size-per-gpu", str(settings.batch_size_per_gpu),
        "--num-workers", str(settings.num_workers),
        "--precision", "bf16",
        *_fractions_argv(),
        "--wandb-project", settings.wandb_project,
        "--wandb-group", settings.wandb_group,
    ]


def _plain_stage(settings: Settings) -> Stage:
    output_dir = settings.outputs / "tone" / "ssl_plain"
    return Stage(
        name="ssl_plain",
        argv=_pretrain_argv(settings, output_dir),
        done_when=output_dir / BACKBONE_FILE,
        note="control arm: two views of the same image",
    )


def _proposed_stage(settings: Settings) -> Stage:
    output_dir = settings.outputs / "tone" / "ssl_proposed"
    shift = settings.outputs / "tone" / "shift2"
    return Stage(
        name="ssl_proposed",
        argv=[
            *_pretrain_argv(settings, output_dir),
            "--transformed-image-root", str(shift),
            "--transformed-manifest-path", str(shift / "tone_transform_manifest.json"),
            "--data-revision", SOURCE_REVISION,
        ],
        done_when=output_dir / BACKBONE_FILE,
        note="proposed arm: the original against its tone-shifted copy",
    )


def _bank_stage(settings: Settings) -> Stage:
    bank = settings.outputs / "tone" / "bank"
    tone = settings.outputs / "tone"
    return Stage(
        name="bank",
        argv=[
            settings.python,
            _script(settings, "scripts", "cluster_frozen_bank.py"),
            "--ham-root", str(settings.ham_root),
            "--backbone", f"dinov2_public={ARCH}:public",
            "--backbone", f"ssl_plain={ARCH}:{tone / 'ssl_plain' / BACKBONE_FILE}",
            "--backbone", f"ssl_proposed={ARCH}:{tone / 'ssl_proposed' / BACKBONE_FILE}",
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


BUILDERS = {
    "export": _export_stage,
    "tone_pairs": _tone_pairs_stage,
    "ssl_plain": _plain_stage,
    "ssl_proposed": _proposed_stage,
    "bank": _bank_stage,
}


def build_stages(settings: Settings) -> list[Stage]:
    """Every stage in order, or the single one named by ``--only``."""
    if settings.only is None:
        return [builder(settings) for builder in BUILDERS.values()]
    if settings.only not in BUILDERS:
        raise ValueError(f"etapa desconocida: {settings.only!r}; use {sorted(BUILDERS)}")
    return [BUILDERS[settings.only](settings)]


def is_done(stage: Stage) -> bool:
    """A stage is finished when the artefact it promises exists."""
    return stage.done_when.is_file()


def pending_stages(stages: list[Stage]) -> list[Stage]:
    return [stage for stage in stages if not is_done(stage)]


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _launch(stage: Stage, settings: Settings, state: dict, state_path: Path) -> bool:
    """Run one stage, retrying it; True when it ended with code zero."""
    for attempt in range(1, settings.max_retries + 1):
        entry = {
            "status": "running",
            "attempt": attempt,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": " ".join(stage.argv),
        }
        state["stages"][stage.name] = entry
        _save_state(state_path, state)
        print(f"[{stage.name}] intento {attempt}/{settings.max_retries}: {' '.join(stage.argv)}")

        completed = subprocess.run(stage.argv, cwd=str(settings.repo))
        entry["returncode"] = completed.returncode
        entry["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry["status"] = "ok" if completed.returncode == 0 else "error"
        _save_state(state_path, state)

        if completed.returncode == 0:
            return True
        print(f"[{stage.name}] salio con codigo {completed.returncode}")
    return False


def run(settings: Settings) -> dict:
    """Run what is left of the pipeline; returns what was launched and what failed."""
    stages = build_stages(settings)
    state_path = settings.outputs / STATE_FILE
    state = _load_state(state_path)
    launched: list[str] = []
    failed: list[str] = []

    for stage in stages:
        if is_done(stage):
            print(f"[{stage.name}] ya esta: {stage.done_when}")
            continue
        if settings.dry_run:
            print(f"[{stage.name}] pendiente ({stage.note})\n    {' '.join(stage.argv)}")
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
    parser.add_argument("--only", default=None, help=f"Run a single stage: {sorted(BUILDERS)}")
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
        only=args.only, dry_run=args.dry_run, max_retries=args.max_retries,
    )
    result = run(settings)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
