"""The v7 final recipe of A* and B*, as the seed-42 and seed-43 finals ran it.

Read from the Weights & Biases config of the four finals (runs 41f0e8d164a3,
8f1684ceea84, 4672e3ff0604, b8e85ebc8ece; project dermssl-segmentacion,
group v7-final): the launcher on the pod lived only in ``v7up/finals.sh``,
so a replica had to be rebuilt from memory. Here it is data, and a test
checks it rebuilds the s43 config field by field. Only the seed, the output
folder and the declared ``num_workers`` change between replicas.
"""

from __future__ import annotations

from pathlib import Path

ARMS = ("A", "B")
BASE_SEED = 42  # Its folder has no seed suffix: final_A, final_B.
RECIPE_NUM_WORKERS = 0  # What s42 and s43 ran; any other value is a declared deviation.
SPLIT_HASH = "a83f927ab798b6f7a0614c3923fda77d082f887d10fdbe73416e34e9574ba71a"
TONE_MANIFEST_REVISION = "a754961bfd97bee21b4e48edd35b71314ed82b50cc6355480252b4d075e60e51"
WANDB_PROJECT, WANDB_GROUP = "dermssl-segmentacion", "v7-final"

COMMON = (
    ("--method", "simsiam_dinov2"), ("--backbone", "dinov2_vitb14"),
    ("--split-hash", SPLIT_HASH), ("--data-revision", "pool-final"),
    ("--freeze-blocks", "8"), ("--layer-decay", "0.75"), ("--hue-jitter", "0.02"),
    ("--anchor-l2sp", "0.0001"), ("--anchor-distill", "0"), ("--head-warmup-steps", "150"),
    ("--max-steps", "5700"), ("--batch-size-per-gpu", "128"), ("--precision", "bf16"),
    ("--checkpoint-every-steps", "500"),
    ("--wandb-project", WANDB_PROJECT), ("--wandb-group", WANDB_GROUP),
)
# The clock cap differs per arm and is part of the recipe: B is slower per step.
BUDGET_GPU_HOURS = {"A": "8", "B": "12"}
TONE_CONSISTENCY_WEIGHT = "0.1"  # B1 won wave 2 (docs/28 4.1).
CHECKPOINT_FRACTIONS = ("0.25", "0.5", "0.75", "1.0")


def final_dir(salidas: Path, arm: str, seed: int) -> Path:
    """``<salidas>/v7/final_A`` for the base seed, ``final_A_s43`` otherwise."""
    _check_arm(arm)
    suffix = "" if seed == BASE_SEED else f"_s{seed}"
    return Path(salidas) / "v7" / f"final_{arm}{suffix}"


def pretrain_argv(arm: str, seed: int, *, salidas: Path, num_workers: int = RECIPE_NUM_WORKERS,
                  resume: bool = False) -> list[str]:
    """Arguments of ``scripts/pretrain.py`` for one final."""
    _check_arm(arm)
    salidas = Path(salidas)
    argv = [flag for pair in COMMON for flag in pair]
    argv += ["--image-root", str(salidas / "tone" / "inputs" / "originals"),
             "--exclude-ids-file", str(salidas / "c1" / "selection" / "pretrain_exclude_ids_finals.txt"),
             "--output-dir", str(final_dir(salidas, arm, seed)),
             "--budget-gpu-hours", BUDGET_GPU_HOURS[arm],
             "--checkpoint-fractions", *CHECKPOINT_FRACTIONS,
             "--num-workers", str(num_workers), "--seed", str(seed)]
    if arm == "B":
        shift = salidas / "tone" / "shift2"
        argv += ["--transformed-image-root", str(shift),
                 "--transformed-manifest-path", str(shift / "tone_transform_manifest.json"),
                 "--transformed-data-revision", TONE_MANIFEST_REVISION,
                 "--tone-consistency-weight", TONE_CONSISTENCY_WEIGHT, "--grad-checkpoint"]
    if resume:
        argv.append("--resume")
    return argv


def deviations(num_workers: int) -> list[str]:
    """What this launch changes from the s42/s43 recipe, besides seed and folder."""
    if num_workers == RECIPE_NUM_WORKERS:
        return []
    return [f"num_workers={num_workers} (receta: {RECIPE_NUM_WORKERS}); las tiradas de torch de las "
            "aumentaciones no son las de la receta (docs/29 §2.5)"]


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"brazo {arm!r}; se esperaba uno de {ARMS}")
