"""D2 and D3 of docs/33 E2: SimSiam-anchored DINOv2 continued from A* s42 clean.

D2 continues on the HAM/ISIC SSL pool only (compute control); D3 adds the PAD
``target_adapt`` images (images only, no labels). They must differ ONLY in the
image root (and so in the output folder): same recipe, init, seed and exclusion.
Here the recipe is data, the roots are symlink trees and the "only the pool
differs" property is a checked function, not a promise.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from dermssl.pretrain import config as config_mod

VARIANTS = ("D2", "D3")
POOL_DIR, PAD_DIR = "pool", "pad_target_adapt"
SEED = 42
POOL_ONLY_FLAGS = frozenset({"--image-root", "--output-dir"})
WANDB_PROJECT, WANDB_GROUP = "dermssl-segmentacion", "bench-clean-d"
CHECKPOINT_FRACTIONS = ("0.25", "0.5", "0.75", "1.0")
RECIPE = (
    ("--method", "simsiam_dinov2"), ("--backbone", "dinov2_vitb14"),
    ("--freeze-blocks", "8"), ("--layer-decay", "0.75"), ("--hue-jitter", "0.02"),
    ("--anchor-l2sp", "0.0001"), ("--anchor-distill", "0"), ("--head-warmup-steps", "150"),
    ("--max-steps", "2000"), ("--checkpoint-every-steps", "500"), ("--seed", str(SEED)),
    ("--batch-size-per-gpu", "128"), ("--num-workers", "0"), ("--precision", "bf16"),
    ("--wandb-project", WANDB_PROJECT), ("--wandb-group", WANDB_GROUP),
)


class ControlDriftError(ValueError):
    """D2 and D3 differ in something other than the pool."""


class PoolCollisionError(ValueError):
    """A PAD image id is also the name of a pool image."""


class RootExistsError(FileExistsError):
    """An image root of this launcher already exists; it is never reused."""


def pretrain_argv(variant: str, *, backbone_weights: Path, image_root: Path, output_dir: Path,
                  exclude_ids_file: Path) -> list[str]:
    """Arguments of ``scripts/pretrain.py`` for D2 or D3."""
    if variant not in VARIANTS:
        raise ValueError(f"variante {variant!r}; se esperaba una de {VARIANTS}")
    argv = [part for pair in RECIPE for part in pair]
    argv += ["--initial-weights", str(backbone_weights), "--exclude-ids-file", str(exclude_ids_file),
             "--image-root", str(image_root), "--output-dir", str(output_dir),
             "--checkpoint-fractions", *CHECKPOINT_FRACTIONS]
    return argv


def _flag_values(argv: list[str]) -> dict[str, tuple[str, ...]]:
    flags: dict[str, list[str]] = {}
    current = None
    for token in argv:
        if token.startswith("--"):
            current = token
            flags[current] = []
        elif current is not None:
            flags[current].append(token)
    return {flag: tuple(values) for flag, values in flags.items()}


def differing_flags(a: list[str], b: list[str]) -> set[str]:
    """Flags present on one side only or with another value."""
    left, right = _flag_values(a), _flag_values(b)
    return {flag for flag in left.keys() | right.keys() if left.get(flag) != right.get(flag)}


def check_only_pool_differs(d2: list[str], d3: list[str]) -> None:
    extra = differing_flags(d2, d3) - POOL_ONLY_FLAGS
    if extra:
        raise ControlDriftError(f"D2 y D3 difieren en {sorted(extra)}; solo pueden diferir en {sorted(POOL_ONLY_FLAGS)}")


def command_line(argv: list[str]) -> str:
    """Shell line for the pod; ``pretrain.py`` refuses a launch without ``GIT_COMMIT``."""
    return "GIT_COMMIT=$(git rev-parse HEAD) python scripts/pretrain.py " + shlex.join(argv)


def _check_no_collision(pool: Path, pad_ids: list[str]) -> None:
    clash = sorted({p.stem for p in config_mod.collect_images(pool)} & set(pad_ids))
    if clash:
        raise PoolCollisionError(f"{len(clash)} ids de PAD ya existen en el pool, p. ej. {clash[:3]}")


def _make_root(root: Path, pool: Path, pad: dict[str, Path] | None) -> Path:
    if root.exists():
        raise RootExistsError(f"{root} ya existe; el raiz de imagenes no se reutiliza")
    root.mkdir(parents=True)
    (root / POOL_DIR).symlink_to(pool, target_is_directory=True)
    if pad:
        (root / PAD_DIR).mkdir()
        for img_id, path in sorted(pad.items()):
            (root / PAD_DIR / f"{img_id}{path.suffix.lower()}").symlink_to(path)
    return root


def build_roots(pool_root: Path, pad_images: dict[str, Path], work_dir: Path) -> dict[str, Path]:
    """Symlink trees ``{"D2": pool, "D3": pool + target_adapt}`` under ``work_dir``; links only, no copies."""
    pool = Path(pool_root).resolve(strict=True)
    if not pad_images:
        raise ValueError("target_adapt sin imagenes: D3 seria igual a D2")
    _check_no_collision(pool, list(pad_images))
    work = Path(work_dir)
    return {"D2": _make_root(work / "d2_root", pool, None),
            "D3": _make_root(work / "d3_root", pool, {i: Path(p).resolve(strict=True) for i, p in pad_images.items()})}
