"""PAD-UFES-20 WiSE-FT alpha curve of A* (docs/30 8). Single run, on the pod.

For A* s42, A* s43 and their soup, the weights are interpolated between public
and the final as v7.1 does (``v71_io.WeightBank``) at alpha in {0, 0.2, ..., 1};
alpha = 0 is exactly public, extracted and fitted once for the three curves.
Each point is read with the first pass's protocol (docs/30 2): the C1 ``logreg``
on CLS at 224, no TTA, fitted on the whole C1 fold-0 train of HAM10000 with the
bank's extraction, and applied to PAD without seeing it. Every fit runs under
``probe_threads.PROBE_BLAS_THREADS``. Readings, strata and PAD-W1/W2 are
``dermssl.pad_ufes_wiseft``; the HAM side is the committed v7.1 holdout curve,
so no HAM image is scored anew and val is never read.

Before touching any image it refuses unless the docs/30 8 block was committed
before this script and is unchanged, the tree is clean, ``--single-evaluation``
is given, the output does not exist, the finals' run.json agree (revision,
exclude list, seed), and every PAD image matches its manifest sha256.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import v71_io as io  # noqa: E402
import v71_val  # noqa: E402

from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import pad_ufes_readings as pr  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402

#: The commit that registered docs/30 8 on its own, before any WiSE-FT code.
REGISTRATION_COMMIT = "dc36db6bda7111460a65601b0d39d27d60dda9af"
BLOCK = ("<!-- registro-pad-wiseft:inicio -->", "<!-- registro-pad-wiseft:fin -->")
THIS_SCRIPT = "scripts/pad_ufes_wiseft.py"
HAM_GRID = REPO / "resultados" / "clasificacion" / "v71" / "holdout_grid.json"
#: run.json name in ``WeightBank.run_manifests`` -> the pretraining seed it must carry.
FINAL_SEEDS = {"A_s42": 42, "A_s43": 43}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--single-evaluation", action="store_true", help="required: PAD is read once")
    for flag in ("--manifest", "--pad-root", "--ham-root", "--split-index", "--holdout", "--exclude-ids-file",
                 "--weights-dir", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--ham-grid", type=Path, default=HAM_GRID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    # 0: the pod's /dev/shm is 64 MB and DataLoader workers crashed there with 8.
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def check_registration() -> str:
    return pg.check_registration(REPO, commit=REGISTRATION_COMMIT, doc=pg.REGISTRATION_DOC,
                                 script=THIS_SCRIPT, start=BLOCK[0], end=BLOCK[1])


def preflight(args: argparse.Namespace) -> tuple[str, str]:
    """Everything that can refuse, before any data is read."""
    if not args.single_evaluation:
        raise SystemExit("ABORT: exige --single-evaluation (PAD se lee una vez)")
    if args.out.expanduser().exists():
        raise SystemExit(f"ABORT: {args.out} ya existe; la evaluacion no se repite")
    try:
        registration = check_registration()
        return registration, rp.clean_revision(REPO)
    except (pg.RegistrationError, rp.DirtyTreeError) as exc:
        raise SystemExit(f"ABORT: {exc}") from exc


def check_seeds(runs: dict) -> None:
    for name, seed in FINAL_SEEDS.items():
        if runs.get(name, {}).get("seed") != seed:
            raise SystemExit(f"ABORT: {name}: el run.json dice semilla {runs.get(name, {}).get('seed')}, no {seed}")


def context_args(args: argparse.Namespace) -> argparse.Namespace:
    """What ``v71_io.load_context`` reads: the C1 split, its index and the docs/28 holdout (ids only)."""
    return argparse.Namespace(ham_root=args.ham_root, image_dir=args.ham_root, split_index=args.split_index,
                              holdout=args.holdout, limit=None)


def check_inputs(args: argparse.Namespace) -> dict:
    """Split, finals and their run.json, all checked before the first forward pass."""
    ctx = io.load_context(context_args(args))
    if tuple(ctx["classes"]) != pr.HAM_CLASSES:
        raise SystemExit(f"ABORT: clases de HAM {ctx['classes']} != {pr.HAM_CLASSES}")
    weights = io.WeightBank(args.weights_dir.expanduser())
    guard = v71_val._guard_val(args, ctx, weights)  # C1 fingerprints, one revision, the finals' exclude list
    check_seeds(guard["runs"])
    return {"ctx": ctx, "weights": weights, "guard": guard}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preflight(args)
    check_inputs(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
