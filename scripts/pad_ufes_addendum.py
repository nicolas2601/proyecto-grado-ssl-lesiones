"""PAD-UFES-20 descriptive addendum of docs/30 7: A71 and seed 44. Single run, on the pod.

Two model families, each on the protocol it was frozen with, both fitted on
the whole C1 fold-0 train of HAM10000 and applied to PAD without seeing it:

* A71, the v7.1 winner for A (``selection.json``, pinned by sha256), and public
  on the same protocol: soup s42+s43, ``cls_attn``, ``la0.5``, D4 TTA, by the
  v7.1 val-read code (``v71_io`` weights and inputs, ``torch_v71.extract_split``,
  ``v71_grid.unit_probas``). The train inputs are v7.1's cached array.
* A* and B* of seed 44, and public, on the docs/30 2 protocol (``logreg`` on
  CLS at 224), as ``pad_ufes_eval`` reads s42/s43.

Readings are ``dermssl.pad_ufes_addendum``: the first pass's, without any
verdict. Every probe fit runs under ``probe_threads.PROBE_BLAS_THREADS``.

Before touching any data it refuses unless the docs/30 7 block was committed
before this script and is unchanged, the tree is clean, ``--single-evaluation``
is given, the output does not exist, the selection matches its sha256, the
v7.1 train inputs are cached, and (later) every PAD image matches its manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import v71_io as io  # noqa: E402

from dermssl import feature_cache as fc  # noqa: E402
from dermssl import pad_ufes_addendum as pa  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import t4_sog  # noqa: E402
from dermssl import v71_plan as plan  # noqa: E402
from dermssl import v71_protocol as vp  # noqa: E402

#: The commit that registered docs/30 7 on its own, before any addendum code.
REGISTRATION_COMMIT = "76aa91e7b68cbfd55bd93aa857f6587efd5717b7"
BLOCK = ("<!-- registro-pad-adenda-a71-s44:inicio -->", "<!-- registro-pad-adenda-a71-s44:fin -->")
THIS_SCRIPT = "scripts/pad_ufes_addendum.py"
SELECTION = REPO / "resultados" / "clasificacion" / "v71" / "selection.json"
RESOLUTION = vp.parse_config_id(pa.A71_CONFIG)["resolution"]
TRAIN_TAG = "train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--single-evaluation", action="store_true", help="required: PAD is read once")
    for flag in ("--manifest", "--pad-root", "--ham-root", "--split-index", "--holdout", "--exclude-ids-file",
                 "--weights-dir", "--v71-inputs", "--pad-cache", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--plain-s44", required=True, help="ARCH:SOURCE of A* seed 44")
    parser.add_argument("--proposed-s44", required=True, help="ARCH:SOURCE of B* seed 44")
    parser.add_argument("--public", default="dinov2_vitb14:public")
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--probe-jobs", type=int, default=2)
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


def check_selection(path: Path) -> dict:
    """``{A71: config, public_A71: config}`` from the frozen v7.1 selection, refused if it moved."""
    sha = rp.file_sha256(path)
    if sha != pa.SELECTION_SHA256:
        raise SystemExit(f"ABORT: {path} tiene sha256 {sha[:12]}, la adenda fijo {pa.SELECTION_SHA256[:12]}")
    entries = plan.val_entries(io.read_json(path))
    chosen = {pa.A71: entries["A71"], pa.PUBLIC_A71: entries["public_A71"]}
    expected = {pa.A71: pa.A71_CONFIG, pa.PUBLIC_A71: pa.PUBLIC_A71_CONFIG}
    if chosen != expected:
        raise SystemExit(f"ABORT: la seleccion nombra {chosen}, la adenda registro A71 = {expected}")
    return chosen


def train_inputs_path(inputs_dir: Path, train_ids) -> Path:
    """v7.1's decoded train array (``v71_io.ImageStore`` naming); never re-decoded here."""
    suffix = t4_sog.input_suffix(t4_sog.IDENTITY)
    path = inputs_dir / f"{TRAIN_TAG}{suffix}_{RESOLUTION}_{fc.fingerprint_ids(list(train_ids))}.npy"
    if not path.is_file():
        raise SystemExit(f"ABORT: falta {path}; A71 se ajusta sobre las entradas cacheadas de la lectura v7.1")
    return path


def check_s44_runs(runs: dict, exclude_sha: str) -> None:
    """A* and B* of seed 44: one revision, one exclude list (the finals' one), seed 44."""
    from dermssl import final_eval as fe  # noqa: PLC0415

    try:
        fe.check_runs_consistent(runs)
        fe.check_same_seed(runs)
    except ValueError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    for role, run in runs.items():
        if run.get("seed") != pa.SEED:
            raise SystemExit(f"ABORT: {role} es de la semilla {run.get('seed')}, no {pa.SEED}")
        if run.get("exclude_ids_sha256") != exclude_sha:
            raise SystemExit(f"ABORT: {role}: el run.json no uso el exclude {exclude_sha[:12]}")


def context_args(args: argparse.Namespace) -> argparse.Namespace:
    """What ``v71_io.load_context`` reads: the C1 split, its index and the docs/28 holdout."""
    return argparse.Namespace(ham_root=args.ham_root, image_dir=args.ham_root, split_index=args.split_index,
                              holdout=args.holdout, limit=None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preflight(args)
    check_selection(args.selection.expanduser())
    raise SystemExit("ABORT: la evaluacion todavia no esta implementada")


if __name__ == "__main__":
    sys.exit(main())
