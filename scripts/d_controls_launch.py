"""Build the D2 / D3 launch of docs/33 E2 and print the exact ``pretrain.py`` command lines.

D2 = A* s42 clean continued for 2000 steps on the SSL pool only (compute control).
D3 = the same, with the PAD ``target_adapt`` images added (images only: only the
id list is read from the split JSON, never a diagnostic, and never target_test).
Both start from ``--backbone-weights`` and differ only in ``--image-root`` and the
output folder; the check is printed.

    python scripts/d_controls_launch.py \
        --backbone-weights $S/bench/v7clean/final_A_s42/simsiam_dinov2_backbone_final.pt \
        --pool-root $S/tone/inputs/originals --pad-root ~/work/datos/pad_ufes_20 \
        --split ~/work/datos/pad_ufes_20/target_split.json \
        --exclude-ids-file $S/bench/pretrain_exclude_ids_clean.txt \
        --work-dir $S/bench/d_controls --runs-dir $S/bench/v7clean --init-weights-honored

BLOCKED BY DEFAULT: at the revision this was written, ``pretrain.py`` records
``--initial-weights`` in run.json but never loads it: SimSiam builds DINOv2 from
the public hub weights and the L2-SP anchor is a copy of THOSE. A launch from
here would train from public weights under the name "continued from A*".
``--init-weights-honored`` is the declared act that this was fixed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pad_ufes_download import IMAGES_DIR  # noqa: E402

from dermssl import pad_target_split as ts  # noqa: E402
from dermssl import pad_ufes_files as pf  # noqa: E402
from dermssl.pretrain import d_controls as dc  # noqa: E402

REFUSAL = ("ABORT: scripts/pretrain.py no carga --initial-weights (solo lo registra en run.json): D2/D3 "
           "arrancarian de DINOv2 publico y el ancla L2-SP apuntaria a el. Corrige el cargador y pasa "
           "--init-weights-honored.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--backbone-weights", "--pool-root", "--pad-root", "--split", "--exclude-ids-file",
                 "--work-dir", "--runs-dir"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--init-weights-honored", action="store_true",
                        help="declares that pretrain.py loads --initial-weights and anchors L2-SP to it")
    parser.add_argument("--dry-run", action="store_true", help="print only; create no symlink tree")
    return parser.parse_args(argv)


def _require_files(args: argparse.Namespace) -> None:
    for path in (args.backbone_weights, args.exclude_ids_file, args.split):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"no existe {path}")


def _roots(args: argparse.Namespace) -> dict[str, Path]:
    work = args.work_dir.expanduser()
    if args.dry_run:
        return {"D2": work / "d2_root", "D3": work / "d3_root"}
    ids = ts.read_target_adapt_ids(args.split.expanduser())
    pad = pf.resolve_images(ids, pf.index_images(args.pad_root.expanduser() / IMAGES_DIR))
    return dc.build_roots(args.pool_root.expanduser(), pad, work)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.init_weights_honored:
        print(REFUSAL, file=sys.stderr)
        return 2
    try:
        if not args.dry_run:
            _require_files(args)
        roots = _roots(args)
    except (OSError, ValueError) as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    argvs = {v: dc.pretrain_argv(v, backbone_weights=args.backbone_weights.expanduser(), image_root=roots[v],
                                 output_dir=args.runs_dir.expanduser() / f"{v}_s{dc.SEED}",
                                 exclude_ids_file=args.exclude_ids_file.expanduser()) for v in dc.VARIANTS}
    dc.check_only_pool_differs(argvs["D2"], argvs["D3"])
    for variant, argv_ in argvs.items():
        print(f"# {variant}\n{dc.command_line(argv_)}\n")
    print("# check: D2 and D3 differ only in --image-root, --output-dir: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
