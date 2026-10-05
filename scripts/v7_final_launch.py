"""Launch one v7 final (A* or B*) with the s42/s43 recipe; only seed and folder change.

The recipe is ``dermssl.pretrain.v7_finals``, read from the W&B config of the
finals. This script builds the ``scripts/pretrain.py`` command, prints it with
any declared deviation, and replaces itself with it. Export ``GIT_COMMIT``
and pin the GPU yourself; ``pretrain.py`` refuses a missing commit.

    GIT_COMMIT=$(git rev-parse HEAD) CUDA_VISIBLE_DEVICES=0 \\
        python scripts/v7_final_launch.py --arm A --seed 44 \\
        --salidas ~/work/proyectonp/salidas --num-workers 4
    python scripts/v7_final_launch.py --arm B --seed 44 --salidas ... --print-only
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.pretrain import v7_finals as vf  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True, choices=vf.ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--salidas", required=True, type=Path, help="raiz de salidas del pod")
    parser.add_argument("--num-workers", type=int, default=vf.RECIPE_NUM_WORKERS,
                        help=f"receta: {vf.RECIPE_NUM_WORKERS}; otro valor se declara como desviacion")
    parser.add_argument("--resume", action="store_true", help="reanudar desde resume.pt de esa final")
    parser.add_argument("--print-only", action="store_true", help="imprime el comando y no lo corre")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    command = [sys.executable, str(REPO / "scripts" / "pretrain.py"),
               *vf.pretrain_argv(args.arm, args.seed, salidas=args.salidas.expanduser(),
                                 num_workers=args.num_workers, resume=args.resume)]
    for note in vf.deviations(args.num_workers):
        print(f"DESVIACION: {note}")
    print(shlex.join(command), flush=True)
    if args.print_only:
        return 0
    os.execv(command[0], command)
    return 1  # execv does not return


if __name__ == "__main__":
    sys.exit(main())
