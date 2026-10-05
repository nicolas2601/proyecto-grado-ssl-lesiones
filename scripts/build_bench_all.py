"""Run every generator of resultados/banco_completo/ (tables, ablation, contrasts, confusion, LaTeX).

    python scripts/build_bench_all.py            # write everything
    python scripts/build_bench_all.py --check    # exit 1 if ANY generated file is stale

Each generator reads only committed JSON and has its own ``--check``; this entry point runs them
all so one command (and one test) covers the whole bench. Pending cells turn into measured ones
by themselves when their JSON lands: re-run this script and commit what it writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_bench_ablation  # noqa: E402
import build_bench_contrasts  # noqa: E402
import build_bench_tables  # noqa: E402

GENERATORS = (("build_bench_tables.py", build_bench_tables), ("build_bench_ablation.py", build_bench_ablation),
              ("build_bench_contrasts.py", build_bench_contrasts))


def _argv(args: argparse.Namespace, module) -> list[str]:
    out = ["--check"] if args.check else []
    out += ["--out-root", str(args.out_root)] if args.out_root else []
    if module is build_bench_tables:
        out += ["--skip-png"] if args.skip_png else []
        out += ["--redraw-png"] if args.redraw_png and not args.check else []
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any generated file is stale")
    parser.add_argument("--skip-png", action="store_true", help="do not draw or require the confusion PNG")
    parser.add_argument("--redraw-png", action="store_true", help="redraw PNG that already exist (write mode)")
    parser.add_argument("--out-root", type=Path, help="output directory (tests use a tmp dir)")
    args = parser.parse_args(argv)
    failed = [name for name, module in GENERATORS if module.main(_argv(args, module)) != 0]
    for name in failed:
        print(f"{name}: FAILED" if not args.check else f"{name}: stale files", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
