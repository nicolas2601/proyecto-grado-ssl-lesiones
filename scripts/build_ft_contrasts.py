"""Paired contrasts of B71-limpio against the ISIC 2020 fine-tuned baselines (writes JSON; ``--check`` is cheap).

    python scripts/build_ft_contrasts.py            # compute and write the JSON of every model with an npz
    python scripts/build_ft_contrasts.py --check    # exit 1 if an npz has no contrast JSON or a stale one

The bootstrap is slow, so the JSON is committed and the tables only read it, like the isic_eval contrasts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.bench import ft_contrasts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None, help="where to write (tests use a tmp dir)")
    args = parser.parse_args(argv)
    if args.check:
        bad = ft_contrasts.stale_models(REPO, args.out_root)
        for model in bad:
            print(f"ft contrast of {model} is missing or stale: run scripts/build_ft_contrasts.py", file=sys.stderr)
        return 1 if bad else 0
    done = ft_contrasts.write_all(REPO, args.out_root)
    print(f"wrote {len(done)} contrast files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
