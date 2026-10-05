"""Build resultados/banco_completo/FITZPATRICK.md (docs/33 E5) from the per-dataset E5 JSON; ``--check`` for CI.

    python scripts/build_fitzpatrick_table.py            # write
    python scripts/build_fitzpatrick_table.py --check    # fails if the file is missing or differs

Every number is read from ``resultados/banco_completo/e5/e5_fitzpatrick_<dataset>.json``; a dataset
without its JSON is rendered as pending. The table states, before any number, that the ITA estimate
is not validated and that the L* terciles are luminance relative to each dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import e5_table as tab  # noqa: E402

DEFAULT_INPUT = REPO / "resultados" / "banco_completo" / "e5"
DEFAULT_OUT = REPO / "resultados" / "banco_completo" / "FITZPATRICK.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the table is missing or stale")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    text = tab.render(tab.load_documents(args.input_dir))
    if args.check:
        fresh = args.out.is_file() and args.out.read_text(encoding="utf-8") == text
        if not fresh:
            print(f"{args.out} is stale or missing: run scripts/build_fitzpatrick_table.py", file=sys.stderr)
        return 0 if fresh else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
