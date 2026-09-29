"""Fill the BLAS-threads limitation region of docs/31 from the committed diagnostics; ``--check`` for CI.

    python scripts/build_blas_limitation_note.py [--doc docs/31-t4-shades-of-gray.md] [--check]

The region between ``<!-- limitacion-hilos-blas:inicio -->`` and ``:fin -->``
sits after the registered T4 block, which this script never touches. The text is
the same one ``build_v71_table.py`` appends to the v7.1 README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import probe_blas_limitation as pbl  # noqa: E402
from dermssl import t4_sog  # noqa: E402

T4_DIR = REPO / "resultados" / "clasificacion" / "t4"
V71_EVAL = REPO / "resultados" / "clasificacion" / "v71" / "v71_final_eval.json"


def facts(t4_dir: Path = T4_DIR, v71_eval: Path = V71_EVAL) -> dict:
    return pbl.load_facts(t4_dir, json.loads(v71_eval.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--doc", type=Path, default=REPO / t4_sog.REGISTRATION_DOC)
    parser.add_argument("--t4", type=Path, default=T4_DIR)
    parser.add_argument("--v71", type=Path, default=V71_EVAL)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = args.doc.read_text(encoding="utf-8")
    text = pbl.fill_region(current, pbl.body(facts(args.t4, args.v71)))
    if args.check:
        if current != text:
            print(f"{args.doc}: la región {pbl.REGION_START} está desactualizada; regenerar")
            return 1
        return 0
    args.doc.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
