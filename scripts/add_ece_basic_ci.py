"""Add the secondary ECE basic interval to an existing single-evaluation JSON.

It is computed from the stored point estimates and percentile bounds only, so
validation is not read again. The block is added once and never overwritten.

    python scripts/add_ece_basic_ci.py resultados/clasificacion/v7/final_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import final_eval as fe  # noqa: E402

KEY = "secondary_ece_basic_ci"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", type=Path)
    args = parser.parse_args(argv)
    doc = json.loads(args.document.read_text(encoding="utf-8"))
    if KEY in doc:
        print(f"{args.document} ya tiene {KEY}; no se sobrescribe")
        return 1
    doc[KEY] = fe.ece_basic_ci(doc)
    args.document.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"P2 con IC basic: {doc[KEY]['P2_with_basic']} (primaria: {doc['predictions']['P2']['verdict']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
