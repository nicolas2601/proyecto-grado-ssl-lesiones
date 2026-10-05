r"""Summarize the E3 seed robustness: F4 vs B71 on the B* seeds 42, 43 and 44 (docs/33 errata).

Reads the three ``e3_final_eval.json`` as written by ``e3_run.py --stage final`` and writes
``e3_robustness.json`` (per-seed rows and paired CI as computed, mean/min/max delta, the
pre-registered ``reading`` and the sha256 of the three inputs). Nothing is recomputed::

    python scripts/e3_robustness_summary.py --s42 salidas/e3/e3_final_eval.json \
      --s43 salidas/e3_rob_s43/e3_final_eval.json --s44 salidas/e3_rob_s44/e3_final_eval.json \
      --selection-sha256 2446c05a18e6e76d16d9953d89a8d28547cde61a6c25cc64aa60fb92c0563777 \
      --out salidas/e3_robustness/e3_robustness.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import e3_robustness as rob  # noqa: E402
from dermssl import e3_seed as seed_map  # noqa: E402
from dermssl import e3_store as st  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for seed in seed_map.SEEDS:
        p.add_argument(f"--s{seed}", type=Path, required=True, help=f"e3_final_eval.json of B* seed {seed}")
    p.add_argument("--selection-sha256", default=None, help="the pinned selection; checked against the three inputs")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = {seed: getattr(args, f"s{seed}").expanduser() for seed in seed_map.SEEDS}
    try:
        docs = {seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in paths.items()}
        shas = {seed: hashlib.sha256(path.read_bytes()).hexdigest() for seed, path in paths.items()}
        doc = rob.summarize(docs, shas, selection_sha256=args.selection_sha256)
    except (OSError, ValueError, KeyError) as exc:  # RobustnessInputError is a ValueError
        print(f"ABORT: {type(exc).__name__}: {exc}")
        return 2
    st.write_json_atomic(args.out.expanduser(), doc)
    print(f"lectura: {doc['reading']} ({doc['n_excluding_zero']}/3 semillas con IC que excluye 0); "
          f"delta macro-F1 medio {doc['delta_macro_f1']['mean']:+.4f}\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
