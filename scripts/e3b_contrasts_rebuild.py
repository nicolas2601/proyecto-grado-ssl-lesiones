"""Rebuild the paired contrasts of E3b WITH MCC from the committed probabilities (CPU only, ~minutes).

    python scripts/e3b_contrasts_rebuild.py --ham-metadata <HAM10000_metadata.csv> \\
        [--dir resultados/banco_completo/e3b_bs64 --dir resultados/banco_completo/e3b]

Reads ``e3b_probas.npz`` (probabilities of every arm on val, with ``y_val`` and ``val_ids``) and
``e3b_summary.json``, recomputes the contrasts with the code path of the run
(``dermssl.e3b.contrasts.paired``, bootstrap seed 20260923, 1000 resamples, grouped by
``lesion_id``) and rewrites ``contrasts`` ONLY if every pre-existing record is reproduced
identically (the MCC records are added). Otherwise it aborts with rc=2 and writes nothing.
The sidecar is refreshed after the final file. No weights, images or GPU are touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.e3b import rebuild  # noqa: E402

DEFAULT_DIRS = ("resultados/banco_completo/e3b_bs64", "resultados/banco_completo/e3b")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ham-metadata", type=Path, required=True, help="HAM10000_metadata.csv (lesion_id per image_id)")
    p.add_argument("--dir", type=Path, action="append", dest="dirs", help="E3b result directory (repeatable)")
    p.add_argument("--n-resamples", type=int, default=1000)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for directory in args.dirs or [REPO / d for d in DEFAULT_DIRS]:
        try:
            report = rebuild.rebuild_dir(directory, args.ham_metadata.expanduser(), n_resamples=args.n_resamples)
        except rebuild.RebuildMismatchError as exc:
            print(f"ABORT {directory}: {exc}")
            return 2
        print(f"[ok] {directory}: {report['equal']}/{report['compared']} previous records identical, "
              f"{report['mcc_added']} MCC records added")
    return 0


if __name__ == "__main__":
    sys.exit(main())
