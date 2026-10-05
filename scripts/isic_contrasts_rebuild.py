"""Regenerate ONLY the paired contrasts of the ISIC bank from the cells already on disk (docs/33 E1).

No dataset, weights, GPU or feature extraction: it reads ``<dataset-dir>/<cell>.json`` and
``<cell>.probas.npz`` and writes ``<dataset-dir>/contrasts/<arm>__vs__<control>.json`` plus its
environment sidecar, with the same pairs, seed and resamples as ``isic_eval_run.py``. The cell key
does not contain the code revision, so finished cells stay valid; a contrast is skipped only when its
key (probas sha256 of both cells, resamples, metric list) is unchanged, so contrasts written before
MCC joined the list are recomputed (the old file is kept as ``.stale-<sha8>``). Legacy ``.probas.npz``
with object arrays load only after their sha256 matches the one the cell registered.

    python scripts/isic_contrasts_rebuild.py --dataset-dir $O/isic2019 --dataset-dir $O/isic2020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import e3_store, probe_threads, run_environment  # noqa: E402
from dermssl.isic_eval import runner as rn  # noqa: E402
from dermssl.isic_eval import scoring  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True,
                        help="<out-dir>/<dataset> holding the cells; repeat for several")
    parser.add_argument("--ci-resamples", type=int, default=scoring.N_RESAMPLES)
    return parser.parse_args(argv)


def rebuild(dataset_dir: Path, args: argparse.Namespace) -> int:
    store, contrasts = e3_store.CellStore(dataset_dir), e3_store.CellStore(dataset_dir / "contrasts")
    done = rn.run_all_contrasts(store, contrasts, n_resamples=args.ci_resamples)
    for name, status in done.items():
        print(f"[{dataset_dir.name}/{name}] {status}")
        if status == "done":  # a recomputed contrast must not keep the environment of the old one
            run_environment.write_sidecar(contrasts.path(name), args,
                                          loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)
    for name, error in contrasts.failures:
        print(f"FAILED {name}: {error}")
    return 1 if contrasts.failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [d for d in args.dataset_dir if not d.expanduser().is_dir()]
    if missing:
        print(f"ABORT: no existe {missing[0]}", file=sys.stderr)
        return 1
    return max(rebuild(d.expanduser(), args) for d in args.dataset_dir)


if __name__ == "__main__":
    sys.exit(main())
