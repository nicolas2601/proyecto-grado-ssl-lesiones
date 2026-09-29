r"""Pod check: ``s_diag_predict.py --workers N`` writes the serial run's masks.

Runs the predictor twice on the first ``--limit`` sorted images, once with
``--workers 1`` (the serial path) and once with ``--workers N``, then compares
every mask PNG by sha256 and ``run.json`` field by field, ignoring only the
runtime fields (``workers`` and the ``environment`` block). Prints PASS or FAIL with every difference and
exits 0 or 1. Every other argument goes to the predictor unchanged::

    cd ~/work/proyectonp/repo_v7 && source ~/work/proyectonp/.venv/bin/activate
    P=$HOME/work/proyectonp
    for METHOD in primary kmeans pca448; do
      python scripts/s_diag_parallel_check.py --work-dir /tmp/sdiag_check_$METHOD \
        --workers 36 --limit 50 --backbone public=dinov2_vitb14:public \
        --image-dir $P/datos_isic2018/ISIC2018_Task1-2_Training_Input \
        --reference-set isic2018 --method $METHOD
    done
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PREDICT = Path(__file__).resolve().with_name("s_diag_predict.py")
RUNTIME_FIELDS = frozenset({"workers", "environment"})


def _mask_digests(masks_dir: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in masks_dir.iterdir()}


def _record(run_dir: Path) -> dict:
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {key: value for key, value in record.items() if key not in RUNTIME_FIELDS}


def compare_runs(first: Path, second: Path) -> list[str]:
    """Every difference between two S-diag runs, runtime fields aside."""
    masks_a, masks_b = _mask_digests(first / "masks"), _mask_digests(second / "masks")
    differences = [f"mask differs: {name}" for name in sorted(masks_a.keys() & masks_b.keys())
                   if masks_a[name] != masks_b[name]]
    differences += [f"mask only in one run: {name}"
                    for name in sorted(masks_a.keys() ^ masks_b.keys())]
    record_a, record_b = _record(first), _record(second)
    differences += [f"run.json differs: {key}" for key in sorted(record_a.keys() | record_b.keys())
                    if record_a.get(key) != record_b.get(key)]
    return differences


def _predict(passthrough: list[str], output_dir: Path, workers: int, limit: int) -> float:
    command = [sys.executable, str(PREDICT), *passthrough, "--output-dir", str(output_dir),
               "--workers", str(workers), "--limit", str(limit)]
    start = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - start


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work-dir", required=True, type=Path, help="must not exist")
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--limit", type=int, default=50)
    args, passthrough = parser.parse_known_args(argv)
    if args.workers < 2 or args.limit < 1:
        parser.error("--workers must be >= 2 and --limit >= 1")
    return args, passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    work = args.work_dir.expanduser()
    if work.exists():
        raise SystemExit(f"ABORT: {work} already exists")
    serial, parallel = work / "serial", work / f"workers{args.workers}"
    seconds = [_predict(passthrough, serial, 1, args.limit),
               _predict(passthrough, parallel, args.workers, args.limit)]
    differences = compare_runs(serial, parallel)
    n_masks = len(list((serial / "masks").iterdir()))
    print(f"serial {seconds[0]:.1f}s, workers={args.workers} {seconds[1]:.1f}s, {n_masks} masks")
    print("\n".join(differences) or f"PASS: {n_masks} masks bit-identical (sha256)")
    if differences:
        print(f"FAIL: {len(differences)} differences")
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
