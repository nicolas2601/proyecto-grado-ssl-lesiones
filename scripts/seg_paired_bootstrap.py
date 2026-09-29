"""Paired image-by-image bootstrap between two segmenters (secondary analysis).

Reads two ``per_image.csv`` files written by ``evaluate_segmentation.py`` on the
SAME reference set, pairs them by ``image_id``, and resamples the images with
the same indices for both, so the per-image difficulty they share cancels out.
The point estimate is the mean per-image difference on the full set (a - b).

Direction is read from the metric, never inferred from the sign: Dice is
higher-is-better, HD95 lower-is-better. A difference counts only when the 95 %
percentile interval excludes zero.

Pod usage (A2 vs student_r3 on ISIC 2018)::

    python scripts/seg_paired_bootstrap.py \\
      --a $S4/a2/evaluacion_isic2018/per_image.csv --label-a A2 \\
      --b $F/evaluacion_isic2018/per_image.csv --label-b student_r3 \\
      --output-json $S4/a2/paired_vs_student_isic2018.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl import run_environment  # noqa: E402

DEFAULT_SEED = 20260923
DEFAULT_RESAMPLES = 1000
HIGHER_IS_BETTER = {"dice": True, "hd95": False}


def _read(path: Path) -> dict[str, dict[str, float]]:
    with open(path, newline="") as handle:
        return {row["image_id"]: {k: float(row[k]) for k in HIGHER_IS_BETTER}
                for row in csv.DictReader(handle)}


def _direction(metric: str, delta: float, excludes_zero: bool) -> str:
    if not excludes_zero:
        return "indistinguishable"
    a_is_larger = delta > 0
    return "better" if a_is_larger == HIGHER_IS_BETTER[metric] else "worse"


def _metric_contrast(metric: str, a: np.ndarray, b: np.ndarray, n_resamples: int,
                     seed: int) -> dict:
    finite = np.isfinite(a) & np.isfinite(b)
    diff = a[finite] - b[finite]
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, diff.size, size=(n_resamples, diff.size))
    means = diff[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    delta = float(diff.mean())
    excludes_zero = bool(low > 0 or high < 0)
    return {
        "delta": delta, "ci95_low": float(low), "ci95_high": float(high),
        "mean_a": float(a[finite].mean()), "mean_b": float(b[finite].mean()),
        "excludes_zero": excludes_zero,
        "direction": _direction(metric, delta, excludes_zero),
        "higher_is_better": HIGHER_IS_BETTER[metric],
        "n": int(diff.size), "n_dropped_non_finite": int((~finite).sum()),
    }


def compare(path_a: Path, path_b: Path, n_resamples: int = DEFAULT_RESAMPLES,
            seed: int = DEFAULT_SEED) -> dict:
    rows_a, rows_b = _read(Path(path_a)), _read(Path(path_b))
    if set(rows_a) != set(rows_b):
        raise ValueError("the two files must score the same images "
                         f"({len(set(rows_a) ^ set(rows_b))} ids differ)")
    ids = sorted(rows_a)
    out: dict = {"n_paired": len(ids)}
    for metric in HIGHER_IS_BETTER:
        a = np.array([rows_a[i][metric] for i in ids], dtype=float)
        b = np.array([rows_b[i][metric] for i in ids], dtype=float)
        out[metric] = _metric_contrast(metric, a, b, n_resamples, seed)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = compare(args.a, args.b, args.n_resamples, args.seed)
    doc = {
        "label_a": args.label_a, "label_b": args.label_b,
        "per_image_a": str(args.a), "per_image_b": str(args.b),
        "rng_seed": args.seed, "n_resamples": args.n_resamples,
        "n_paired": result.pop("n_paired"), "metrics": result,
        "note": "secondary; delta = a - b, paired percentile 95 % interval",
        "environment": run_environment.snapshot(args),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({m: {k: v for k, v in r.items() if k in ("delta", "ci95_low", "ci95_high",
                                                            "direction")}
                      for m, r in doc["metrics"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
