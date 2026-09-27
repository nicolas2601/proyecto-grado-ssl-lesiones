"""Fixed image subset of a cache, to measure the numeric floor of a saliency rerun.

S2 (a) recomputes CCAM saliency on CPU in fp32; the published r0 saliency
came from GPU bf16. Running both precisions on the SAME unmodified images
gives the coverage difference that is numeric noise, not hair removal.
Writes ``<out>/cache`` (images, field, manifest of the subset),
``<out>/published_saliency.npy`` and ``<out>/indices.json``.

    python precision_subset.py --cache-dir salidas/final/cache \
        --saliency salidas/final/ccam/saliency.npy --out salidas/v7_seg/s2_precision
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_PLAN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PLAN_ROOT / "comun" / "nucleo"))

import cache_io  # noqa: E402

N_IMAGES = 500
SEED = 20260927


def subset_indices(total: int, n: int = N_IMAGES, seed: int = SEED) -> np.ndarray:
    """Sorted, reproducible sample without replacement."""
    if n > total:
        raise ValueError(f"se piden {n} imagenes de {total}")
    return np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))


def write_subset(cache_dir: Path, saliency: Path, out: Path, n: int = N_IMAGES, seed: int = SEED) -> dict:
    images = cache_io.open_cache(cache_dir / "images.npy")
    idx = subset_indices(int(images.shape[0]), n, seed)
    target = out / "cache"
    target.mkdir(parents=True, exist_ok=True)
    for name, source in (("images.npy", images), ("field.npy", cache_io.open_cache(cache_dir / "field.npy"))):
        sub = cache_io.create_cache(target / name, (len(idx), *source.shape[1:]), source.dtype)
        sub[:] = source[idx]
        sub.flush()
    published = cache_io.open_cache(saliency)
    sal = cache_io.create_cache(out / "published_saliency.npy", (len(idx), *published.shape[1:]), published.dtype)
    sal[:] = published[idx]
    sal.flush()
    with open(cache_dir / "manifest.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(target / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for new_index, old in enumerate(idx):
            writer.writerow({**rows[int(old)], "index": new_index})
    record = {"n": len(idx), "seed": seed, "indices": idx.tolist(), "source_cache": str(cache_dir)}
    (out / "indices.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--saliency", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in write_subset(args.cache_dir, args.saliency, args.out).items() if k != "indices"}))


if __name__ == "__main__":
    main()
