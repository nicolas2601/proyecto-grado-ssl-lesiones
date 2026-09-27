"""pHash near-duplicates between val + internal holdout and the pretraining pool.

Closes the gap the lesion-id audit cannot see (docs/27 10.1): a HAM lesion
re-photographed in ISIC 2019/2020 under another id and other bytes. CPU only;
run it niced, it pauses nothing by itself.

    OMP_NUM_THREADS=8 nice -n 19 python scripts/c1_near_dup_check.py \
        --ham-root ~/work/datos/ham10000 --image-root $T/inputs/originals \
        --holdout $S/selection/selection_holdout.json \
        --exclude-ids-file $S/selection/pretrain_exclude_ids.txt \
        --out $S/selection/near_dup_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cluster_frozen_bank as bank  # noqa: E402
from c1_build_val_index import rebuild_split  # noqa: E402

from dermssl import near_duplicates as nd  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.pretrain import config as config_mod  # noqa: E402

MAX_DISTANCE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--exclude-ids-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-distance", type=int, default=MAX_DISTANCE)
    return parser.parse_args()


def hash_all(paths: list[Path], workers: int) -> np.ndarray:
    with Pool(workers) as pool:
        return np.array(pool.map(nd.phash, paths, chunksize=64), dtype=np.uint64)


def main() -> int:
    args = parse_args()
    out = args.out.expanduser()
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe")
    revision = rp.clean_revision(REPO)
    split, _ = rebuild_split(args.ham_root.expanduser())
    holdout = json.loads(args.holdout.expanduser().read_text(encoding="utf-8"))
    index = bank.index_images([args.ham_root.expanduser()])
    query_ids = sorted(set(split["val_ids"]) | set(holdout["holdout_ids"]))
    query_paths = [Path(index[i]) for i in query_ids]

    exclude_file = args.exclude_ids_file.expanduser()
    pool_all = config_mod.collect_images(args.image_root.expanduser())
    kept, _ = config_mod.apply_exclusion(pool_all, config_mod.load_excluded_ids(exclude_file))
    kept = [Path(p) for p in kept]

    print(f"hashing {len(query_paths)} consultas y {len(kept)} del pool", flush=True)
    pairs = nd.near_pairs(
        hash_all(query_paths, args.workers), hash_all(kept, args.workers),
        max_distance=args.max_distance,
    )
    document = {
        "code_revision": revision,
        "exclude_ids_sha256": rp.file_sha256(exclude_file),
        "hash": "imagehash.phash 64 bits",
        "max_distance": args.max_distance,
        "n_queries": len(query_paths),
        "n_pool_kept": len(kept),
        "n_pairs": len(pairs),
        "pairs": [
            {"query_id": query_ids[q], "query_path": str(query_paths[q]),
             "pool_path": str(kept[p]), "hamming": d}
            for q, p, d in pairs
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"pares con Hamming <= {args.max_distance}: {len(pairs)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
