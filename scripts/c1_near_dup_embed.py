"""DINOv2 embedding near-duplicates between val + internal holdout and the pool.

pHash collapses on dermoscopy (docs/27 10.1), so this compares frozen public
DINOv2 CLS features instead. Queries reuse the validated v6 bank cache; the
pool is extracted with the bank's own eval transform, and two guards run
before any number is reported:

* re-extracting 20 bank images must reproduce the cache (cosine > 0.9999),
  or the pool features would not be comparable with the queries;
* 20 validation images recompressed (JPEG q=75) and rescaled to 90 % must
  score at or above the threshold against their originals, or the
  threshold cannot see a real copy.

    OMP_NUM_THREADS=8 nice -n 19 python scripts/c1_near_dup_embed.py \
        --ham-root ~/work/datos/ham10000 --image-root $T/inputs/originals \
        --holdout $S/selection/selection_holdout.json \
        --exclude-ids-file $S/selection/pretrain_exclude_ids.txt \
        --bank-cache-dir $T/bank/cache --bank-backbone dinov2_vitb14 \
        --out $S/selection/near_dup_embed.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cluster_frozen_bank as bank  # noqa: E402
from c1_build_val_index import rebuild_split  # noqa: E402

from dermssl import near_duplicates as nd  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.feature_cache import FeatureCache, fingerprint_ids  # noqa: E402
from dermssl.pretrain import config as config_mod  # noqa: E402

RESOLUTION = 224
THRESHOLD = 0.95
TOP_K = 5
N_REVIEW = 20
N_GUARD = 20
REPRODUCE_MIN_COSINE = 0.9999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--ham-root", "--image-root", "--holdout", "--exclude-ids-file",
                 "--bank-cache-dir", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--bank-backbone", required=True, help="name in the bank cache keys")
    parser.add_argument("--arch", default="dinov2_vitb14")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def bank_features(cache_dir: Path, backbone: str, split: dict) -> dict[str, np.ndarray]:
    """id -> cached public feature, for the train and val rows of the bank."""
    cache, by_id = FeatureCache(cache_dir), {}
    for name in ("train", "val"):
        ids = list(split[f"{name}_ids"])
        hits = sorted(cache_dir.glob(f"v1__{backbone}__{name}__r{RESOLUTION}__eval__*.features.npy"))
        if len(hits) != 1:
            raise SystemExit(f"ABORT: esperaba 1 cache {backbone}/{name}, hay {len(hits)}")
        key = hits[0].name[: -len(".features.npy")]
        if key.rsplit("__", 1)[-1] != fingerprint_ids(ids):
            raise SystemExit(f"ABORT: el cache {key} no esta alineado con los ids de {name}")
        entry = cache.load(key)
        if entry is None:
            raise SystemExit(f"ABORT: no se pudo cargar {key}")
        by_id.update(zip(ids, entry.features, strict=True))
    return by_id


def extract(model, index: dict, device, batch_size: int) -> np.ndarray:
    ids = list(index)
    feats, _ = bank.extract_features(
        model, ids, index, dict.fromkeys(ids, 0), RESOLUTION, device, batch_size, 0
    )
    return feats


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (nd.unit_rows(a) * nd.unit_rows(b)).sum(axis=1)


def reproduction_guard(model, ids, ham_index, cached, device, batch_size) -> float:
    fresh = extract(model, {i: ham_index[i] for i in ids}, device, batch_size)
    worst = float(row_cosine(fresh, np.stack([cached[i] for i in ids])).min())
    if worst <= REPRODUCE_MIN_COSINE:
        raise SystemExit(f"ABORT: re-extraccion del banco da coseno minimo {worst:.6f}")
    return worst


def positive_control(model, ids, ham_index, cached, device, batch_size) -> list[float]:
    from PIL import Image  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        index = {}
        for image_id in ids:
            with Image.open(ham_index[image_id]) as image:
                rgb = image.convert("RGB")
                small = rgb.resize((int(rgb.width * 0.9), int(rgb.height * 0.9)), Image.BILINEAR)
                path = Path(tmp) / f"{image_id}.jpg"
                small.save(path, quality=75)
                index[image_id] = path
        fresh = extract(model, index, device, batch_size)
    return row_cosine(fresh, np.stack([cached[i] for i in ids])).tolist()


def review_pairs(scores, index, query_ids, pool_paths) -> list[dict]:
    flat = [(float(scores[q, j]), q, int(index[q, j]))
            for q in range(len(query_ids)) for j in range(scores.shape[1])]
    flat.sort(reverse=True)
    return [{"cosine": s, "query_id": query_ids[q], "pool_path": str(pool_paths[p])}
            for s, q, p in flat[:N_REVIEW]]


def main() -> int:
    args = parse_args()
    out = args.out.expanduser()
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe")
    revision = rp.clean_revision(REPO)
    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split, _ = rebuild_split(args.ham_root.expanduser())
    holdout = json.loads(args.holdout.expanduser().read_text(encoding="utf-8"))
    cached = bank_features(args.bank_cache_dir.expanduser(), args.bank_backbone, split)
    ham_index = bank.index_images([args.ham_root.expanduser()])
    query_ids = sorted(set(split["val_ids"]) | set(holdout["holdout_ids"]))
    guard_ids = sorted(split["val_ids"])[:N_GUARD]

    model = bank.build_frozen_backbone(args.arch, "public", RESOLUTION, device)["model"]
    reproduced = reproduction_guard(model, guard_ids, ham_index, cached, device, args.batch_size)
    control = positive_control(model, guard_ids, ham_index, cached, device, args.batch_size)
    if min(control) < THRESHOLD:
        raise SystemExit(f"ABORT: control positivo con coseno {min(control):.4f} < {THRESHOLD}")

    exclude_file = args.exclude_ids_file.expanduser()
    pool_all = config_mod.collect_images(args.image_root.expanduser())
    kept, _ = config_mod.apply_exclusion(pool_all, config_mod.load_excluded_ids(exclude_file))
    pool_paths = [Path(p) for p in kept]
    print(f"extrayendo {len(pool_paths)} imagenes del pool", flush=True)
    pool_feats = extract(model, {str(p): p for p in pool_paths}, device, args.batch_size)

    queries = np.stack([cached[i] for i in query_ids])
    scores, index = nd.cosine_topk(queries, pool_feats, k=TOP_K)
    top1 = scores[:, 0]
    document = {
        "code_revision": revision,
        "exclude_ids_sha256": rp.file_sha256(exclude_file),
        "backbone": f"{args.arch}:public (bank cache {args.bank_backbone})",
        "transform": f"bank eval: Resize {int(RESOLUTION * 1.14)} + CenterCrop {RESOLUTION} + ImageNet norm",
        "reproduction_min_cosine": reproduced,
        "positive_control_cosines": control,
        "top1": nd.summarize_top1(top1, threshold=THRESHOLD),
        "at_or_above": [
            {"query_id": query_ids[q], "pool_path": str(pool_paths[index[q, 0]]),
             "cosine": float(top1[q])}
            for q in np.flatnonzero(top1 >= THRESHOLD)
        ],
        "review_top20": review_pairs(scores, index, query_ids, pool_paths),
        "n_queries": len(query_ids),
        "n_pool_kept": len(pool_paths),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(json.dumps({k: document[k] for k in ("top1", "reproduction_min_cosine")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
