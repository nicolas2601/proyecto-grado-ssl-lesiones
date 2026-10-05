"""Build the finals exclusion list: val + holdout + their pool near-duplicates.

Rule registered in docs/27 10.1 before this file existed: every pool image
whose public DINOv2 CLS cosine against ANY val/holdout image is >= 0.95 is
excluded from the finals' pretraining pool. The pilots keep their list.

    OMP_NUM_THREADS=8 nice -n 19 python scripts/c1_finals_exclude.py \
        --ham-root ~/work/datos/ham10000 --image-root $T/inputs/originals \
        --holdout $S/selection/selection_holdout.json \
        --exclude-ids-file $S/selection/pretrain_exclude_ids.txt \
        --bank-cache-dir $T/bank/cache --bank-backbone dinov2_public_vitb14 \
        --out $S/selection/pretrain_exclude_ids_finals.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import c1_near_dup_embed as embed  # noqa: E402
import cluster_frozen_bank as bank  # noqa: E402
from c1_build_val_index import rebuild_split  # noqa: E402

from dermssl import near_duplicates as nd  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.pretrain import config as config_mod  # noqa: E402

THRESHOLD = 0.95
SENSITIVITY = (0.94, 0.93)


def main() -> int:
    args = embed.parse_args()
    out = args.out.expanduser()
    report_path = out.with_suffix(".json")
    if out.exists() or report_path.exists():
        raise SystemExit(f"ABORT: {out} o su reporte ya existen")
    revision = rp.clean_revision(REPO)
    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split, _ = rebuild_split(args.ham_root.expanduser())
    holdout = json.loads(args.holdout.expanduser().read_text(encoding="utf-8"))
    cached = embed.bank_features(args.bank_cache_dir.expanduser(), args.bank_backbone, split)
    ham_index = bank.index_images([args.ham_root.expanduser()])
    query_ids = sorted(set(split["val_ids"]) | set(holdout["holdout_ids"]))
    guard_ids = sorted(split["val_ids"])[: embed.N_GUARD]

    model = bank.build_frozen_backbone(args.arch, "public", embed.RESOLUTION, device)["model"]
    reproduced = embed.reproduction_guard(model, guard_ids, ham_index, cached, device, args.batch_size)
    control = embed.positive_control(model, guard_ids, ham_index, cached, device, args.batch_size)
    if min(control) < THRESHOLD:
        raise SystemExit(f"ABORT: control positivo {min(control):.4f} < {THRESHOLD}")

    base_file = args.exclude_ids_file.expanduser()
    base_ids = config_mod.load_excluded_ids(base_file)
    pool_all = config_mod.collect_images(args.image_root.expanduser())
    kept, _ = config_mod.apply_exclusion(pool_all, base_ids)
    pool_paths = [Path(p) for p in kept]
    feats = embed.extract(model, {str(p): p for p in pool_paths}, device, args.batch_size)
    best = nd.pool_max_cosine(np.stack([cached[i] for i in query_ids]), feats)

    def flagged(threshold: float) -> list[str]:
        return sorted({pool_paths[i].stem for i in np.flatnonzero(best >= threshold)})

    new_ids = flagged(THRESHOLD)
    final_ids = sorted(base_ids | set(new_ids))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(final_ids) + "\n", encoding="utf-8")
    final_pool, _ = config_mod.apply_exclusion(pool_all, frozenset(final_ids))
    report = {
        "code_revision": revision,
        "rule": "val + holdout + pool images with max public DINOv2 CLS cosine >= 0.95 (docs/27 10.1)",
        "base_exclude_sha256": rp.file_sha256(base_file),
        "finals_exclude_sha256": rp.file_sha256(out),
        "reproduction_min_cosine": reproduced,
        "positive_control_min": min(control),
        "pool_before": len(pool_paths),
        "pool_images_at_threshold": int((best >= THRESHOLD).sum()),
        "new_ids": len(new_ids),
        "final_pool": len(final_pool),
        "sensitivity_not_applied": {str(t): len(flagged(t)) for t in SENSITIVITY},
        "flagged": [
            {"pool_path": str(pool_paths[i]), "max_cosine": float(best[i])}
            for i in np.argsort(-best) if best[i] >= THRESHOLD
        ],
    }
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".pool_features.npz"), features=feats,
                        paths=np.array([str(p) for p in pool_paths]))
    print(json.dumps({k: v for k, v in report.items() if k != "flagged"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
