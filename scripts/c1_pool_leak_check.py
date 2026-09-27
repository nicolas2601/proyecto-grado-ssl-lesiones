"""Audit that a pretraining pool shares no lesion with the holdout or validation.

Recomputes the pool the run trained on with the loop's own functions
(``collect_images`` + ``apply_exclusion``), checks the counts against the
run's ``run.json``, and intersects the kept images' HAM lesion ids with the
lesions of the docs/28 internal holdout and of the C1 validation. Exit code 1
on any shared lesion or count mismatch.

    python scripts/c1_pool_leak_check.py --ham-root ~/work/datos/ham10000 \
        --image-root $T/inputs/originals --holdout $S/selection/selection_holdout.json \
        --exclude-ids-file $S/selection/pretrain_exclude_ids.txt \
        --run-json $V/pilot_A1/run.json --out $S/selection/pool_leak_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from c1_build_val_index import rebuild_split  # noqa: E402

from dermssl import run_provenance as rp  # noqa: E402
from dermssl.pretrain import config as config_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--exclude-ids-file", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def lesions_of(ids, lesion_by_id: dict) -> set:
    return {lesion_by_id[i] for i in ids if i in lesion_by_id}


def main() -> int:
    args = parse_args()
    run = json.loads(args.run_json.expanduser().read_text(encoding="utf-8"))
    exclude_file = args.exclude_ids_file.expanduser()
    split, meta = rebuild_split(args.ham_root.expanduser())
    lesion_by_id = dict(zip(meta["image_id"], meta["lesion_id"], strict=True))
    holdout = json.loads(args.holdout.expanduser().read_text(encoding="utf-8"))

    pool = config_mod.collect_images(args.image_root.expanduser())
    excluded = config_mod.load_excluded_ids(exclude_file)
    kept, report = config_mod.apply_exclusion(pool, excluded)
    kept_ids = [Path(p).stem for p in kept]
    kept_lesions = lesions_of(kept_ids, lesion_by_id)

    val_ids, holdout_ids = set(split["val_ids"]), set(holdout["holdout_ids"])
    checks = {
        "exclude_sha_matches_run": rp.file_sha256(exclude_file) == run.get("exclude_ids_sha256"),
        "kept_matches_run": len(kept) == run["execution"]["image_count"],
        "exclusion_report_matches_run": report == run.get("exclusion"),
        "ids_shared_with_val": len(set(kept_ids) & val_ids),
        "ids_shared_with_holdout": len(set(kept_ids) & holdout_ids),
        "lesions_shared_with_val": len(kept_lesions & lesions_of(val_ids, lesion_by_id)),
        "lesions_shared_with_holdout": len(kept_lesions & lesions_of(holdout_ids, lesion_by_id)),
    }
    document = {
        "run_json": str(args.run_json),
        "code_revision_of_run": run.get("code_revision"),
        "pool_total": len(pool),
        "n_excluded_ids": len(excluded),
        "n_val_ids": len(val_ids),
        "n_holdout_ids": len(holdout_ids),
        "expected_kept": len(pool) - sum(Path(p).stem in val_ids | holdout_ids for p in pool),
        "kept": len(kept),
        "kept_with_ham_lesion": sum(i in lesion_by_id for i in kept_ids),
        "exclusion": report,
        "checks": checks,
    }
    ok = all(v is True for k, v in checks.items() if isinstance(v, bool)) and not any(
        v for k, v in checks.items() if not isinstance(v, bool)
    )
    document["verdict"] = "PASS" if ok else "FAIL"
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(json.dumps(document, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
