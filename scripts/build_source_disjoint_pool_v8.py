"""Freeze a training pool that shares no source cohort with the evaluation split.

Identifier-level checks cannot reach the 57.7% of the evaluation split with no
``lesion_id`` and the 85% with no ``patient_id``, because that data is absent at
the archive. Source-cohort membership is complete for all 2594 evaluation
images, so disjointness at the cohort level is provable where identity is not.

Measured asymmetry that makes this cheap: 2578 of 2594 evaluation images (99.4%)
come from cohorts the pool also touches, but the pool holds only 249 images in
those cohorts. Dropping them costs 0.73% of the selection pool and leaves no
image from any cohort that feeds the evaluation.

HAM10000 and BCN20000 contribute 14196 selection images and zero evaluation
images, so they are untouched.

Writes new manifests plus a receipt recording the exclusion rule, the excluded
ids and the hashes, so the pool is auditable rather than asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

# Cohorts contributing to BOTH the evaluation split and the pool. Pool images in
# these are excluded. Cohorts that touch only one side are not a leakage path.
SHARED_COHORTS = ("SONIC", "MSK-1", "MSK-2", "MSK-3", "MSK-4", "MSK-5", "UDA-1")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collections", type=Path, required=True)
    parser.add_argument("--eval-cases", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="selection.manifest.json and refit.manifest.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    membership = json.loads(args.collections.read_text(encoding="utf-8"))["membership"]
    eval_ids = {json.loads(line)["image_id"] for line in args.eval_cases.open(encoding="utf-8")}

    # Verify the premise rather than trusting the recorded cohort list: a cohort
    # only justifies exclusion if it really touches both sides.
    shared: dict[str, int] = {}
    for cohort in SHARED_COHORTS:
        ids = set(membership[cohort])
        in_eval = len(ids & eval_ids)
        if in_eval == 0:
            raise SystemExit(
                f"{cohort} contributes no evaluation image; it is not a leakage path "
                "and must not be used to exclude pool images"
            )
        shared[cohort] = in_eval

    excluded_union: set[str] = set()
    for cohort in SHARED_COHORTS:
        excluded_union |= set(membership[cohort])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict = {
        "schema": "dermlip_source_disjoint_pool.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": (
            "Exclude every pool image belonging to a source cohort that also "
            "contributes images to the evaluation split."
        ),
        "shared_cohorts": shared,
        "evaluation_split_size": len(eval_ids),
        "collections_source": json.loads(args.collections.read_text(encoding="utf-8"))["source"],
        "patient_independence_accredited": False,
        "source_cohort_disjoint": True,
        "pools": {},
    }

    for manifest_path in args.manifest:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = data["records"]
        kept = [r for r in records if r["image_id"] not in excluded_union]
        dropped = [r["image_id"] for r in records if r["image_id"] in excluded_union]

        # A pool that still touches an evaluation cohort would make the claim false.
        remaining = {r["image_id"] for r in kept} & excluded_union
        if remaining:
            raise SystemExit(f"exclusion failed, {len(remaining)} images remain")

        name = manifest_path.stem.replace(".manifest", "")
        out_path = args.out_dir / f"{name}.manifest.json"
        new_data = dict(data)
        new_data["records"] = kept
        new_data["count"] = len(kept)
        new_data["source_cohort_disjoint_from_evaluation"] = True
        new_data["excluded_image_ids"] = sorted(dropped)
        new_data["supersedes_manifest"] = str(manifest_path)

        # pool_contract carries its own copy of the stage pins. Copying it verbatim
        # would leave a manifest whose top-level count says 34115 while its contract
        # still says 34330, with the v2 cohort digest. The library validator does not
        # check these fields, so the inconsistency would survive into a gate.
        from dermssl.segmentation_methods.common.protocol import cohort_digest

        contract = dict(new_data["pool_contract"])
        dataset_counts: dict[str, int] = {}
        for record in kept:
            dataset_counts[record["dataset"]] = dataset_counts.get(record["dataset"], 0) + 1
        contract["stage_count"] = len(kept)
        contract["stage_cohort_sha256"] = cohort_digest(kept)
        contract["stage_dataset_counts"] = dataset_counts
        # Recorded, not silently rewritten: the contract describes disjointness
        # against a 1822-image evaluation manifest, while every run evaluates on
        # 2594. That mismatch is inherited from v2 and is not ours to assert away.
        contract["evaluation_reference_caveat"] = {
            "contract_evaluation_count": data["pool_contract"].get("evaluation_count"),
            "contract_evaluation_manifest_sha256": data["pool_contract"].get(
                "evaluation_manifest_sha256"
            ),
            "exclusion_computed_against_evaluation_count": len(eval_ids),
            "note": (
                "The source-cohort exclusion in this manifest was computed against "
                "the 2594-image ISIC 2018 Task 1 split actually used for evaluation. "
                "The inherited contract pins a different, 1822-image evaluation "
                "manifest. The contract's disjointness statement therefore does not "
                "cover the evaluation split in use, and this manifest does not claim "
                "that it does."
            ),
        }
        new_data["pool_contract"] = contract

        payload = json.dumps(new_data, indent=1).encode("utf-8")
        out_path.write_bytes(payload)

        receipt["pools"][name] = {
            "manifest": str(out_path),
            "manifest_sha256": sha256_bytes(payload),
            "images_before": len(records),
            "images_after": len(kept),
            "excluded": len(dropped),
            "excluded_fraction": len(dropped) / len(records),
            "excluded_image_ids": sorted(dropped),
        }
        print(
            f"{name}: {len(records)} -> {len(kept)} "
            f"(excluidas {len(dropped)}, {len(dropped) / len(records):.2%})"
        )

    # Persist the per-image attribution. Without it the disjointness claim rests on
    # aggregates computed in a session that is gone, and "HAM10000 and BCN20000
    # contribute zero evaluation images" is unverifiable from the receipts.
    attribution: dict[str, dict[str, list[str]]] = {"evaluation": {}, "pool_excluded": {}}
    unattributed = set(eval_ids)
    for cohort, ids in membership.items():
        # Challenge partitions aggregate source cohorts; attributing by them would
        # double count, so only source cohorts are used for attribution.
        if cohort.startswith("Challenge"):
            continue
        hit = sorted(set(ids) & eval_ids)
        if hit:
            attribution["evaluation"][cohort] = hit
            unattributed -= set(hit)
        dropped_here = sorted(set(ids) & excluded_union)
        if dropped_here:
            attribution["pool_excluded"][cohort] = dropped_here

    cohort_map = {
        "schema": "dermlip_source_cohort_attribution.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collections_source": receipt["collections_source"],
        "evaluation_split_size": len(eval_ids),
        "evaluation_attributed": len(eval_ids) - len(unattributed),
        "evaluation_unattributed": sorted(unattributed),
        "shared_cohorts": list(SHARED_COHORTS),
        "eval_only_cohorts": sorted(
            c
            for c, hit in attribution["evaluation"].items()
            if c not in SHARED_COHORTS and hit
        ),
        "attribution": attribution,
    }
    cohort_map_path = args.out_dir / "source_cohort_attribution.json"
    cohort_map_payload = json.dumps(cohort_map, indent=1).encode("utf-8")
    cohort_map_path.write_bytes(cohort_map_payload)
    receipt["source_cohort_attribution"] = {
        "path": str(cohort_map_path),
        "sha256": sha256_bytes(cohort_map_payload),
        "evaluation_attributed": cohort_map["evaluation_attributed"],
        "evaluation_unattributed": len(unattributed),
        "eval_only_cohorts": cohort_map["eval_only_cohorts"],
    }
    if unattributed:
        raise SystemExit(
            f"{len(unattributed)} evaluation images have no source cohort; "
            "the disjointness claim cannot be made without complete attribution"
        )
    print(
        f"atribucion de cohorte: {cohort_map['evaluation_attributed']}/{len(eval_ids)} "
        f"del eval | cohortes solo-eval: {cohort_map['eval_only_cohorts']}"
    )

    receipt_path = args.out_dir / "source_disjoint_pool.receipt.json"
    receipt_payload = json.dumps(receipt, indent=2).encode("utf-8")
    receipt_path.write_bytes(receipt_payload)
    print(f"\nrecibo: {receipt_path}")
    print(f"  sha256 del recibo: {sha256_bytes(receipt_payload)}")
    for name, block in receipt["pools"].items():
        print(f"  {name}.manifest.json sha256: {block['manifest_sha256']}")


if __name__ == "__main__":
    main()
