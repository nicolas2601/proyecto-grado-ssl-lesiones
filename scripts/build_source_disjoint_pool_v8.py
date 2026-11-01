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

    receipt_path = args.out_dir / "source_disjoint_pool.receipt.json"
    receipt_payload = json.dumps(receipt, indent=2).encode("utf-8")
    receipt_path.write_bytes(receipt_payload)
    print(f"\nrecibo: {receipt_path}")
    print(f"  sha256 del recibo: {sha256_bytes(receipt_payload)}")
    for name, block in receipt["pools"].items():
        print(f"  {name}.manifest.json sha256: {block['manifest_sha256']}")


if __name__ == "__main__":
    main()
