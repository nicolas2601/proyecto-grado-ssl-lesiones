"""Derive the v3 pool's receipt.json from the v2 bridge receipt.

The trainer validates the receipt sitting beside the manifest against a fixed
shape: schema ``v2_shared_segmentation_pool_bridge.v1``, status
``preflight_bridge_built_execution_disabled``, both arms listed,
``same_underlying_pool_for_both_arms`` true, and eight named source-artifact
hashes that must equal the pinned contract.

Those eight are UPSTREAM inputs (candidate source pool, split crosswalk, the two
stage source manifests, the ISIC 2018 evaluation manifest and its exclusion
receipt, the PH2 eligible manifest and curator exclusions). None of them change
when the pool is re-frozen by source cohort, so the v3 receipt inherits them
verbatim rather than restating them.

What does change is the per-stage counts and hashes, which are rewritten, plus a
derivation note so the file does not read as if it described the v2 pool.

An earlier attempt shipped a custom receipt schema here, which the trainer
rejected. The lesson is the same one the attestation taught: derive from the
artifact the validator expects, do not invent a parallel shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-receipt", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.v2_receipt.read_text(encoding="utf-8"))
    pins = json.loads((args.pool_dir / "pins.json").read_text(encoding="utf-8"))

    pools = receipt.get("pools")
    if not isinstance(pools, dict):
        raise SystemExit("v2 receipt has no pools block to rewrite")

    for stage, pin in pins.items():
        if stage not in pools:
            raise SystemExit(f"v2 receipt has no {stage} pool entry")
        entry = dict(pools[stage])
        entry["path"] = str((args.pool_dir / f"{stage}.manifest.json").resolve())
        entry["sha256"] = pin["file_sha256"]
        # The trainer reads this entry as "count", not "record_count": binding it
        # under the wrong key leaves the v2 value in place and fails with
        # "receipt does not bind this manifest and cohort".
        entry["count"] = pin["record_count"]
        entry["dataset_counts"] = pin["dataset_counts"]
        entry["cohort_sha256"] = pin["cohort_sha256"]
        pools[stage] = entry
        print(f"{stage}: {pin['record_count']} records, {pin['file_sha256'][:16]}")

    receipt["pools"] = pools
    receipt["derivation"] = {
        "kind": "mechanical_re_derivation",
        "is_new_human_attestation": False,
        "derived_from": str(args.v2_receipt.resolve()),
        "derived_from_sha256": hashlib.sha256(args.v2_receipt.read_bytes()).hexdigest(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_changed": "pools[*].path, sha256, record_count, dataset_counts, cohort_sha256",
        "what_did_not_change": (
            "schema, status, arms, source_artifacts, evaluation, ph2, gates, limitations"
        ),
        "rule": (
            "Pool re-frozen by excluding every image from a source cohort that also "
            "feeds the ISIC 2018 Task 1 evaluation split. Source artifacts are "
            "upstream of that exclusion and therefore unchanged."
        ),
        "companion_receipts": {
            "exclusion": "source_disjoint_pool.receipt.json",
            "cohort_attribution": "source_cohort_attribution.json",
        },
    }
    receipt["source_cohort_disjoint_from_evaluation"] = True
    receipt["patient_independence_accredited"] = False

    out = args.pool_dir / "receipt.json"
    payload = json.dumps(receipt, indent=2).encode("utf-8")
    out.write_bytes(payload)
    print(f"\nwrote {out}")
    print(f"  receipt.json sha256: {hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
