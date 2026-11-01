"""Derive the v3 pool-use attestation for the source-disjoint pool.

This is a MECHANICAL RE-DERIVATION, not a new human attestation. The consumption
policy is copied unchanged from v2; only the pinned artifacts move, because the
pool itself changed under a documented exclusion rule with its own receipt.

What stays exactly as attested in v2:
  * schema, status, method_scopes, stage, schema_fields, freeze_semantics
  * training_consumable, selection_policy
  * patient_independence_accredited = false  and  holdout_novelty = false

Those last two matter: the library validator *requires* both to be false and
raises otherwise. Source-cohort disjointness is a different property and is
recorded under its own key, never by flipping the patient-independence flag.

What moves: pool_root, preflight_receipt, stages and stage_manifests, to the
source-disjoint manifests.

Also writes the ``receipt.json`` the validator expects beside the manifests
(``manifest_dir / "receipt.json"``), describing the v3 derivation.

Emits the V3_POOL_USE_BINDINGS literal and the attestation SHA-256 so they can be
added to the library additively, leaving the v2 path untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

AUTHORIZED_BY = "Maria Paula Saavedra (project author), 2026-11-01"
EXCLUSION_RULE = (
    "Exclude every pool image belonging to a source cohort that also contributes "
    "images to the ISIC 2018 Task 1 evaluation split (SONIC, MSK-1..5, UDA-1). "
    "HAM10000 and BCN20000 are untouched: they contribute zero evaluation images."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-attestation", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--exclusion-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    v2 = json.loads(args.v2_attestation.read_text(encoding="utf-8"))
    pins = json.loads(args.pins.read_text(encoding="utf-8"))
    pool_dir = args.pool_dir.resolve()

    # 1. The receipt the validator looks for beside the manifests.
    #
    # If one is already there, it is read and NOT overwritten. The trainer requires
    # this file to use the v2 bridge schema (see derive_pool_v3_bridge_receipt.py);
    # an earlier version of this script wrote its own shape here and clobbered it,
    # which the trainer rejected with "receipt schema or source artifact pins
    # mismatch". Deriving the attestation must not redefine its own inputs.
    receipt_path = pool_dir / "receipt.json"
    reuse_existing = receipt_path.is_file()

    receipt = {
        "schema": "dermlip_source_disjoint_pool_receipt.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "derived_from": str(Path(v2["bindings"]["preflight_receipt"]["path"])),
        "derived_from_sha256": v2["bindings"]["preflight_receipt"]["sha256"],
        "exclusion_rule": EXCLUSION_RULE,
        "exclusion_receipt": {
            "path": str(args.exclusion_receipt.resolve()),
            "sha256": sha256_file(args.exclusion_receipt),
        },
        "source_cohort_disjoint_from_evaluation": True,
        "patient_independence_accredited": False,
        "stages": {
            stage: {
                "record_count": pin["record_count"],
                "dataset_counts": pin["dataset_counts"],
                "cohort_sha256": pin["cohort_sha256"],
                "sha256": pin["file_sha256"],
            }
            for stage, pin in pins.items()
        },
    }
    if reuse_existing:
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        print(f"receipt.json already present, reused unchanged: {receipt_sha}")
    else:
        receipt_bytes = json.dumps(receipt, indent=2).encode("utf-8")
        receipt_path.write_bytes(receipt_bytes)
        receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
        print(f"receipt.json written: {receipt_sha}")

    # 2. Bindings: v2 verbatim, with only the pool-identity fields moved.
    bindings = json.loads(json.dumps(v2["bindings"]))
    bindings["pool_root"] = str(pool_dir) + "/"
    bindings["preflight_receipt"] = {"path": str(receipt_path), "sha256": receipt_sha}
    for stage, pin in pins.items():
        bindings["stages"][stage] = {
            "path": str(pool_dir / f"{stage}.manifest.json"),
            "sha256": pin["file_sha256"],
            "record_count": pin["record_count"],
            "dataset_counts": pin["dataset_counts"],
            "cohort_sha256": pin["cohort_sha256"],
        }

    # 3. Attestation: v2 verbatim, with the moved pins and an explicit derivation note.
    v3 = json.loads(json.dumps(v2))
    v3["bindings"] = bindings
    v3["preflight_receipt"] = bindings["preflight_receipt"]
    for stage, pin in pins.items():
        v3["stage_manifests"][stage] = dict(bindings["stages"][stage])
    v3["version"] = "20261101-v3"
    v3["supersedes"] = {
        "path": str(args.v2_attestation.resolve()),
        "sha256": sha256_file(args.v2_attestation),
        "version": v2.get("version"),
    }
    v3["source_cohort_disjoint_from_evaluation"] = True
    v3["derivation"] = {
        "kind": "mechanical_re_derivation",
        "is_new_human_attestation": False,
        "policy_changed": False,
        "what_changed": "pool_root, preflight_receipt, stages, stage_manifests",
        "rule": EXCLUSION_RULE,
        "authorized_by": AUTHORIZED_BY,
        "note": (
            "The consumption policy is unchanged from v2. Only the pinned pool "
            "artifacts move, because the pool was re-frozen under the recorded "
            "exclusion rule. patient_independence_accredited stays false: source "
            "cohort disjointness is a different property and does not accredit it."
        ),
    }

    # Invariants the library validator enforces; fail here rather than at launch.
    assert v3["patient_independence_accredited"] is False
    assert v3["holdout_novelty"] is False
    assert v3["bindings"]["patient_independence_accredited"] is False
    assert v3["schema"] == v2["schema"]
    assert v3["status"] == v2["status"]
    assert v3["schema_fields"] == v2["schema_fields"]
    assert v3["training_consumable"] is True

    payload = json.dumps(v3, indent=2, sort_keys=True).encode("utf-8")
    args.out.write_bytes(payload)
    attestation_sha = hashlib.sha256(payload).hexdigest()

    print(f"wrote {args.out}")
    print(f"V3_POOL_USE_ATTESTATION_SHA256 = {attestation_sha!r}")
    print("\n--- V3_POOL_USE_BINDINGS (python literal) ---")
    print("V3_POOL_USE_BINDINGS = " + json.dumps(bindings, indent=4, sort_keys=True)
          .replace(": true", ": True").replace(": false", ": False").replace(": null", ": None"))


if __name__ == "__main__":
    main()
