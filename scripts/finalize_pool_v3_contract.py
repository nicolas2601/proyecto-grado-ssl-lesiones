"""Produce the final v3 pool with a fully self-consistent pool_contract.

Two defects are fixed here, both found by validating against the library instead
of trusting the build:

1. The -002 build updated ``stage_count`` / ``stage_cohort_sha256`` /
   ``stage_dataset_counts`` but left ``selection_count``, ``selection_cohort_sha256``,
   ``selection_dataset_counts`` and the ``refit_*`` twins at their v2 values. A
   manifest whose stage pins say 34115 while its selection pins say 34330 is
   internally inconsistent, and nothing in the chain catches it.

2. ``validate_v2_shared_pool_contract`` compares the contract by FULL DICT
   EQUALITY against a hardcoded constant. So the contract has to be exactly the
   shape the library expects, and the V3 constant has to be derived from the
   finished manifest rather than hand-written, or the two drift.

This writes both manifests, recomputes every pin with the library's own
``cohort_digest``, and emits the ``V3_SHARED_POOL_CONTRACT`` literal so the
library constant is generated from the artifact it validates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

STAGE_KEYS = (
    "stage",
    "stage_count",
    "stage_cohort_sha256",
    "stage_dataset_counts",
    "stage_source_manifest_sha256",
)


def to_literal(value, indent: int = 4) -> str:
    text = json.dumps(value, indent=indent, sort_keys=True)
    return text.replace(": true", ": True").replace(": false", ": False").replace(": null", ": None")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True, help="the -002 pool dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    from dermssl.segmentation_methods.common.protocol import cohort_digest

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, dict] = {}
    documents: dict[str, dict] = {}

    # Pass 1: every stage pin, because each contract has to carry both stages.
    for stage in ("selection", "refit"):
        document = json.loads((args.src_dir / f"{stage}.manifest.json").read_text(encoding="utf-8"))
        records = document["records"]
        stages[stage] = {
            "count": len(records),
            "dataset_counts": dict(Counter(r["dataset"] for r in records)),
            "cohort_sha256": cohort_digest(records),
        }
        documents[stage] = document

    # Pass 2: write both manifests with a contract consistent in every field.
    pins: dict[str, dict] = {}
    for stage, document in documents.items():
        contract = dict(document["pool_contract"])
        for other in ("selection", "refit"):
            contract[f"{other}_count"] = stages[other]["count"]
            contract[f"{other}_dataset_counts"] = stages[other]["dataset_counts"]
            contract[f"{other}_cohort_sha256"] = stages[other]["cohort_sha256"]
        contract["stage"] = stage
        contract["stage_count"] = stages[stage]["count"]
        contract["stage_dataset_counts"] = stages[stage]["dataset_counts"]
        contract["stage_cohort_sha256"] = stages[stage]["cohort_sha256"]
        document["pool_contract"] = contract

        out_path = args.out_dir / f"{stage}.manifest.json"
        payload = json.dumps(document, indent=1).encode("utf-8")
        out_path.write_bytes(payload)
        pins[stage] = {
            "record_count": stages[stage]["count"],
            "dataset_counts": stages[stage]["dataset_counts"],
            "cohort_sha256": stages[stage]["cohort_sha256"],
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
        # Internal consistency, asserted rather than assumed.
        assert contract["stage_count"] == document["count"] == len(document["records"])
        assert contract[f"{stage}_count"] == contract["stage_count"]
        assert contract[f"{stage}_cohort_sha256"] == contract["stage_cohort_sha256"]
        print(f"{stage}: {pins[stage]['record_count']} records, sha {pins[stage]['file_sha256']}")

    # Carry the companion receipts over unchanged; their content does not depend
    # on the contract fix, only their directory does.
    for name in ("receipt.json", "source_disjoint_pool.receipt.json", "source_cohort_attribution.json"):
        src = args.src_dir / name
        if src.is_file():
            (args.out_dir / name).write_bytes(src.read_bytes())

    (args.out_dir / "pins.json").write_text(json.dumps(pins, indent=1), encoding="utf-8")

    # The V3 constant, derived from the finished selection manifest so it cannot
    # drift from the artifact the library will compare it against.
    base = {
        k: v for k, v in documents["selection"]["pool_contract"].items() if k not in STAGE_KEYS
    }
    print("\n--- V3_SHARED_POOL_CONTRACT (python literal) ---")
    print("V3_SHARED_POOL_CONTRACT = " + to_literal(base))


if __name__ == "__main__":
    main()
