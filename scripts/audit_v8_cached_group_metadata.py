#!/usr/bin/env python3
"""Rejoin verified cached metadata to the current filtered pool, offline in UNABIA."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

METADATA_SCRIPT_SHA256 = "3b71ad8afe78d0764174d09b82c075e655f3b5d06e7ca198056e06dd07a60420"
REQUIRED_MAPPING = {"ham10000_p1": 212, "ham10000_p2": 212, "isic2019": 65, "isic2020": 70}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected, tracked, expected_bytes=None):
    actual = identity(path)
    require(actual["sha256"] == expected, f"SHA-256 mismatch: {path}")
    require(expected_bytes is None or actual["bytes"] == expected_bytes, f"Size mismatch: {path}")
    tracked[str(Path(path).resolve())] = actual
    return actual


def publish(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return identity(path)


def closed_flags():
    return {"patient_lesion_independence": "unresolved", "freeze_blocked": True,
            "freeze_blocked_reason": "clinical_independence_unresolved",
            **{key: False for key in ("manifest_frozen", "training_consumable", "execution_ready",
                "execution_enabled", "launch_ready", "training_started", "patient_independence_accredited",
                "exposure_review_complete", "isolation_global_verified", "duplicates_identity_verified",
                "scientific_gate_receipts_issued")}}


def no_network(*args, **kwargs):
    raise RuntimeError("Network access is forbidden in this cached audit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata-report", "metadata-script", "application-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-collection", action="append", required=True)
    parser.add_argument("--expected-pool-count", type=int, default=34784)
    args = parser.parse_args()
    created = False
    stage = "validate_inputs"
    tracked = {}
    try:
        require(sys.platform.startswith("linux") and Path("/home/jovyan/work").is_dir(), "UNABIA runtime required")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created = True
        code = identity(Path(__file__))
        verify(args.metadata_report, args.metadata_report_sha256, tracked)
        require(args.metadata_script_sha256 == METADATA_SCRIPT_SHA256, "Unexpected metadata module revision")
        verify(args.metadata_script, args.metadata_script_sha256, tracked)
        verify(args.application_receipt, args.application_receipt_sha256, tracked)
        parent = json.loads(args.metadata_report.read_text(encoding="utf-8"))
        application = json.loads(args.application_receipt.read_text(encoding="utf-8"))
        filtered = application["outputs"]["filtered_pool"]
        require(isinstance(filtered, dict), "outputs.filtered_pool must be a provenance object")
        pool_path = Path(filtered["path"])
        verify(pool_path, filtered["sha256"], tracked, filtered["bytes"])
        mapping = {}
        for item in args.source_collection:
            source, separator, number = item.partition("=")
            require(separator and source not in mapping and number.isdigit(), "Invalid or duplicate source mapping")
            mapping[source] = int(number)
        require(mapping == REQUIRED_MAPPING, "All four explicit current-pool source mappings are required")
        sources = parent["sources"]
        require(isinstance(sources, dict), "Cached report sources must be an object")
        stage = "verify_cached_sources"
        for label, source in sources.items():
            verify(Path(source["path"]), source["sha256"], tracked, source["bytes"])
        stage = "offline_import_and_current_pool_join"
        with ExitStack() as stack:
            for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                           "urllib.request.urlopen", "urllib.request.OpenerDirector.open"):
                stack.enter_context(patch(target, no_network))
            specification = importlib.util.spec_from_file_location("v8_pinned_cached_metadata", args.metadata_script)
            require(specification is not None and specification.loader is not None, "Cannot import pinned module")
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            pool, schema = module.load_pool(pool_path, mapping)
            require(len(pool) == args.expected_pool_count, "Unexpected filtered pool row count")
            metadata = {}
            source_descriptions = {}
            for collection in (63, 65, 70, 212):
                label = str(collection)
                source = sources[label]
                records, description = module.load_metadata(Path(source["path"]), "collection_" + label)
                metadata[collection] = records
                source_descriptions[label] = {"cached_provenance": source, "current_parse": description}
            require(len(metadata[63]) == 2594, "Cached collection 63 must contain 2594 images")
            eval_ids = set(metadata[63])
            pool_ids = {row["canonical_id"] for row in pool}
            references = {}
            for label, (_, role) in module.CHALLENGE_SOURCES.items():
                source = sources[label]
                require(source["role"] == role, "Cached challenge role mismatch: " + label)
                if role == "duplicate_list_reference":
                    description = module.duplicate_list_summary(Path(source["path"]), pool_ids, eval_ids)
                    references[label] = description
                else:
                    records, description = module.load_metadata(Path(source["path"]), label)
                    references[label] = module.reference_joins(records, pool, eval_ids, label, role)
                source_descriptions[label] = {"cached_provenance": source, "current_parse": description}
            require(set(sources) == {"63", "65", "70", "212"} | set(module.CHALLENGE_SOURCES),
                    "Unexpected cached source inventory")
            report = module.build_report(pool, metadata, mapping)
        report.update({"schema": "v8_cached_current_pool_group_metadata.v1", "schema_version": 1,
            "status": "cached_metadata_current_pool_join_completed", "pool_inventory": {**tracked[str(pool_path.resolve())], "schema": schema},
            "sources": source_descriptions, "challenge_reference_joins": references,
            "parent_metadata_report": tracked[str(args.metadata_report.resolve())],
            "snapshot_policy": "Historical cached retrieval dates retained; no new retrieval or image/ZIP proof.",
            "scope": "Current inventory joins and same-string group candidates only; labels ignored for learning.",
            "limitations": ["Zero group comparisons do not prove clinical independence.",
                             "No image reads, ZIP decoding, pHash recalculation or HTTP requests.",
                             "Historical source dates do not certify current Archive metadata."],
            **closed_flags()})
        stage = "recheck_inputs_before_publication"
        for original in tracked.values():
            require(identity(original["path"]) == original, "Input changed during audit: " + original["path"])
        require(identity(Path(__file__)) == code, "Audit code changed during execution")
        output = publish(args.output_dir / "cached_group_metadata_audit.json", report)
        receipt = {"schema": "v8_cached_group_metadata_receipt.v1", "schema_version": 1,
                   "status": report["status"], "code": code, "inputs": list(tracked.values()),
                   "report": output, **closed_flags()}
        final = publish(args.output_dir / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": final, "pool_rows": len(pool), **closed_flags()}))
        return 0
    except Exception as error:
        failure = {"schema": "v8_cached_group_metadata_receipt.v1", "schema_version": 1,
                   "status": "failed_closed", "stage": stage, "error_type": type(error).__name__,
                   "diagnostic": str(error) if isinstance(error, (ValueError, KeyError, OSError)) else "Input/schema/IO failure; inspect pinned inputs at this stage",
                   "inputs_verified_before_failure": list(tracked.values()), **closed_flags()}
        if created:
            publish(args.output_dir / "receipt.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
