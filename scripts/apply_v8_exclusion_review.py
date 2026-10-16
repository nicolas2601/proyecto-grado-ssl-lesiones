#!/usr/bin/env python3
"""Apply recorded exclusions to a provisional V8 inventory on UNABIA.

Consumes saved, hash-pinned evidence. An optional externally pinned pHash receipt
can add conservative training-path exclusions while keeping identity unverified.
It does not decode images, recompute pHash, certify independence, freeze manifests,
or enable training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from build_v8_exclusion_review import (
    base_id, load_pool, nonempty, provenance, require,
)


PAIR_FIELDS = (
    "eval_id", "eval_member", "train_id", "train_path",
    "hamming_distance_63bit", "exact_file_sha256_match",
)
LEDGER_FIELDS = (*PAIR_FIELDS, "in_nearest_queue", "status", "decision", "reviewer",
                 "reviewed_at", "evidence", "exclusion_applied")
REQUIRED_EXCLUSIONS = 1917


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def file_state(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def verify_source(reference: dict[str, Any], label: str,
                  input_stats: dict[Path, tuple[int, int]]) -> dict[str, Any]:
    require(isinstance(reference, dict), f"Invalid source reference: {label}")
    path = Path(nonempty(reference.get("path"), f"{label} path"))
    expected_hash = nonempty(reference.get("sha256"), f"{label} SHA256")
    require(type(reference.get("bytes")) is int, f"Missing source byte count: {label}")
    before = file_state(path)
    actual = provenance(path)
    require(actual["sha256"] == expected_hash.lower(), f"SHA256 mismatch: {label} ({path})")
    require(actual["bytes"] == reference["bytes"], f"Source byte count mismatch: {label}")
    require(file_state(path) == before, f"Source changed while checking SHA256: {label}")
    input_stats[path.resolve()] = before
    return actual


def pair_key(pair: dict[str, Any]) -> tuple[str, str]:
    return (nonempty(pair.get("eval_id"), "pair eval_id"),
            nonempty(pair.get("train_path"), "pair train_path"))


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"Invalid {label} row at line {line_number}")
            rows.append(row)
    return rows


def same_source(reference: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    require(isinstance(reference, dict), f"Invalid source reference: {label}")
    require(Path(nonempty(reference.get("path"), f"{label} path")).resolve()
            == Path(expected["path"]).resolve(), f"{label} path disagrees with pinned source")
    require(reference.get("sha256") == expected.get("sha256"),
            f"{label} SHA256 disagrees with pinned source")
    require(reference.get("bytes") == expected.get("bytes"),
            f"{label} byte count disagrees with pinned source")


def derive_nearest_queue(report: dict[str, Any]) -> dict[str, Any]:
    screen = report.get("screen")
    require(isinstance(screen, dict), "Missing pHash screen section in current report")
    pairs = screen.get("candidate_pairs")
    require(isinstance(pairs, list), "Missing pHash candidate pairs in current report")
    minimum: dict[str, int] = {}
    for pair in pairs:
        require(isinstance(pair, dict), "Invalid pHash report pair")
        eval_id = nonempty(pair.get("eval_id"), "pHash report evaluation ID")
        distance = pair.get("hamming_distance_63bit")
        require(type(distance) is int and distance >= 0,
                "Invalid pHash report distance")
        minimum[eval_id] = min(distance, minimum.get(eval_id, distance))
    return {"closest_pair_records": [
        {field: pair[field] for field in PAIR_FIELDS}
        for pair in pairs if pair["hamming_distance_63bit"] == minimum[pair["eval_id"]]
    ]}


def assert_readiness_gates_false(value: Any) -> None:
    flags = {"executable", "training_consumable", "manifest_frozen", "execution_enabled",
             "execution_ready", "launch_ready", "training_started",
             "patient_independence_accredited", "clinical_independence_accredited"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in flags:
                require(child is False, f"Readiness gate is not false: {key}")
            assert_readiness_gates_false(child)
    elif isinstance(value, list):
        for child in value:
            assert_readiness_gates_false(child)


def validate_phash_receipt(
    receipt: dict[str, Any], receipt_reference: dict[str, Any],
    register: dict[str, Any], pool: dict[str, dict[str, Any]], base_exclusions: set[str],
    verified_sources: dict[str, dict[str, Any]],
    input_stats: dict[Path, tuple[int, int]],
) -> tuple[set[str], dict[str, Any]]:
    require(receipt.get("schema") == "v8_phash_conservative_adjudication.v1",
            "Unknown pHash adjudication receipt schema")
    require(receipt.get("artifact_role") == "audit_only_conservative_path_eligibility_decisions"
            and receipt.get("policy") == "exclude_training_path_conservatively"
            and receipt.get("decision_scope")
            == "path_eligibility_only_no_visual_or_clinical_identity_adjudication"
            and receipt.get("duplicate_identity_status") == "unverified",
            "pHash receipt does not declare the expected conservative audit-only policy")
    require(receipt.get("exclusions_applied_to_inventory") is False,
            "pHash receipt claims source inventory was modified")
    for flag in ("executable", "training_consumable", "manifest_frozen", "execution_enabled",
                 "launch_ready", "training_started", "patient_independence_accredited"):
        require(receipt.get(flag) is False, f"pHash receipt audit-only flag changed: {flag}")
    require(receipt.get("patient_lesion_independence") == "unresolved",
            "pHash receipt no longer declares patient/lesion independence unresolved")

    source_refs = receipt.get("sources")
    require(isinstance(source_refs, dict)
            and set(source_refs) == {"register", "phash_report", "pool_records"},
            "pHash receipt source set is incomplete or unexpected")
    for name, expected in (("register", verified_sources["register"]),
                           ("phash_report", verified_sources["phash_report"]),
                           ("pool_records", verified_sources["pool_records"])):
        reference = source_refs[name]
        same_source(reference, expected, f"pHash receipt {name}")

    report = load_json(Path(verified_sources["phash_report"]["path"]))
    screen = report.get("screen")
    require(isinstance(screen, dict), "Missing pHash screen section in pinned report")
    report_pairs = screen.get("candidate_pairs")
    register_pairs = register.get("phash_candidate_pairs")
    require(isinstance(report_pairs, list) and isinstance(register_pairs, list),
            "Missing pHash candidate pairs in report or register")
    expected_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in report_pairs:
        require(isinstance(row, dict) and set(PAIR_FIELDS) <= row.keys(),
                "Invalid pHash report pair schema")
        key = pair_key(row)
        nonempty(row.get("eval_member"), "pHash report evaluation member")
        nonempty(row.get("train_id"), "pHash report training ID")
        require(type(row.get("hamming_distance_63bit")) is int
                and 0 <= row["hamming_distance_63bit"] <= 8,
                "Invalid pHash report distance")
        require(row.get("exact_file_sha256_match") is False,
                "Unexpected exact SHA256 match in pHash report")
        require(key not in expected_pairs, "Duplicate pHash report pair")
        expected_pairs[key] = row
    require(all(isinstance(row, dict) for row in register_pairs),
            "Invalid pHash register pair schema")
    registered_pairs = {pair_key(row): row for row in register_pairs}
    require(len(registered_pairs) == len(register_pairs)
            and set(registered_pairs) == set(expected_pairs),
            "pHash report and register pair coverage mismatch")
    for key, row in expected_pairs.items():
        registered = registered_pairs[key]
        require(all(registered.get(field) == row.get(field) for field in PAIR_FIELDS),
                "pHash report and register pair evidence mismatch")

    outputs = receipt.get("outputs")
    require(isinstance(outputs, dict) and set(outputs) == {"pairs", "paths"},
            "pHash receipt output set is incomplete or unexpected")
    decision_sources = {}
    for name in ("pairs", "paths"):
        reference = outputs[name]
        require(isinstance(reference, dict), f"Invalid pHash {name} output reference")
        require(type(reference.get("rows")) is int, f"Missing pHash {name} row count")
        decision_sources[name] = verify_source(reference, f"pHash {name} decisions", input_stats)

    pair_decisions = load_jsonl(Path(decision_sources["pairs"]["path"]), "pHash pair decision")
    path_decisions = load_jsonl(Path(decision_sources["paths"]["path"]), "pHash path decision")
    require(len(pair_decisions) == len(expected_pairs)
            and outputs["pairs"]["rows"] == len(expected_pairs),
            "pHash pair decision count mismatch")
    seen_decisions: set[tuple[str, str]] = set()
    excluded_paths: set[str] = set()
    for decision in pair_decisions:
        key = pair_key(decision)
        require(key not in seen_decisions and key in expected_pairs,
                "Duplicate or unknown pHash pair decision")
        source_pair = expected_pairs[key]
        require(type(decision.get("hamming_distance_63bit")) is int
                and 0 <= decision["hamming_distance_63bit"] <= 8
                and decision.get("exact_file_sha256_match") is False,
                "Invalid pHash pair decision evidence types")
        require(all(decision.get(field) == source_pair.get(field) for field in PAIR_FIELDS),
                "pHash pair decision disagrees with report evidence")
        require(decision.get("disposition") == "exclude_training_path_conservatively"
                and decision.get("duplicate_identity_status") == "unverified"
                and decision.get("exclusion_applied") is False
                and decision.get("decision_basis")
                == "conservative_eligibility_from_saved_phash_screen",
                "pHash pair decision violates the conservative audit policy")
        seen_decisions.add(key)
        excluded_paths.add(key[1])
    require(seen_decisions == set(expected_pairs), "pHash pair decisions omit report evidence")

    expected_path_decisions = []
    for path in sorted(excluded_paths):
        require(path in pool, f"pHash decision path is absent from original pool: {path}")
        row = pool[path]
        expected_path_decisions.append({
            "train_path": path, "train_id": base_id(row["image_id"]),
            "dataset": row["dataset"], "disposition": "exclude_training_path_conservatively",
            "duplicate_identity_status": "unverified", "exclusion_applied": False,
        })
    require(path_decisions == expected_path_decisions
            and outputs["paths"]["rows"] == len(expected_path_decisions),
            "pHash path decisions disagree with report/register/pool evidence")
    require(not (excluded_paths & base_exclusions),
            "pHash decision path overlaps a reserved base-ID exclusion")

    counts = receipt.get("counts")
    expected_counts = {
        "pool_records": len(pool), "candidate_pair_decisions": len(expected_pairs),
        "candidate_evaluation_ids": len({key[0] for key in expected_pairs}),
        "phash_excluded_training_paths": len(excluded_paths),
        "base_id_exclusion_proposals": len(register.get("base_id_exclusion_proposals", [])),
        "union_excluded_training_paths": len(base_exclusions | excluded_paths),
        "remaining_candidate_count": len(pool) - len(base_exclusions | excluded_paths),
    }
    require(isinstance(counts, dict), "Missing pHash receipt counts")
    for name, expected in expected_counts.items():
        require(type(counts.get(name)) is int and counts[name] == expected,
                f"pHash receipt count mismatch: {name}")
    expected_per_source: dict[str, dict[str, int]] = {}
    for dataset, before in sorted(Counter(row["dataset"] for row in pool.values()).items()):
        excluded = sum(pool[path]["dataset"] == dataset
                       for path in base_exclusions | excluded_paths)
        expected_per_source[dataset] = {
            "before": before, "excluded_by_union": excluded,
            "remaining_candidate_count": before - excluded,
        }
    per_source = receipt.get("per_source")
    require(isinstance(per_source, dict) and set(per_source) == set(expected_per_source),
            "pHash receipt per-source inventory does not match original pool")
    for dataset, expected in expected_per_source.items():
        actual = per_source[dataset]
        require(isinstance(actual, dict) and set(actual) == set(expected)
                and all(type(actual[name]) is int and actual[name] == value
                        for name, value in expected.items()),
                f"pHash receipt per-source counts disagree: {dataset}")
    return excluded_paths, {
        "receipt": receipt_reference,
        "pair_decisions": decision_sources["pairs"],
        "path_decisions": decision_sources["paths"],
    }


def validate_register(
    register: dict[str, Any], pool: dict[str, dict[str, Any]],
    queue: dict[str, Any], register_reference: dict[str, Any],
    queue_reference: dict[str, Any],
) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    require(register.get("schema") == "v8_exclusion_review_register.v1", "Unknown register schema")
    require(register.get("consistency", {}).get("all_checks_passed") is True,
            "Register consistency was not verified")
    evaluation = register.get("evaluation_images")
    proposals = register.get("base_id_exclusion_proposals")
    pairs = register.get("phash_candidate_pairs")
    require(isinstance(evaluation, list) and isinstance(proposals, list) and isinstance(pairs, list),
            "Missing evaluation, exclusion, or pHash records")
    members = {nonempty(row.get("eval_id"), "evaluation ID"):
               nonempty(row.get("eval_member"), "evaluation member") for row in evaluation}
    require(len(members) == len(evaluation), "Duplicate evaluation ID")
    exclusions: set[str] = set()
    kinds: Counter[str] = Counter()
    for proposal in proposals:
        record = proposal.get("pool_record")
        require(isinstance(record, dict), "Invalid exclusion pool record")
        path = nonempty(record.get("image_path"), "exclusion pool path")
        require(path in pool and record == pool[path], f"Exclusion disagrees with original pool: {path}")
        image_id = base_id(record["image_id"])
        require(proposal.get("eval_id") == image_id and image_id in members,
                f"Exclusion is not linked to a full reserved base ID: {path}")
        require(proposal.get("eval_member") == members[image_id], "Exclusion evaluation member mismatch")
        kind = "exact_stem" if record["image_id"] == image_id else "terminal_downsampled_derivative"
        require(proposal.get("match_kind") == kind, "Exclusion match kind mismatch")
        require(proposal.get("status") == "proposed_required_not_applied"
                and proposal.get("exclusion_applied") is False, "Exclusion proposal is already applied or changed")
        require(path not in exclusions, "Duplicate exclusion path")
        exclusions.add(path)
        kinds[kind] += 1
    require(len(exclusions) == REQUIRED_EXCLUSIONS, "Expected exactly 1917 recorded base-ID exclusions")
    require(exclusions == {path for path, row in pool.items() if base_id(row["image_id"]) in members},
            "Recorded proposals do not cover every reserved base ID in the original pool")

    seen: dict[tuple[str, str], dict[str, Any]] = {}
    minimum: dict[str, int] = {}
    for pair in pairs:
        require(isinstance(pair, dict) and set(PAIR_FIELDS) <= pair.keys(), "Invalid pHash pair schema")
        key = pair_key(pair)
        eval_id, path = key
        require(key not in seen, "Duplicate pHash pair")
        require(eval_id in members and pair["eval_member"] == members[eval_id], "pHash evaluation mismatch")
        require(path in pool and path not in exclusions, "pHash path is unknown or already excluded by base ID")
        require(pair["train_id"] == base_id(pool[path]["image_id"]), "pHash training base ID mismatch")
        distance = pair["hamming_distance_63bit"]
        require(type(distance) is int and 0 <= distance <= 8, "Invalid saved pHash distance")
        require(pair["exact_file_sha256_match"] is False, "Unexpected exact SHA256 match in pending pHash evidence")
        require(pair.get("status") == "review_pending" and pair.get("exclusion_applied") is False,
                "pHash register contains previously applied or adjudicated records")
        seen[key] = pair
        minimum[eval_id] = min(distance, minimum.get(eval_id, distance))
    candidate_pool = register.get("phash_pool_records")
    require(isinstance(candidate_pool, list), "Missing saved pHash pool records")
    candidate_paths = [row.get("image_path") for row in candidate_pool]
    require(len(set(candidate_paths)) == len(candidate_paths)
            and set(candidate_paths) == {key[1] for key in seen}, "pHash pool record coverage mismatch")
    require(all(pool.get(row["image_path"]) == row for row in candidate_pool),
            "pHash pool records disagree with original inventory")
    nearest_rows = queue.get("closest_pair_records")
    require(isinstance(nearest_rows, list), "Missing saved closest_pair_records")
    nearest: set[tuple[str, str]] = set()
    for row in nearest_rows:
        key = pair_key(row)
        require(key not in nearest and key in seen, "Duplicate or unknown nearest-review pair")
        require(all(row.get(field) == seen[key][field] for field in PAIR_FIELDS), "Nearest pair evidence mismatch")
        nearest.add(key)
    require(nearest == {key for key, row in seen.items()
                        if row["hamming_distance_63bit"] == minimum[key[0]]},
            "Nearest queue does not preserve every tied nearest saved pair")
    require(len(nearest) == queue_reference.get("rows"), "Nearest queue row count mismatch")

    observed = {
        "pool_records": len(pool), "evaluation_images": len(evaluation),
        "base_id_overlaps": len(exclusions), "exact_stem_overlaps": kinds["exact_stem"],
        "downsampled_overlaps": kinds["terminal_downsampled_derivative"],
        "phash_candidate_pairs": len(pairs), "phash_unique_evaluation_ids": len(minimum),
        "phash_unique_pool_paths": len(candidate_paths),
    }
    for expected in (register["consistency"].get("observed"), register_reference.get("observed_counts")):
        require(isinstance(expected, dict), "Missing saved evidence counts")
        for name, value in expected.items():
            require(name in observed and observed[name] == value, f"Evidence count mismatch: {name}")
    ledger = [{**{field: pair[field] for field in PAIR_FIELDS},
               "in_nearest_queue": str(pair_key(pair) in nearest).lower(),
               "status": "review_pending", "decision": "", "reviewer": "",
               "reviewed_at": "", "evidence": "", "exclusion_applied": "false"}
              for pair in pairs]
    counts = {"pool_records_before": len(pool), "pool_records_after": len(pool) - len(exclusions),
              "base_id_exclusions_applied": len(exclusions),
              "exact_stem_exclusions_applied": kinds["exact_stem"],
              "downsampled_exclusions_applied": kinds["terminal_downsampled_derivative"],
              "phash_candidate_pairs": len(pairs), "phash_pending_pairs": len(pairs),
              "nearest_review_pairs": len(nearest), "phash_exclusions_applied": 0}
    return exclusions, ledger, counts


def apply(readiness_path: Path, output: Path, phash_receipt_path: Path | None = None,
          phash_receipt_sha256: str | None = None) -> dict[str, Any]:
    require(not output.exists() and not output.is_symlink(), "Output directory already exists; choose a new directory")
    require((phash_receipt_path is None) == (phash_receipt_sha256 is None),
            "pHash receipt path and SHA256 must be provided together")
    input_stats = {readiness_path.resolve(): file_state(readiness_path)}
    readiness_reference = provenance(readiness_path)
    readiness = load_json(readiness_path)
    if phash_receipt_path is not None:
        assert_readiness_gates_false(readiness)
    data = readiness["data"]
    historical_register_reference = data["step_1_freeze_gate_review"]["exclusion_review_register"]
    historical_queue_reference = data["isic2018_pool_phash_screen"]["nearest_review_queue"]
    readiness_reconciliation: dict[str, Any] | None = None
    if phash_receipt_path is None:
        register_reference = historical_register_reference
        queue_reference = historical_queue_reference
        register_source = verify_source({"path": register_reference["artifact_path"],
                                         "bytes": register_reference["artifact_bytes"],
                                         "sha256": register_reference["artifact_sha256"]},
                                        "exclusion register", input_stats)
        queue_source = verify_source(queue_reference, "nearest queue", input_stats)
        register = load_json(Path(register_source["path"]))
        queue = load_json(Path(queue_source["path"]))
        sources = register.get("sources")
        require(isinstance(sources, dict)
                and {"pool_records", "evaluation_zip", "phash_report"} <= sources.keys(),
                "Missing original input provenance")
        verified = {name: verify_source(reference, name, input_stats)
                    for name, reference in sources.items()}
        require(queue.get("source_report_sha256") == verified["phash_report"]["sha256"],
                "Nearest queue source report SHA256 mismatch")
        all_sources = {"readiness": readiness_reference, "register": register_source,
                       "nearest_queue": queue_source, **verified}
        pool = load_pool(Path(verified["pool_records"]["path"]))
        exclusions, ledger, counts = validate_register(
            register, pool, queue, register_reference, queue_reference)
    else:
        receipt_hash = nonempty(phash_receipt_sha256, "pHash adjudication receipt SHA256")
        receipt_reference = verify_source(
            {"path": str(phash_receipt_path), "bytes": phash_receipt_path.stat().st_size,
             "sha256": receipt_hash}, "pHash adjudication receipt", input_stats)
        adjudication_receipt = load_json(phash_receipt_path)
        receipt_source_refs = adjudication_receipt.get("sources")
        require(isinstance(receipt_source_refs, dict)
                and set(receipt_source_refs) == {"register", "phash_report", "pool_records"},
                "pHash receipt source set is incomplete or unexpected")
        current_sources = {name: verify_source(reference, f"receipt-pinned {name}", input_stats)
                           for name, reference in receipt_source_refs.items()}
        register_source = current_sources["register"]
        register = load_json(Path(register_source["path"]))
        sources = register.get("sources")
        require(isinstance(sources, dict)
                and set(sources) == {"pool_records", "evaluation_zip", "phash_report"},
                "Current register is missing original input provenance")
        same_source(sources["pool_records"], current_sources["pool_records"],
                    "Current register pool source")
        same_source(sources["phash_report"], current_sources["phash_report"],
                    "Current register pHash report source")
        verified = {
            "pool_records": current_sources["pool_records"],
            "phash_report": current_sources["phash_report"],
            "evaluation_zip": verify_source(sources["evaluation_zip"], "evaluation_zip", input_stats),
        }
        verified["register"] = register_source
        report = load_json(Path(verified["phash_report"]["path"]))
        queue = derive_nearest_queue(report)
        queue_reference = {"rows": len(queue["closest_pair_records"])}
        pool = load_pool(Path(verified["pool_records"]["path"]))
        current_register_reference = {
            "observed_counts": register.get("consistency", {}).get("observed")}
        exclusions, ledger, counts = validate_register(
            register, pool, queue, current_register_reference, queue_reference)
        historical_register_pin = {
            "path": historical_register_reference.get("artifact_path"),
            "bytes": historical_register_reference.get("artifact_bytes"),
            "sha256": historical_register_reference.get("artifact_sha256"),
        }
        current_register_pin = {
            "path": register_source["path"], "bytes": register_source["bytes"],
            "sha256": register_source["sha256"],
        }
        pin_differences = [key for key in ("path", "bytes", "sha256")
                           if historical_register_pin.get(key) != current_register_pin.get(key)]
        historical_counts = historical_register_reference.get("observed_counts", {})
        current_counts = register.get("consistency", {}).get("observed", {})
        count_differences = sorted(key for key in set(historical_counts) | set(current_counts)
                                   if historical_counts.get(key) != current_counts.get(key))
        discrepancies = []
        if pin_differences:
            discrepancies.append({"source": "exclusion_review_register",
                                  "different_fields": pin_differences})
        if count_differences:
            discrepancies.append({"source": "exclusion_review_register.observed_counts",
                                  "different_fields": count_differences})
        discrepancies.append({
            "source": "nearest_review_queue",
            "status": "historical_readiness_pin_not_consumed",
            "current_queue": "derived from the receipt-pinned pHash report",
        })
        readiness_reconciliation = {
            "mode": "receipt_pinned_current_sources",
            "historical_readiness_pins": {
                "exclusion_review_register": historical_register_pin,
                "nearest_review_queue": historical_queue_reference,
            },
            "current_register_source": current_register_pin,
            "readiness_register_pin_matches_current": not pin_differences,
            "readiness_observed_counts_match_current": not count_differences,
            "nearest_queue_rows_derived": len(queue["closest_pair_records"]),
            "discrepancies": discrepancies,
        }
        all_sources = {"readiness": readiness_reference, "register": register_source, **verified}
    if phash_receipt_path is None:
        readiness_reconciliation = None
    source_paths = [Path(reference["path"]).resolve() for reference in all_sources.values()]
    require(len(set(source_paths)) == len(source_paths), "Audit source paths must be distinct")
    require(output.resolve() not in source_paths, "Output directory cannot replace an input")
    phash_exclusions: set[str] = set()
    phash_sources: dict[str, Any] = {}
    if phash_receipt_path is not None:
        phash_exclusions, phash_sources = validate_phash_receipt(
            adjudication_receipt, receipt_reference, register, pool, exclusions,
            {"register": register_source, "phash_report": verified["phash_report"],
             "pool_records": verified["pool_records"]}, input_stats)
        for row in ledger:
            if row["train_path"] in phash_exclusions:
                row["decision"] = "exclude_training_path_conservatively"
                row["evidence"] = f"conservative_path_policy_receipt:{phash_sources['receipt']['sha256']}"
                row["exclusion_applied"] = "true"
        counts = dict(counts)
        counts["phash_exclusions_applied"] = len(phash_exclusions)
        counts["pool_records_after"] = len(pool) - len(exclusions | phash_exclusions)
        counts["union_exclusions_applied"] = len(exclusions | phash_exclusions)
    require(all(file_state(path) == stat for path, stat in input_stats.items()),
            "An input changed during validation")

    if phash_receipt_path is not None:
        all_sources.update({
            "phash_adjudication_receipt": phash_sources["receipt"],
            "phash_pair_decisions": phash_sources["pair_decisions"],
            "phash_path_decisions": phash_sources["path_decisions"],
        })
    source_paths = [Path(reference["path"]).resolve() for reference in all_sources.values()]
    require(len(set(source_paths)) == len(source_paths), "Audit source paths must be distinct")
    require(output.resolve() not in source_paths, "Output directory cannot replace an input")

    # Only a newly created directory is owned here; existing artifacts are never removed.
    created: list[Path] = []
    output.mkdir()
    try:
        filtered_name = ("training_pool_base_id_filtered.jsonl" if phash_receipt_path is None
                         else "training_pool_base_id_phash_eligibility_filtered.jsonl")
        filtered = output / filtered_name
        with filtered.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(filtered)
            for path, row in pool.items():
                if path not in exclusions and path not in phash_exclusions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        ledger_path = output / "phash_adjudication_ledger.csv"
        with ledger_path.open("x", encoding="utf-8", newline="") as handle:
            created.append(ledger_path)
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            writer.writerows(ledger)
        receipt = {
            "schema": ("v8_base_id_exclusion_application.v1" if phash_receipt_path is None
                       else "v8_provisional_exclusion_application.v2"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_role": "provisional_inventory_and_unadjudicated_similarity_ledger",
            "manifest_frozen": False, "training_consumable": False,
            "executable": False,
            "patient_independence_accredited": False, "patient_lesion_groups_preserved": False,
            "execution_enabled": False, "launch_ready": False, "training_started": False,
            "similarity_adjudication_complete": False, "learning_exposure_review_complete": False,
            "images_decoded": False, "expert_masks_read": False, "phash_recomputed": False,
            "original_inputs_modified": False, "counts": counts, "sources": all_sources,
            "outputs": {"filtered_pool": provenance(filtered), "phash_ledger": provenance(ledger_path)},
            "policy": ("Only recorded full base-ID exclusions applied. All saved pHash pairs remain pending. "
                      if phash_receipt_path is None else
                      "Recorded full base-ID exclusions and receipt-pinned conservative pHash path exclusions "
                      "were applied to this provisional output. pHash similarity does not establish duplicate, "
                      "lesion, or patient identity; clinical and patient/lesion independence remain unresolved. ")
                      + "Final manifests require similarity adjudication, exposure review, and mask validation. "
                      "Patient-level independence remains unaccredited.",
        }
        if phash_receipt_path is not None:
            receipt.update({
                "patient_lesion_independence": "unresolved",
                "clinical_independence_accredited": False,
                "execution_ready": False,
                "readiness_pin_reconciliation": readiness_reconciliation,
                "application_mode": "base_id_plus_conservative_phash_path_eligibility",
            })
        require(all(file_state(path) == stat for path, stat in input_stats.items()),
                "An input changed while writing provisional outputs")
        receipt_path = output / "receipt.json"
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(receipt_path)
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except Exception:
        for path in reversed(created):
            path.unlink()
        output.rmdir()
        raise
    return {"output_dir": str(output.resolve()), "receipt": provenance(receipt_path), "counts": counts,
            "manifest_frozen": False, "training_consumable": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True, help="Saved readiness.review.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New provisional output directory")
    parser.add_argument("--phash-adjudication-receipt", type=Path,
                        help="Optional conservative audit-only pHash receipt; uses its pinned current sources")
    parser.add_argument("--phash-adjudication-receipt-sha256",
                        help="Externally pinned SHA256; historical readiness pins are recorded, not consumed")
    args = parser.parse_args()
    try:
        print(json.dumps(apply(args.readiness, args.output_dir,
                               args.phash_adjudication_receipt,
                               args.phash_adjudication_receipt_sha256), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Exclusion application not produced: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
