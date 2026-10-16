#!/usr/bin/env python3
"""Apply recorded base-ID exclusions to a provisional V8 inventory on UNABIA.

Consumes saved, hash-pinned evidence. Does not decode images, recompute pHash,
adjudicate similarity, freeze manifests, or enable training.
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


def apply(readiness_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists() and not output.is_symlink(), "Output directory already exists; choose a new directory")
    input_stats = {readiness_path.resolve(): file_state(readiness_path)}
    readiness_reference = provenance(readiness_path)
    readiness = load_json(readiness_path)
    data = readiness["data"]
    register_reference = data["step_1_freeze_gate_review"]["exclusion_review_register"]
    queue_reference = data["isic2018_pool_phash_screen"]["nearest_review_queue"]
    register_source = verify_source({"path": register_reference["artifact_path"],
                                     "bytes": register_reference["artifact_bytes"],
                                     "sha256": register_reference["artifact_sha256"]},
                                    "exclusion register", input_stats)
    queue_source = verify_source(queue_reference, "nearest queue", input_stats)
    register = load_json(Path(register_source["path"]))
    queue = load_json(Path(queue_source["path"]))
    sources = register.get("sources")
    require(isinstance(sources, dict) and {"pool_records", "evaluation_zip", "phash_report"} <= sources.keys(),
            "Missing original input provenance")
    verified = {name: verify_source(reference, name, input_stats) for name, reference in sources.items()}
    require(queue.get("source_report_sha256") == verified["phash_report"]["sha256"],
            "Nearest queue source report SHA256 mismatch")
    all_sources = {"readiness": readiness_reference, "register": register_source,
                   "nearest_queue": queue_source, **verified}
    source_paths = [Path(reference["path"]) for reference in all_sources.values()]
    require(len(set(source_paths)) == len(source_paths), "Audit source paths must be distinct")
    require(output.resolve() not in source_paths, "Output directory cannot replace an input")
    pool = load_pool(Path(verified["pool_records"]["path"]))
    exclusions, ledger, counts = validate_register(register, pool, queue, register_reference, queue_reference)
    require(all(file_state(path) == stat for path, stat in input_stats.items()),
            "An input changed during validation")

    # Only a newly created directory is owned here; existing artifacts are never removed.
    created: list[Path] = []
    output.mkdir()
    try:
        filtered = output / "training_pool_base_id_filtered.jsonl"
        with filtered.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(filtered)
            for path, row in pool.items():
                if path not in exclusions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        ledger_path = output / "phash_adjudication_ledger.csv"
        with ledger_path.open("x", encoding="utf-8", newline="") as handle:
            created.append(ledger_path)
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            writer.writerows(ledger)
        receipt = {
            "schema": "v8_base_id_exclusion_application.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_role": "provisional_inventory_and_unadjudicated_similarity_ledger",
            "manifest_frozen": False, "training_consumable": False,
            "patient_independence_accredited": False, "patient_lesion_groups_preserved": False,
            "execution_enabled": False, "launch_ready": False, "training_started": False,
            "similarity_adjudication_complete": False, "learning_exposure_review_complete": False,
            "images_decoded": False, "expert_masks_read": False, "phash_recomputed": False,
            "original_inputs_modified": False, "counts": counts, "sources": all_sources,
            "outputs": {"filtered_pool": provenance(filtered), "phash_ledger": provenance(ledger_path)},
            "policy": "Only recorded full base-ID exclusions applied. All saved pHash pairs remain pending. "
                      "Final manifests require similarity adjudication, exposure review, and mask validation. "
                      "Patient-level independence remains unaccredited.",
        }
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
    args = parser.parse_args()
    try:
        print(json.dumps(apply(args.readiness, args.output_dir), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Exclusion application not produced: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
