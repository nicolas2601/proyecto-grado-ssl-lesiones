#!/usr/bin/env python3
"""Record conservative training-path decisions from saved pHash screening evidence.

This produces an audit artifact on UNABIA. Similarity does not establish duplicate
identity, lesion identity, or patient independence. No learning manifest is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PAIR_FIELDS = (
    "eval_id", "eval_member", "train_id", "train_path",
    "hamming_distance_63bit", "exact_file_sha256_match",
)
FLAGS = {
    "executable": False, "training_consumable": False, "manifest_frozen": False,
    "execution_enabled": False, "launch_ready": False, "training_started": False,
    "patient_independence_accredited": False,
    "patient_lesion_independence": "unresolved",
}
POLICY = "exclude_training_path_conservatively"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nonempty(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"Missing or invalid {label}")
    return value


def base_id(value: Any) -> str:
    return nonempty(value, "image ID").removesuffix("_downsampled")


def adjudicate_pairs(candidate_pairs: list[dict[str, Any]],
                     pool_records: list[dict[str, Any]],
                     evaluation_images: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide eligibility for every edge while leaving identity unverified.

    Validate the complete batch before returning any decision. Inputs are never
    modified, and distinct evaluation edges may exclude the same training path.
    """
    require(isinstance(candidate_pairs, list), "Expected a candidate pair list")
    require(isinstance(pool_records, list), "Expected a pool record list")
    require(isinstance(evaluation_images, list), "Expected an evaluation image list")
    pool: dict[str, dict[str, Any]] = {}
    for row in pool_records:
        require(isinstance(row, dict), "Invalid pool record")
        path = nonempty(row.get("image_path"), "pool image path")
        nonempty(row.get("image_id"), "pool image ID")
        require(path not in pool, "Ambiguous pool path mapping")
        pool[path] = row
    evaluation: dict[str, str] = {}
    for row in evaluation_images:
        require(isinstance(row, dict), "Invalid evaluation record")
        image_id = nonempty(row.get("eval_id"), "evaluation ID")
        member = nonempty(row.get("eval_member"), "evaluation member")
        require(image_id not in evaluation, "Ambiguous evaluation ID mapping")
        evaluation[image_id] = member
    seen: set[tuple[str, str]] = set()
    for row in candidate_pairs:
        require(isinstance(row, dict) and set(PAIR_FIELDS) <= row.keys(),
                "Missing pHash pair evidence")
        image_id = nonempty(row["eval_id"], "pair evaluation ID")
        path = nonempty(row["train_path"], "pair training path")
        require(image_id in evaluation and row["eval_member"] == evaluation[image_id],
                "Unmapped evaluation pair")
        require(path in pool and row["train_id"] == base_id(pool[path]["image_id"]),
                "Unmapped training pair")
        distance = row["hamming_distance_63bit"]
        require(type(distance) is int and 0 <= distance <= 8,
                "Invalid saved pHash distance")
        require(row["exact_file_sha256_match"] is False,
                "Expected screening candidates without exact SHA256 matches")
        key = (image_id, path)
        require(key not in seen, "Duplicate pHash pair edge")
        seen.add(key)
    decisions = [
        {**deepcopy(row), "disposition": POLICY,
         "duplicate_identity_status": "unverified", "exclusion_applied": False,
         "decision_basis": "conservative_eligibility_from_saved_phash_screen"}
        for row in candidate_pairs
    ]
    return {"pair_decisions": decisions,
            "excluded_training_paths": sorted({key[1] for key in seen}), **FLAGS}


def read_pinned(path: Path, expected_hash: str) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected_hash.lower(), f"SHA256 mismatch: {path}")
    return payload, {"path": str(path.resolve()), "sha256": digest, "bytes": len(payload)}


def read_register_source(path: Path, reference: Any) -> tuple[bytes, dict[str, Any]]:
    require(isinstance(reference, dict), "Missing register source reference")
    require(Path(nonempty(reference.get("path"), "source path")).resolve() == path.resolve(),
            "Input path disagrees with pinned register source")
    payload, source = read_pinned(path, nonempty(reference.get("sha256"), "source hash"))
    require(type(reference.get("bytes")) is int and source["bytes"] == reference["bytes"],
            "Register source byte count mismatch")
    return payload, source


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "rows": len(rows), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def run(register_path: Path, report_path: Path, pool_path: Path, output: Path,
        register_sha256: str, report_sha256: str) -> dict[str, Any]:
    require(not output.exists() and not output.is_symlink(), "Output already exists")
    register_bytes, register_source = read_pinned(register_path, register_sha256)
    register = json.loads(register_bytes)
    require(register.get("schema") == "v8_exclusion_review_register.v1", "Unknown register schema")
    require(register.get("consistency", {}).get("all_checks_passed") is True,
            "Unverified register consistency")
    for key in ("executable", "training_consumable", "manifest_frozen",
                "execution_enabled", "launch_ready", "training_started"):
        require(register.get(key) is False, f"Register audit-only flag changed: {key}")
    sources = register.get("sources", {})
    pool_bytes, pool_source = read_register_source(pool_path, sources.get("pool_records"))
    report_bytes, report_source = read_register_source(report_path, sources.get("phash_report"))
    require(report_source["sha256"] == report_sha256.lower(), "Explicit pHash pin mismatch")
    pool = [json.loads(line) for line in pool_bytes.splitlines() if line.strip()]
    report = json.loads(report_bytes)
    pairs = report.get("screen", {}).get("candidate_pairs")
    result = adjudicate_pairs(pairs, pool, register.get("evaluation_images"))
    saved_pairs = register.get("phash_candidate_pairs")
    require(isinstance(saved_pairs, list) and len(saved_pairs) == len(pairs),
            "Register pHash pair count mismatch")
    saved = {(row["eval_id"], row["train_path"]): row for row in saved_pairs}
    require(len(saved) == len(saved_pairs), "Duplicate pair in pinned register")
    for row in pairs:
        recorded = saved.get((row["eval_id"], row["train_path"]), {})
        require(all(recorded.get(key) == row[key] for key in PAIR_FIELDS),
                "pHash screen and pinned register pair evidence disagree")
        require(recorded.get("status") == "review_pending"
                and recorded.get("exclusion_applied") is False,
                "Pinned register pair was already adjudicated or applied")
    pool_by_path = {row["image_path"]: row for row in pool}
    eval_members = {row["eval_id"]: row["eval_member"] for row in register["evaluation_images"]}
    proposals = register.get("base_id_exclusion_proposals")
    require(isinstance(proposals, list), "Missing base-ID exclusion proposals")
    base_paths: set[str] = set()
    for proposal in proposals:
        row = proposal.get("pool_record", {})
        path = row.get("image_path")
        require(path in pool_by_path and row == pool_by_path[path], "Unmapped base-ID proposal")
        image_id = base_id(row["image_id"])
        require(proposal.get("eval_id") == image_id and image_id in eval_members
                and proposal.get("eval_member") == eval_members[image_id],
                "Base-ID proposal evaluation mismatch")
        require(proposal.get("status") == "proposed_required_not_applied"
                and proposal.get("exclusion_applied") is False,
                "Base-ID proposal was already applied or changed")
        require(path not in base_paths, "Duplicate base-ID exclusion proposal")
        base_paths.add(path)
    require(base_paths == {path for path, row in pool_by_path.items()
                           if base_id(row["image_id"]) in eval_members},
            "Incomplete base-ID exclusion coverage")
    phash_paths = set(result["excluded_training_paths"])
    all_paths = base_paths | phash_paths
    path_decisions = [
        {"train_path": path, "train_id": base_id(pool_by_path[path]["image_id"]),
         "dataset": pool_by_path[path]["dataset"], "disposition": POLICY,
         "duplicate_identity_status": "unverified", "exclusion_applied": False}
        for path in result["excluded_training_paths"]
    ]
    per_source = {}
    for dataset, count in sorted(Counter(row["dataset"] for row in pool).items()):
        excluded = sum(pool_by_path[path]["dataset"] == dataset for path in all_paths)
        per_source[dataset] = {"before": count, "excluded_by_union": excluded,
                               "remaining_candidate_count": count - excluded}
    script = Path(__file__).resolve()
    code_bytes = script.read_bytes()
    receipt = {
        "schema": "v8_phash_conservative_adjudication.v1", **FLAGS,
        "artifact_role": "audit_only_conservative_path_eligibility_decisions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": POLICY, "duplicate_identity_status": "unverified",
        "decision_scope": "path_eligibility_only_no_visual_or_clinical_identity_adjudication",
        "exclusions_applied_to_inventory": False,
        "sources": {"register": register_source, "phash_report": report_source,
                    "pool_records": pool_source},
        "code": {"path": str(script), "bytes": len(code_bytes),
                 "sha256": hashlib.sha256(code_bytes).hexdigest()},
        "counts": {"pool_records": len(pool), "candidate_pair_decisions": len(pairs),
                   "candidate_evaluation_ids": len({row["eval_id"] for row in pairs}),
                   "phash_excluded_training_paths": len(phash_paths),
                   "base_id_exclusion_proposals": len(base_paths),
                   "union_excluded_training_paths": len(all_paths),
                   "remaining_candidate_count": len(pool) - len(all_paths)},
        "per_source": per_source,
    }
    # Exclusive reservation prevents accidental overwrite. Receipt is the completion
    # marker: an interrupted directory without receipt.json is never complete.
    output.mkdir(parents=False, exist_ok=False)
    receipt["outputs"] = {
        "pairs": write_jsonl(output / "pair_decisions.jsonl", result["pair_decisions"]),
        "paths": write_jsonl(output / "training_path_decisions.jsonl", path_decisions),
    }
    pending_receipt = output / ".receipt.pending"
    with pending_receipt.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending_receipt.rename(output / "receipt.json")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True, type=Path)
    parser.add_argument("--phash-report", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--register-sha256", required=True)
    parser.add_argument("--phash-report-sha256", required=True)
    args = parser.parse_args()
    try:
        receipt = run(args.register, args.phash_report, args.pool, args.output_dir,
                      args.register_sha256, args.phash_report_sha256)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Adjudication aborted: {error}\n")
    print(json.dumps({"receipt": str((args.output_dir / "receipt.json").resolve()),
                      "counts": receipt["counts"], **FLAGS}, sort_keys=True))


if __name__ == "__main__":
    main()
