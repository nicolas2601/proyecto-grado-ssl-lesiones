#!/usr/bin/env python3
"""Build an audit-only V8 exclusion proposal from existing cluster evidence.

This does not produce a training manifest, apply exclusions, adjudicate pHash
pairs, or establish patient/lesion independence. Run against the saved inputs
on UNABIA; image pixels are not decoded and pHash is not recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED = {
    "evaluation_images": 2594,
    "pool_records": 40896,
    "exact_stem_overlaps": 583,
    "downsampled_overlaps": 1334,
    "base_id_overlaps": 1917,
    "phash_candidate_pairs": 15702,
    "phash_unique_pool_paths": 4195,
    "phash_unique_evaluation_ids": 544,
}
ID_PATTERN = re.compile(r"ISIC_[0-9]+\Z")
SHA_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
DERIVATIVE_SUFFIX = "_downsampled"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nonempty(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"Invalid {label}")
    return value


def stem(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).stem


def base_id(image_id: str) -> str:
    if image_id.endswith(DERIVATIVE_SUFFIX):
        return image_id[: -len(DERIVATIVE_SUFFIX)]
    return image_id


def provenance(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_pool(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    image_ids: set[str] = set()
    fields = {"bytes", "dataset", "height", "image_id", "image_path", "image_sha256", "width"}
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank pool record at line {number}")
            record = json.loads(line)
            require(isinstance(record, dict) and fields <= record.keys(), f"Invalid pool schema at line {number}")
            image_path = nonempty(record["image_path"], f"pool path at line {number}")
            image_id = nonempty(record["image_id"], f"pool image_id at line {number}")
            nonempty(record["dataset"], f"pool dataset at line {number}")
            require(image_id == stem(image_path), f"Pool image_id/path disagreement at line {number}")
            require(isinstance(record["image_sha256"], str) and SHA_PATTERN.fullmatch(record["image_sha256"]) is not None, f"Invalid pool SHA256 at line {number}")
            for field in ("bytes", "width", "height"):
                require(type(record[field]) is int and record[field] > 0, f"Invalid pool {field} at line {number}")
            require(image_path not in records, f"Duplicate pool path at line {number}")
            require(image_id not in image_ids, f"Duplicate pool image_id at line {number}")
            records[image_path] = record
            image_ids.add(image_id)
    return records


def load_evaluation(path: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or PurePosixPath(info.filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image_id = stem(info.filename)
            require(ID_PATTERN.fullmatch(image_id) is not None, f"Unexpected evaluation image member: {info.filename}")
            require(image_id not in members, f"Duplicate evaluation ID: {image_id}")
            require(info.file_size > 0, f"Empty evaluation member: {info.filename}")
            members[image_id] = info.filename
    return members


def load_pairs(path: Path, pool: dict[str, dict[str, Any]], evaluation: dict[str, str]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        report = json.load(handle)
    require(isinstance(report, dict) and isinstance(report.get("screen"), dict), "Missing pHash screen object")
    raw_pairs = report["screen"].get("candidate_pairs")
    require(isinstance(raw_pairs, list), "Missing pHash screen.candidate_pairs list")
    required_fields = {"eval_id", "eval_member", "exact_file_sha256_match", "hamming_distance_63bit", "train_id", "train_path"}
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for number, pair in enumerate(raw_pairs, 1):
        require(isinstance(pair, dict) and required_fields <= pair.keys(), f"Invalid pHash pair schema at row {number}")
        eval_id = nonempty(pair["eval_id"], f"pHash eval_id at row {number}")
        train_path = nonempty(pair["train_path"], f"pHash train_path at row {number}")
        require(eval_id in evaluation, f"Unknown pHash evaluation ID at row {number}")
        require(train_path in pool, f"Unknown pHash pool path at row {number}")
        eval_member = nonempty(pair["eval_member"], f"pHash eval_member at row {number}")
        require(eval_member == evaluation[eval_id], f"pHash evaluation member disagrees with ZIP at row {number}")
        require(pair["train_id"] == base_id(pool[train_path]["image_id"]), f"pHash train_id disagrees with pool base ID at row {number}")
        distance = pair["hamming_distance_63bit"]
        require(type(distance) is int and 0 <= distance <= 8, f"Invalid pHash distance at row {number}")
        require(type(pair["exact_file_sha256_match"]) is bool, f"Invalid pHash exact-match flag at row {number}")
        require(not pair["exact_file_sha256_match"], f"Unexpected exact SHA256 match at pHash row {number}; evidence requires review")
        require(base_id(pool[train_path]["image_id"]) not in evaluation, f"pHash pool contains reserved base ID at row {number}")
        key = (eval_id, train_path)
        require(key not in seen, f"Duplicate pHash evaluation/pool pair at row {number}")
        seen.add(key)
        pairs.append({
            "eval_id": eval_id,
            "eval_member": eval_member,
            "train_id": pair["train_id"],
            "train_path": train_path,
            "hamming_distance_63bit": distance,
            "exact_file_sha256_match": False,
            "status": "review_pending",
            "exclusion_applied": False,
        })
    return sorted(pairs, key=lambda item: (item["eval_id"], item["hamming_distance_63bit"], item["train_path"]))


def build_register(pool: dict[str, dict[str, Any]], evaluation: dict[str, str], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    proposals = []
    for image_path, record in sorted(pool.items()):
        eval_id = base_id(record["image_id"])
        if eval_id in evaluation:
            proposals.append({
                "eval_id": eval_id,
                "eval_member": evaluation[eval_id],
                "match_kind": "exact_stem" if eval_id == record["image_id"] else "terminal_downsampled_derivative",
                "pool_record": record,
                "status": "proposed_required_not_applied",
                "exclusion_applied": False,
            })
    kind_counts = Counter(item["match_kind"] for item in proposals)
    observed = {
        "evaluation_images": len(evaluation),
        "pool_records": len(pool),
        "exact_stem_overlaps": kind_counts["exact_stem"],
        "downsampled_overlaps": kind_counts["terminal_downsampled_derivative"],
        "base_id_overlaps": len(proposals),
        "phash_candidate_pairs": len(pairs),
        "phash_unique_pool_paths": len({pair["train_path"] for pair in pairs}),
        "phash_unique_evaluation_ids": len({pair["eval_id"] for pair in pairs}),
    }
    for name, expected in EXPECTED.items():
        require(observed[name] == expected, f"Evidence count mismatch for {name}: expected {expected}, got {observed[name]}")
    require(all(item["pool_record"]["dataset"] == "isic2019" for item in proposals), "Reserved base-ID overlaps unexpectedly include another dataset")
    require(len({item["eval_id"] for item in proposals}) == EXPECTED["base_id_overlaps"], "Multiple overlap records share an evaluation base ID")
    candidate_paths = sorted({pair["train_path"] for pair in pairs})
    return {
        "schema": "v8_exclusion_review_register.v1",
        "artifact_role": "audit_only_exclusion_proposals_and_unadjudicated_candidates",
        "executable": False,
        "training_consumable": False,
        "manifest_frozen": False,
        "patient_lesion_independence": "unresolved",
        "patient_lesion_groups_preserved": False,
        "execution_enabled": False,
        "launch_ready": False,
        "training_started": False,
        "policy": {
            "evaluation_use": "Reserve all ISIC2018 Task 1 images for evaluation; expert masks are for scoring only.",
            "base_id_proposals": "Exclude these records from all learning exposure before freezing final manifests; proposals have not been applied.",
            "phash_candidates": "Similarity candidates require adjudication; distance alone does not confirm duplication or authorize exclusion.",
            "independence_blocker": "Image-ID and similarity screening cannot establish patient/lesion independence without group evidence across sources.",
            "historical_evaluation": "ISIC2018 is a previously consulted benchmark, not an untouched final test.",
            "training_contract": "Do not pass this review register to training or treat it as a frozen eligibility manifest.",
        },
        "consistency": {"expected": EXPECTED, "observed": observed, "all_checks_passed": True},
        "base_id_exclusion_proposals": proposals,
        "phash_candidate_pairs": pairs,
        "phash_pool_records": [pool[path] for path in candidate_paths],
        "evaluation_images": [{"eval_id": image_id, "eval_member": evaluation[image_id]} for image_id in sorted(evaluation)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-records", type=Path, required=True, help="Saved original_records.jsonl inventory")
    parser.add_argument("--evaluation-zip", type=Path, required=True, help="Official ISIC2018 Task 1 Training Input ZIP")
    parser.add_argument("--phash-report", type=Path, required=True, help="Saved full pHash report; no pHash computation is performed")
    parser.add_argument("--output", type=Path, required=True, help="New audit JSON path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        inputs = {"pool_records": args.pool_records, "evaluation_zip": args.evaluation_zip, "phash_report": args.phash_report}
        require(not args.output.exists(), "Output already exists; choose a new audit artifact path")
        resolved_inputs = {path.resolve() for path in inputs.values()}
        require(len(resolved_inputs) == 3, "Primary inputs must be distinct files")
        require(args.output.resolve() not in resolved_inputs, "Output cannot replace an input")
        require(all(path.is_file() for path in inputs.values()), "Every input must be an existing file")
        before = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in inputs.items()}
        sources = {name: provenance(path) for name, path in inputs.items()}
        pool = load_pool(args.pool_records)
        evaluation = load_evaluation(args.evaluation_zip)
        pairs = load_pairs(args.phash_report, pool, evaluation)
        register = build_register(pool, evaluation, pairs)
        after = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in inputs.items()}
        require(before == after, "An input changed while reading; no register was written")
        register["sources"] = sources
        payload = json.dumps(register, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(json.dumps({"output": str(args.output.resolve()), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "counts": register["consistency"]["observed"], "executable": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Review register not produced: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
