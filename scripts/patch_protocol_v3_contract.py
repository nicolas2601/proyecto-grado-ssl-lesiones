"""Teach validate_v2_shared_pool_contract the v3 contract, additively.

The validator compares a manifest's ``pool_contract`` by full dict equality
against a single hardcoded constant, and it has exactly one caller
(``common/data.py``) which passes no constant. So the minimal change is at one
point: accept either the v2 contract or the v3 one, and nothing else.

This WIDENS an existing guard rather than adding a parallel path, which is the
honest way to describe it. The widening is bounded: two named, hashed contracts
are accepted, any third shape still raises, and the v2 constant is untouched so
every completed run validates exactly as before.

``V3_SHARED_POOL_CONTRACT`` is derived from the finished v3 manifest instead of
hand-written, because the comparison is full-equality and a hand-copied literal
drifts from the artifact it is supposed to pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STAGE_KEYS = (
    "stage",
    "stage_count",
    "stage_cohort_sha256",
    "stage_dataset_counts",
    "stage_source_manifest_sha256",
)

OLD_BODY = """    expected = dict(V2_SHARED_POOL_CONTRACT)
    expected.update({
        'stage': stage,
        'stage_count': V2_SHARED_POOL_CONTRACT[f'{stage}_count'],
        'stage_dataset_counts': V2_SHARED_POOL_CONTRACT[f'{stage}_dataset_counts'],
        'stage_source_manifest_sha256': V2_SHARED_POOL_CONTRACT[f'{stage}_source_manifest_sha256'],
        'stage_cohort_sha256': V2_SHARED_POOL_CONTRACT[f'{stage}_cohort_sha256'],
    })
    if contract != expected:
        raise ValueError('V2 shared-pool source, split, evaluation, or PH2 pins mismatch')"""

NEW_BODY = """    # Two pinned contracts are accepted: the original v2 pool and the v3
    # source-cohort-disjoint pool. Both are full-equality comparisons against a
    # constant, so a third shape still raises. v2 is unchanged, so every run that
    # validated before validates identically now.
    expected = None
    for _constant in (V2_SHARED_POOL_CONTRACT, V3_SHARED_POOL_CONTRACT):
        _candidate = dict(_constant)
        _candidate.update({
            'stage': stage,
            'stage_count': _constant[f'{stage}_count'],
            'stage_dataset_counts': _constant[f'{stage}_dataset_counts'],
            'stage_source_manifest_sha256': _constant[f'{stage}_source_manifest_sha256'],
            'stage_cohort_sha256': _constant[f'{stage}_cohort_sha256'],
        })
        if contract == _candidate:
            expected = _candidate
            break
    if expected is None:
        raise ValueError('V2 shared-pool source, split, evaluation, or PH2 pins mismatch')"""


def to_literal(value, indent: int = 4) -> str:
    text = json.dumps(value, indent=indent, sort_keys=True)
    return text.replace(": true", ": True").replace(": false", ": False").replace(": null", ": None")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="final v3 selection manifest")
    parser.add_argument("--protocol", type=Path, action="append", required=True)
    args = parser.parse_args()

    contract = json.loads(args.manifest.read_text(encoding="utf-8"))["pool_contract"]
    base = {k: v for k, v in contract.items() if k not in STAGE_KEYS}
    for stage in ("selection", "refit"):
        for suffix in ("count", "dataset_counts", "cohort_sha256", "source_manifest_sha256"):
            if f"{stage}_{suffix}" not in base:
                raise SystemExit(f"contract missing {stage}_{suffix}")

    block = (
        "\n\n# --- v3: source-cohort-disjoint pool contract (additive; v2 above unchanged) ---\n"
        f"# Derived from {args.manifest}\n"
        "# Generated, not hand-written: the comparison is full dict equality.\n"
        "V3_SHARED_POOL_CONTRACT = " + to_literal(base) + "\n"
    )

    for path in args.protocol:
        text = path.read_text(encoding="utf-8")
        if "V3_SHARED_POOL_CONTRACT" in text:
            print(f"  {path}: already patched, skipping")
            continue
        if text.count(OLD_BODY) != 1:
            raise SystemExit(f"{path}: validator body not found exactly once")
        before = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Constant first so the widened validator can reference it.
        marker = "def validate_v2_shared_pool_contract("
        head, sep, tail = text.partition(marker)
        text = head + block.lstrip("\n") + "\n\n" + sep + tail
        text = text.replace(OLD_BODY, NEW_BODY)
        path.write_text(text, encoding="utf-8")
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  {path}\n    sha256 {before} -> {after}")
        reread = path.read_text(encoding="utf-8")
        assert "V2_SHARED_POOL_CONTRACT = {" in reread
        assert "'selection_count': 34330" in reread, "v2 constant must stay intact"


if __name__ == "__main__":
    main()
