#!/usr/bin/env python3
"""Screening policy for v10.5.0 from screening_v10_4.py output (no new numbers).

HAM7: offset and development metrics from the report; PAD6: unchanged head, block copied from the
v10.4.0 policy. Pins the v10.5.0 model, manifest and HAM head digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PUBLIC = ("target_recall", "target_to_nevus", "nevus_recall", "target_precision", "target_alert_rate", "macro_f1")
SCOPE = ("HAM7: offset selected on the MILK10k dev split (never used to fit the v10.5.0 reader) with single-view "
         "FP32 features, as served. PAD6: unchanged v10.1.0 head and offset (its development metrics come from the "
         "v10.1.0 D4 features). Scores are not clinical probabilities.")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public(metrics: dict) -> dict:
    return {key: round(float(metrics[key]), 4) for key in PUBLIC}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "base-policy", "manifest", "rule", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    base = json.loads(args.base_policy.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest["release"] != "v10.5.0":
        raise SystemExit("manifest is not v10.5.0")
    ham = {"task": "HAM", "target": "mel", "log_offset": float(report["selected_delta"]),
           "head_sha256": manifest["files"]["heads/HAM-reader.json"],
           "development": {"at_zero": public(report["nested"]["at_zero"]), "nested": public(report["nested"]["metrics"])}}
    policy = {"schema": base["schema"], "rule_id": base["rule_id"], "release": "v10.5.0",
              "model_sha256": manifest["files"]["model.onnx"], "manifest_sha256": sha(args.manifest), "scope": SCOPE,
              "heads": {"MOCO_HAM7": ham, "MOCO_PAD6": base["heads"]["MOCO_PAD6"]},
              "evidence": {"report_sha256": sha(args.report), "rule_sha256": sha(args.rule)}}
    args.out.write_text(json.dumps(policy, indent=1) + "\n")
    print(json.dumps({"policy_sha256": sha(args.out), "log_offset": ham["log_offset"]}))


if __name__ == "__main__":
    main()
