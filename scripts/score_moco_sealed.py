#!/usr/bin/env python3
"""MoCo V10 release scorer. Three modes; sealed rows are read ONLY by `sealed`, once.

dev-check       recompute dev OOF metrics from stored readouts (reproduction gate, no sealed access)
fit-candidate   fit the 448 CLS4+D4 candidate heads on development rows only (no sealed access)
sealed          one-shot: score ham_holdout/pad_test with reference A (224 CLS4, shipped heads) and the
                candidate (448 CLS4+D4); requires --confirm-candidate and an unclaimed receipt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Locations come from the environment, never from a cluster default:
# DERMSSL_CLF1_ROOT = clinical reference workspace, DERMSSL_CANONICAL_DIR = canonical inputs.
ROOT = Path(os.environ.get("DERMSSL_CLF1_ROOT", ""))
VARIANTS = ROOT / "results/moco-v10-variants-20261109"
PLAN = ROOT / "config/cross-role-content-plan-v1.json"
CANONICAL = Path(os.environ.get("DERMSSL_CANONICAL_DIR", ""))
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from dermssl import run_environment  # noqa: E402
from dermssl.release import moco_head_fit, moco_inputs, moco_sealed  # noqa: E402
from dermssl.release.moco_head_fit import reader_probabilities  # noqa: E402

TASKS = ("HAM", "PAD")
CLASSES = moco_inputs.CLASSES
SEALED = moco_sealed.SEALED_ROLES
CANDIDATE_FEATURES = {"HAM": "ham_dev-cls4_d4.npz", "PAD": "pad_adapt-cls4_d4.npz"}


def dev_check(readouts):
    report = {}
    for task in TASKS:
        _, labels, _, rows, _ = moco_inputs.load_development(ROOT, task)
        for seed in (42, 43, 44):
            path = Path(readouts) / f"s{seed}-r448-cls4_d4-{task}.json"
            doc = json.loads(path.read_text())
            if doc["image_ids"] != [r["image_id"] for r in rows]:
                raise ValueError(f"Readout order differs from development manifest: {path.name}")
            for run in doc["runs"]:
                metrics = moco_sealed.class_metrics(labels, np.asarray(run["oof_probabilities"]), CLASSES[task])
                stored = run["metrics"]["macro_f1"]
                if abs(metrics["macro_f1"] - stored) > 1e-9:
                    raise ValueError(f"Reproduction failed for {path.name}: {metrics['macro_f1']} vs {stored}")
                report.setdefault(task, []).append({"seed": seed, "macro_f1": metrics["macro_f1"],
                                                    "balanced_accuracy": metrics["balanced_accuracy"]})
    summary = {task: {"macro_f1_mean": float(np.mean([r["macro_f1"] for r in report[task]])),
                      "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in report[task]]))}
               for task in TASKS}
    return {"status": "reproduced", "runs": report, "summary": summary}


def fit_candidate(features_dir, out):
    from threadpoolctl import threadpool_limits
    heads = {}
    for task in TASKS:
        _, labels, groups, rows, provenance = moco_inputs.load_development(ROOT, task)
        npz = np.load(Path(features_dir) / CANDIDATE_FEATURES[task], allow_pickle=False)
        if npz["image_ids"].tolist() != [r["image_id"] for r in rows]:
            raise ValueError("448 D4 feature order differs from development manifest")
        with threadpool_limits(limits=2):
            artifact = moco_head_fit.fit_reader(npz["features"], labels, groups, CLASSES[task])
        artifact["provenance"] = provenance | {"feature_recipe": "cls4_d4 @448, D4 mean, BF16 extraction"}
        heads[task] = artifact
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for task, artifact in heads.items():
        moco_sealed.write_json(out / f"{task}-candidate-reader.json", artifact)
        run_environment.write_sidecar(out / f"{task}-candidate-reader.json")
    return {task: {"C": heads[task]["C"], "n_images": heads[task]["n_images"]} for task in TASKS}


def sealed_rows(role):
    plan = json.loads(PLAN.read_text())
    expected = {m["role"]: m["sha256"] for m in plan["manifests"]}
    path = CANONICAL / f"{role}.jsonl"
    if moco_inputs.sha256_file(path) != expected[role]:
        raise ValueError(f"Sealed manifest digest mismatch for {role}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def extract(rows, size, mode):
    """Same extractor as the development variants (variant_features.extract), CUDA BF16."""
    import torch
    from torchvision import transforms as T
    from PIL import Image, ImageOps
    sys.path[:0] = [str(VARIANTS / "scripts"), str(ROOT / "scripts"), str(ROOT / "scripts/arms")]
    import variant_features as vf
    import moco_v5_ops_20261107 as ops
    from moco_v3_20261029_runtime import _load_backbone
    weights = ROOT / "runs/moco-v4-quota4-s42-full-20261107/simsiam_dinov2_backbone_final.pt"
    if vf.sha(weights) != moco_inputs.ENCODER_SHA256:
        raise ValueError("Encoder checkpoint digest mismatch")
    model = _load_backbone(ops.PUBLIC)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    return vf.extract(model, rows, size, torch, T, Image, ImageOps, np)[mode]


def extract_check(features_dir, rows_limit=64):
    """Re-extract the first development rows with the sealed-path extractor and compare to stored 448 D4 features."""
    _, _, _, rows, _ = moco_inputs.load_development(ROOT, "HAM")
    rows = rows[:rows_limit]
    stored = np.load(Path(features_dir) / CANDIDATE_FEATURES["HAM"], allow_pickle=False)["features"][:rows_limit]
    fresh = extract(rows, 448, "cls4_d4")
    unit_fresh = fresh / np.linalg.norm(fresh, axis=1, keepdims=True)
    unit_stored = stored / np.linalg.norm(stored, axis=1, keepdims=True)
    return {"n": len(rows), "max_abs_diff": float(np.abs(fresh - stored).max()),
            "cosine_min": float(np.sum(unit_fresh * unit_stored, axis=1).min())}


def sealed(candidate_dir, reference_dir, confirm, out):
    if confirm != "confirmed-final-candidate":
        raise SystemExit("Refusing: the final candidate must be confirmed by the lead first")
    out = Path(out)
    moco_sealed.claim_once(out.with_suffix(".claim"))
    reference_heads = {task: shipped_reader(reference_dir, task) for task in TASKS}
    candidate_heads = {task: json.loads((Path(candidate_dir) / f"{task}-candidate-reader.json").read_text())
                       for task in TASKS}
    results = {"reference_A": {}, "candidate": {}}
    for task in TASKS:
        role = SEALED[task]
        rows = sealed_rows(role)
        labels = moco_inputs._load_labels(task, rows)
        feats_224 = extract(rows, 224, "cls4")
        feats_448 = extract(rows, 448, "cls4_d4")
        results["reference_A"][task] = moco_sealed.class_metrics(
            labels, reader_probabilities(feats_224, reference_heads[task]), CLASSES[task]) | {"role": role}
        results["candidate"][task] = moco_sealed.class_metrics(
            labels, reader_probabilities(feats_448, candidate_heads[task]), CLASSES[task]) | {"role": role}
    results["release_rule"] = moco_sealed.release_rule(results["candidate"], results["reference_A"])
    moco_sealed.write_json(out, results)
    run_environment.write_sidecar(out)
    return results["release_rule"]


def shipped_reader(reference_dir, task):
    """Shipped v10.0.0 reader (224 CLS4, C by macro-F1) from the digest-verified release runtime heads."""
    npz = np.load(Path(reference_dir) / f"{task}-reader.npz")
    meta = json.loads((Path(reference_dir) / f"{task}-reader.json").read_text())
    return {"weight": npz["weight"].tolist(), "bias": npz["bias"].tolist(), "classes": meta["classes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("dev-check")
    check.add_argument("--readouts", type=Path, default=VARIANTS / "readouts")
    fit = sub.add_parser("fit-candidate")
    fit.add_argument("--features-dir", type=Path, default=VARIANTS / "s42-r448")
    fit.add_argument("--out", type=Path, required=True)
    sub.add_parser("extract-check").add_argument("--features-dir", type=Path, default=VARIANTS / "s42-r448")
    seal = sub.add_parser("sealed")
    seal.add_argument("--candidate-dir", type=Path, required=True)
    seal.add_argument("--reference-dir", type=Path, required=True)
    seal.add_argument("--confirm-candidate", required=True)
    seal.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (ROOT.parts and CANONICAL.parts):
        parser.error("set DERMSSL_CLF1_ROOT and DERMSSL_CANONICAL_DIR")
    if args.mode == "dev-check":
        print(json.dumps(dev_check(args.readouts), indent=2, allow_nan=False))
    elif args.mode == "extract-check":
        print(json.dumps(extract_check(args.features_dir), indent=2))
    elif args.mode == "fit-candidate":
        print(json.dumps(fit_candidate(args.features_dir, args.out), indent=2))
    else:
        print(json.dumps(sealed(args.candidate_dir, args.reference_dir, args.confirm_candidate, args.out), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
