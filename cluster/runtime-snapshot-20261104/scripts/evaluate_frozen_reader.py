"""Evaluate one predeclared supervised reader; image encoders remain frozen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from evaluate_overlay_readout import CLASSES, evaluate_encoder, load_labels
from frozen_reader_heads import concat_features, evaluate
from probe_preprocessing_readout import align_features

READERS = ("small-mlp-reader", "svm-reader", "two-encoder-concat-reader")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_feature_rows(path, rows):
    with np.load(path, allow_pickle=False) as data:
        if not {"features", "image_ids", "groups"} <= set(data.files):
            raise ValueError("Missing feature group identities")
        if len(data["groups"]) != len(data["image_ids"]):
            raise ValueError("Feature group identity vector is incomplete")
        ids = [row["image_id"] for row in rows]
        features = align_features(data["features"], data["image_ids"], ids)
        lookup = dict(zip(data["image_ids"].tolist(), data["groups"].tolist()))
        if any(lookup[row["image_id"]] != row["identity_component_id"] for row in rows):
            raise ValueError("Feature patient identities differ from approved manifest")
    return features


def summarize(runs, classes):
    result = {name: {"mean": float(np.mean([r["metrics"][name] for r in runs])),
                     "sd_ddof0": float(np.std([r["metrics"][name] for r in runs]))}
              for name in ("macro_f1", "balanced_accuracy", "ece")}
    for name in ("recall", "auc_ovr"):
        result[name] = {label: {"mean": float(np.mean([r["metrics"][name][label] for r in runs])),
                                "sd_ddof0": float(np.std([r["metrics"][name][label] for r in runs]))}
                        for label in classes}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--reader", choices=READERS, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen reader input changed: {path}")
    output = Path(protocol["output_directory"]) / (args.reader + ".json")
    if output.exists():
        raise FileExistsError("Preserve reader evidence")
    rows = [json.loads(line) for line in Path(protocol["manifest"]).read_text().splitlines()]
    receipt_path = Path(protocol["overlay_receipt"])
    receipt = json.loads(receipt_path.read_text())
    manifest = Path(protocol["manifest"])
    if manifest.name != "pad_adapt.jsonl" or manifest.parent != receipt_path.parent:
        raise ValueError("Only the approved PAD adaptation cohort is allowed")
    if digest(manifest) != receipt["output_sha256"][manifest.name]:
        raise ValueError("Approved overlay manifest changed")
    labels = load_labels("PAD", rows, Path(protocol["metadata"]), protocol["metadata_sha256"])
    groups = np.asarray([row["identity_component_id"] for row in rows])
    ids = np.asarray([row["image_id"] for row in rows])
    x = load_feature_rows(protocol["adapted_features"], rows)
    started = time.monotonic()
    if args.reader == "two-encoder-concat-reader":
        public = load_feature_rows(protocol["public_features"], rows)
        x = concat_features(x, ids, public, ids)
        runs = evaluate_encoder(x, labels, groups, CLASSES["PAD"])
        encoder = "Frozen public-dinov2 plus frozen adapted-simsiam"
    else:
        if args.reader == "small-mlp-reader":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        runs = evaluate(args.reader, x, labels, groups, CLASSES["PAD"])
        encoder = "Frozen adapted-simsiam baseline-current-recipe training seed42"
    baseline = json.loads(Path(protocol["baseline_readout"]).read_text())
    if baseline["image_ids"] != ids.tolist() or baseline["groups"] != groups.tolist():
        raise ValueError("Reader comparison cohort differs from baseline")
    for candidate, reference in zip(runs, baseline["runs"]):
        if candidate["seed"] != reference["seed"]:
            raise ValueError("CV seeds must be paired")
        for left, right in zip(candidate["folds"], reference["folds"]):
            if any(left[key] != right[key] for key in ("train_groups_sha256", "test_groups_sha256")):
                raise ValueError("Patient CV folds must be paired")
    result = {"reader": args.reader, "encoder": encoder, "encoder_training": "SSL/public pretraining; frozen in this experiment",
              "reader_training": "Supervised with permitted PAD adaptation diagnoses only",
              "image_ids": ids.tolist(), "groups": groups.tolist(), "classes": CLASSES["PAD"],
              "runs": runs, "summary": summarize(runs, CLASSES["PAD"]),
              "protocol_sha256": digest(args.protocol), "input_sha256": protocol["input_sha256"],
              "seconds": time.monotonic() - started,
              "gpu_allocated_hours_for_reader": (time.monotonic() - started) / 3600 if args.reader == "small-mlp-reader" else 0,
              "limits": "Exploratory transductive development. CV seed variation is not independent encoder-training replication. Report all predeclared heads; no clinical promotion."}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"reader": args.reader, "summary": result["summary"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
