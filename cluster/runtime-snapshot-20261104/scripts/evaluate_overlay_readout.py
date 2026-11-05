"""Fixed balanced-LR nested grouped CV on approved derived development cohorts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from probe_preprocessing_readout import (SEEDS, C_GRID, OUTER_FOLDS, INNER_FOLDS,
                                         align_features, grouped_folds, metrics)

CLASSES = {"PAD": ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK"),
           "HAM": ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_labels(kind, rows, metadata, expected_sha):
    if file_sha256(metadata) != expected_sha:
        raise ValueError("Diagnostic metadata digest mismatch")
    wanted = {row["image_id"] for row in rows}
    if len(wanted) != len(rows):
        raise ValueError("Duplicate allowed image identities")
    found = {}
    with Path(metadata).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        id_key = "img_id" if kind == "PAD" else "image_id"
        label_key = "diagnostic" if kind == "PAD" else ("label" if "label" in reader.fieldnames else "dx")
        if not {id_key, label_key} <= set(reader.fieldnames or []):
            raise ValueError("Missing diagnostic identity fields")
        for row in reader:
            identity = Path(row[id_key]).stem if kind == "PAD" else row[id_key]
            if identity not in wanted:
                continue
            if identity in found:
                raise ValueError("Duplicate allowed diagnostic identity")
            found[identity] = row[label_key].strip()
    if set(found) != wanted or set(found.values()) - set(CLASSES[kind]):
        raise ValueError("Missing or unknown allowed diagnoses")
    return np.asarray([found[row["image_id"]] for row in rows])


def validate_support(labels, groups, classes):
    if set(labels) != set(classes):
        raise ValueError("All native classes need support")
    if any(len(set(groups[labels == label])) < OUTER_FOLDS for label in classes):
        raise ValueError("Insufficient grouped class support for fixed folds")
    for seed in SEEDS:
        for number, (train, _) in enumerate(grouped_folds(labels, groups, OUTER_FOLDS, seed)):
            grouped_folds(labels[train], groups[train], INNER_FOLDS, seed + number)


def group_sha(indices, groups):
    return hashlib.sha256(json.dumps(sorted(set(groups[indices]))).encode()).hexdigest()


def evaluate_encoder(features, labels, groups, classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    validate_support(labels, groups, classes)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    classes = tuple(classes)

    def fit(train, c):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(features[train], labels[train])
        if tuple(model.classes_) != classes:
            raise ValueError("Training fold class order differs from fixed ontology")
        return model

    runs = []
    for seed in SEEDS:
        oof, seen, traces = np.zeros((len(labels), len(classes))), np.zeros(len(labels), int), []
        for fold, (train, test) in enumerate(grouped_folds(labels, groups, OUTER_FOLDS, seed)):
            scores = {c: [] for c in C_GRID}
            for inner_train, valid in grouped_folds(labels[train], groups[train], INNER_FOLDS, seed + fold):
                for c in C_GRID:
                    predicted = fit(train[inner_train], c).predict(features[train[valid]])
                    scores[c].append(f1_score(labels[train[valid]], predicted, labels=classes,
                                              average="macro", zero_division=0))
            selected = max(C_GRID, key=lambda c: np.mean(scores[c]))
            oof[test] = fit(train, selected).predict_proba(features[test])
            seen[test] += 1
            traces.append({"fold": fold, "C": selected, "tau": 0,
                           "train_groups_sha256": group_sha(train, groups),
                           "test_groups_sha256": group_sha(test, groups)})
        if not np.all(seen == 1):
            raise ValueError("OOF coverage must be exactly one prediction per allowed image")
        runs.append({"seed": seed, "metrics": metrics(labels, oof, classes),
                     "folds": traces, "oof_probabilities": oof.tolist()})
        print(json.dumps({"CV_seed": seed, "macro_f1": runs[-1]["metrics"]["macro_f1"]}), flush=True)
    return runs


def build_report(kind, rows, features, labels, provenance):
    classes = CLASSES[kind]
    groups = np.asarray([row["identity_component_id"] for row in rows])
    runs = evaluate_encoder(features, labels, groups, classes)
    report = {"kind": kind, "classes": classes, "image_ids": [r["image_id"] for r in rows],
              "groups": groups.tolist(), "provenance": provenance, "runs": runs,
              "scope": "Exploratory frozen readout development, not clinical validation; CV seeds are not independent training replicates",
              "settings": {"reader": "balancedLR", "C_grid": C_GRID, "tau": 0,
                           "outer": OUTER_FOLDS, "inner": INNER_FOLDS, "CV_seeds": SEEDS}}
    report["summary"] = {name: {"mean": float(np.mean([r["metrics"][name] for r in runs])),
                                "sd_ddof0": float(np.std([r["metrics"][name] for r in runs]))}
                         for name in ("macro_f1", "balanced_accuracy", "ece")}
    report["support"] = {label: {"images": int(np.sum(labels == label)),
                                "groups": len(set(groups[labels == label]))} for label in classes}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(CLASSES), required=True)
    for name in ("features", "manifest", "metadata", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("metadata-sha", "receipt-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing readout evidence")
    role = "pad_adapt" if args.kind == "PAD" else "ham_dev"
    if args.manifest.stem != role or args.manifest.parent != args.receipt.parent:
        raise ValueError("Only approved development roles are eligible")
    receipt = json.loads(args.receipt.read_text())
    if (file_sha256(args.receipt) != args.receipt_sha or receipt["status"] != "COMPLETE"
            or file_sha256(args.manifest) != receipt["output_sha256"][args.manifest.name]):
        raise ValueError("Overlay provenance mismatch")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    ids = [row["image_id"] for row in rows]
    with np.load(args.features, allow_pickle=False) as data:
        x = align_features(data["features"], data["image_ids"], ids)
        group_lookup = dict(zip(data["image_ids"].tolist(), data["groups"].tolist()))
        if any(group_lookup[r["image_id"]] != r["identity_component_id"] for r in rows):
            raise ValueError("Feature group identities do not match derived manifest")
    labels = load_labels(args.kind, rows, args.metadata, args.metadata_sha)
    started = time.monotonic()
    result = build_report(args.kind, rows, x, labels,
                          {"features_sha256": file_sha256(args.features),
                           "manifest_sha256": file_sha256(args.manifest),
                           "metadata_sha256": args.metadata_sha, "overlay_sha256": args.receipt_sha})
    result["seconds"] = time.monotonic() - started
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "summary": result["summary"],
                      "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
