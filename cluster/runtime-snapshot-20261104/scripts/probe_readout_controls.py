"""PAD target_adapt-only controls for a frozen S0_squash224 linear readout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
SEEDS = (20261025, 20261026, 20261027)
PERMUTATIONS = 3
FRACTIONS = (0.25, 0.5, 1.0)
GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
OUTER_FOLDS, INNER_FOLDS = 5, 3


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def permute_group_labels(labels, groups, seed):
    labels, groups = np.asarray(labels, dtype=str), np.asarray(groups, dtype=str)
    if labels.ndim != 1 or groups.shape != labels.shape or not len(labels):
        raise ValueError("labels and groups must be nonempty aligned vectors")
    unique_groups = np.unique(groups)
    group_vectors = [labels[groups == group].copy() for group in unique_groups]
    group_sizes = [len(vector) for vector in group_vectors]
    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    size_strata = {}
    for index, size in enumerate(group_sizes):
        size_strata.setdefault(size, []).append(index)
    for members in size_strata.values():
        donor_indices = np.asarray(members)[rng.permutation(len(members))]
        for recipient, donor in zip(members, donor_indices):
            permuted[groups == unique_groups[recipient]] = group_vectors[donor]
    return permuted


def learning_curve_group_indices(labels, groups, seed, fraction=1.0):
    labels, groups = np.asarray(labels, dtype=str), np.asarray(groups, dtype=str)
    if labels.ndim != 1 or groups.shape != labels.shape or not len(labels):
        raise ValueError("labels and groups must be nonempty aligned vectors")
    if not 0 < fraction <= 1:
        raise ValueError("learning-curve fraction must be in (0, 1]")
    unique_groups = np.unique(groups)
    group_labels = np.asarray(["|".join(np.unique(labels[groups == group])) for group in unique_groups])
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(group_labels):
        candidates = np.flatnonzero(group_labels == label)
        take = len(candidates) if fraction == 1.0 else max(1, int(np.ceil(len(candidates) * fraction)))
        if take > len(candidates):
            raise ValueError(f"Unsupported learning-curve fraction for class {label}")
        selected.extend(rng.permutation(candidates)[:take].tolist())
    chosen = set(unique_groups[selected])
    return np.flatnonzero(np.isin(groups, list(chosen)))


def align_features(features, observed_ids, rows):
    expected_ids = [row["image_id"] for row in rows]
    observed_ids, expected_ids = np.asarray(observed_ids, dtype=str), np.asarray(expected_ids, dtype=str)
    if (observed_ids.ndim != 1 or len(observed_ids) != len(features)
            or len(set(observed_ids.tolist())) != len(observed_ids)):
        raise ValueError("Feature cache has duplicate or misaligned identity rows")
    if set(observed_ids) != set(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Feature cache identities do not match the authorized manifest")
    lookup = {image_id: index for index, image_id in enumerate(observed_ids)}
    return np.asarray(features)[[lookup[image_id] for image_id in expected_ids]]


def read_diagnoses(rows, metadata_path):
    requested = [row["image_id"] for row in rows]
    wanted = set(requested)
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate PAD query identity")
    found = {}
    with Path(metadata_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"img_id", "diagnostic"} <= set(reader.fieldnames or ()):
            raise ValueError("PAD metadata requires img_id and diagnostic")
        for row in reader:
            image_id = Path(row["img_id"]).stem
            if image_id not in wanted:
                continue
            if image_id in found:
                raise ValueError("duplicate PAD diagnostic identity")
            diagnosis = row["diagnostic"].strip()
            if diagnosis not in CLASSES:
                raise ValueError("Unknown PAD native ontology value")
            found[image_id] = diagnosis
    if set(found) != set(requested):
        raise ValueError("missing PAD diagnostic identity")
    return np.asarray([found[image_id] for image_id in requested])


def group_support(labels, groups):
    return {label: int(len(set(groups[labels == label]))) for label in CLASSES}


def metric_summary(truth, predicted):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "recall": dict(zip(CLASSES, map(float, recall_score(truth, predicted, labels=CLASSES, average=None, zero_division=0)))),
        "image_support": {label: int(np.count_nonzero(truth == label)) for label in CLASSES},
    }


def summarize_runs(runs):
    metrics = ("balanced_accuracy", "macro_f1")
    summary = {key: {"mean": float(np.mean([run[key] for run in runs])),
                     "sd": float(np.std([run[key] for run in runs]))} for key in metrics}
    summary["recall"] = {
        label: {"mean": float(np.mean([run["recall"][label] for run in runs])),
                "sd": float(np.std([run["recall"][label] for run in runs]))}
        for label in CLASSES
    }
    return summary


def make_estimator(c):
    return LogisticRegression(C=c, class_weight="balanced", max_iter=2000)


def choose_c(features, labels, groups, seed):
    splitter = StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    scores = {c: [] for c in GRID}
    for train, validation in splitter.split(features, labels, groups):
        if set(labels[train]) != set(CLASSES) or set(labels[validation]) != set(CLASSES):
            raise ValueError("Unsupported inner fold: not all native classes are represented")
        for c in GRID:
            estimator = make_estimator(c).fit(features[train], labels[train])
            scores[c].append(f1_score(labels[validation], estimator.predict(features[validation]),
                                      labels=CLASSES, average="macro", zero_division=0))
    return max(GRID, key=lambda c: np.mean(scores[c]))


def score_nested(features, labels, groups, seed, fractions=(1.0,)):
    splitter = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=seed)
    results = {fraction: [] for fraction in fractions}
    for fold, (outer_train, outer_test) in enumerate(splitter.split(features, labels, groups)):
        if set(labels[outer_train]) != set(CLASSES) or set(labels[outer_test]) != set(CLASSES):
            raise ValueError(f"Unsupported outer fold {fold}: not all native classes are represented")
        for fraction in fractions:
            if fraction == 1.0:
                selected_train = np.arange(len(outer_train))
            else:
                selected_train = learning_curve_group_indices(labels[outer_train], groups[outer_train],
                                                              seed + fold, fraction)
            train_x, train_y = features[outer_train][selected_train], labels[outer_train][selected_train]
            train_groups = groups[outer_train][selected_train]
            if set(train_y) != set(CLASSES):
                raise ValueError(f"Unsupported learning-curve fold {fold} at fraction {fraction}")
            c = choose_c(train_x, train_y, train_groups, seed)
            estimator = make_estimator(c).fit(train_x, train_y)
            metric = metric_summary(labels[outer_test], estimator.predict(features[outer_test]))
            metric.update({"outer_fold": fold, "selected_c": float(c),
                           "train_images": int(len(train_y)),
                           "train_patient_groups": int(len(set(train_groups))),
                           "test_images": int(len(outer_test)),
                           "test_patient_groups": int(len(set(groups[outer_test])))})
            results[fraction].append(metric)
    return results


def load_data(root):
    root = Path(root).resolve()
    paths = json.loads((root / "config/paths.json").read_text())
    protocol = json.loads((root / "config/clinical-development-protocol-v1.json").read_text())
    manifest_path = root / "inputs/pad-adapt-image-only.jsonl"
    cache_path = root / "results/preprocessing-variants-v1/S0_squash224.npz"
    manifest_sha, cache_sha = digest(manifest_path), digest(cache_path)
    metadata_sha = digest(Path(paths["metadata_csv"]))
    if manifest_sha != protocol["cohort"]["manifest_sha256"] or metadata_sha != protocol["cohort"]["metadata_sha256"]:
        raise ValueError("PAD adaptation manifest or diagnosis metadata digest changed")
    rows = sorted((json.loads(line) for line in manifest_path.read_text().splitlines()), key=lambda row: row["image_id"])
    if (len(rows) != protocol["cohort"]["images"] or len({row["image_id"] for row in rows}) != len(rows)
            or len({row["identity_component_id"] for row in rows}) != protocol["cohort"]["patients"]):
        raise ValueError("PAD adaptation identities or patient counts changed")
    labels = read_diagnoses(rows, paths["metadata_csv"])
    with np.load(cache_path, allow_pickle=False) as cached:
        if not {"features", "image_ids"} <= set(cached.files):
            raise ValueError("S0_squash224 feature cache missing required arrays")
        features = align_features(cached["features"], cached["image_ids"], rows)
        cache_ids = cached["image_ids"].astype(str)
    groups = np.asarray([row["identity_component_id"] for row in rows], dtype=str)
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0):
        raise ValueError("Feature cache contains invalid zero or nonfinite vectors")
    features = features.astype(np.float64)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    trace = {"manifest_sha256": manifest_sha, "metadata_sha256": metadata_sha,
             "features_sha256": cache_sha, "protocol_sha256": digest(root / "config/clinical-development-protocol-v1.json"),
             "cache_image_ids_sha256": hashlib.sha256("\n".join(cache_ids.tolist()).encode()).hexdigest()}
    return root, rows, features, labels, groups, trace


def run_controls(features, labels, groups):
    permutation_results, learning_results = [], {fraction: [] for fraction in FRACTIONS}
    for seed in SEEDS:
        for repeat in range(PERMUTATIONS):
            shuffled = permute_group_labels(labels, groups, seed * 100 + repeat)
            scored = score_nested(features, shuffled, groups, seed)[1.0]
            permutation_results.append({"seed": seed, "permutation": repeat, "outer_folds": scored,
                                        "summary": summarize_runs(scored)})
        scored = score_nested(features, labels, groups, seed, FRACTIONS)
        for fraction, folds in scored.items():
            learning_results[fraction].append({"seed": seed, "outer_folds": folds,
                                               "summary": summarize_runs(folds)})
    return permutation_results, learning_results


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    started = time.monotonic()
    root, rows, features, labels, groups, trace = load_data(args.root)
    if set(labels) != set(CLASSES):
        raise ValueError("PAD target_adapt must contain all six native diagnoses")
    permutations, learning = run_controls(features, labels, groups)
    report = {"scope": "PAD target_adapt-only frozen S0_squash224 linear readout; development control, not clinical validation",
              "representation": "S0_squash224", "encoder_frozen": True, "encoder_training": False,
              "classes": list(CLASSES), "images": len(rows), "patient_groups": len(set(groups)),
              "support": {"images": {label: int(np.count_nonzero(labels == label)) for label in CLASSES},
                          "patient_groups": group_support(labels, groups)},
              "method": {"outer_group_folds": OUTER_FOLDS, "inner_group_folds": INNER_FOLDS,
                         "seeds": list(SEEDS), "C_grid": list(GRID), "C_selection": "inner macro-F1",
                         "learning_curve_fractions": list(FRACTIONS), "permutations_per_seed": PERMUTATIONS,
                         "permutation_unit": "patient diagnosis vectors shuffled only within equal image-count strata",
                         "permutation_inference": "sanity only; no p-value or formal randomization guarantee",
                         "learning_curve_tuning": "C selected only within each outer-training subset; outer test unchanged"},
              "permutation_sanity": permutations, "learning_curve": {str(fraction): runs for fraction, runs in learning.items()},
              "trace": trace, "duration_seconds": time.monotonic() - started}
    output = root / "results/readout-controls-v1"
    if output.exists():
        raise FileExistsError("Preserve existing control outputs; register a new version")
    output.mkdir(parents=False, exist_ok=False)
    destination = output / "readout-controls.json"
    destination.write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(destination), "trace": trace,
                      "duration_seconds": report["duration_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
