"""Jointly selected linear readout for explicitly exploratory PAD development."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from collections.abc import Sequence

import numpy as np

FROZEN_SRC = Path(__file__).resolve().parents[2] / "ssl-phone-20261024/code/src"
if not (FROZEN_SRC / "dermssl").is_dir():
    FROZEN_SRC = Path("/home/jovyan/work/clf1_ssl_phone_20261024/code/src")
if (FROZEN_SRC / "dermssl").is_dir() and str(FROZEN_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_SRC))
from dermssl.metrics import balanced_accuracy as _balanced_accuracy, confusion_matrix as _confusion_matrix
from dermssl.metrics import ece as _ece, macro_f1 as _macro_f1, recall_per_class as _recall_per_class
from dermssl.metrics import binary_auc as _binary_auc

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
VARIANTS = ("squash224", "squash336", "squash336_tta", "squash448", "multilayer")
MODELS = ("S0", "refit")
SEEDS = (20261025, 20261026, 20261027)
OUTER_FOLDS, INNER_FOLDS = 5, 3
C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
ECE_BINS = 15
MAX_ITER = 2000


def _strings(values: Sequence[str], name: str, *, unique: bool = True) -> np.ndarray:
    result = np.asarray(values).astype(str)
    if result.ndim != 1 or not len(result) or any(not value.strip() for value in result):
        raise ValueError(f"{name} must be a non-empty vector without blank values")
    if unique and len(set(result.tolist())) != len(result):
        raise ValueError(f"duplicate {name} identities")
    return result


def align_features(features: np.ndarray, image_ids: Sequence[str], expected_ids: Sequence[str]) -> np.ndarray:
    """Align features to manifest order; reject any identity set mismatch."""
    observed = _strings(image_ids, "feature image")
    expected = _strings(expected_ids, "manifest image")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(observed) or not np.isfinite(values).all():
        raise ValueError("features must be a finite matrix aligned with feature image identities")
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("feature and manifest image identity sets differ")
    lookup = {identity: index for index, identity in enumerate(observed)}
    aligned = values[[lookup[identity] for identity in expected]]
    if np.any(np.linalg.norm(aligned, axis=1) == 0):
        raise ValueError("zero-norm feature identity")
    return aligned


def validate_fold_support(labels: Sequence[str], classes: Sequence[str], n_splits: int) -> None:
    y = _strings(labels, "label", unique=False)
    if set(y) != set(classes):
        raise ValueError("all fixed classes must be represented in development labels")
    minimum = min(int(np.sum(y == label)) for label in classes)
    if minimum < n_splits:
        raise ValueError(f"class support {minimum} is insufficient for {n_splits} folds")


def grouped_folds(labels: Sequence[str], groups: Sequence[str], n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    y = _strings(labels, "label", unique=False)
    group = _strings(groups, "identity component", unique=False)
    if len(y) != len(group):
        raise ValueError("labels and identity components must be aligned")
    validate_fold_support(y, tuple(dict.fromkeys(y)), n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, group))
    for train, test in folds:
        if set(group[train]) & set(group[test]):
            raise ValueError("identity component leaks across grouped folds")
        for subset in (train, test):
            missing = set(np.unique(y)) - set(np.unique(y[subset]))
            if missing:
                raise ValueError(f"fold class support missing {sorted(missing)}")
    if len(np.unique(np.concatenate([test for _, test in folds]))) != len(y):
        raise ValueError("folds must test each image exactly once")
    return folds


def metrics(truth: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]) -> dict:
    labels = np.asarray(truth).astype(str)
    proba = np.asarray(probabilities, dtype=np.float64)
    class_names = tuple(classes)
    if len(set(class_names)) != len(class_names) or set(labels) - set(class_names):
        raise ValueError("truth labels must use unique fixed class order")
    if proba.shape != (len(labels), len(class_names)) or not np.isfinite(proba).all():
        raise ValueError("probability matrix must be finite with fixed class columns")
    if np.any(proba < 0) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities must be non-negative and sum to one")
    y = np.asarray([class_names.index(item) for item in labels])
    pred = proba.argmax(axis=1)
    aucs = {}
    for index, label in enumerate(class_names):
        binary = y == index
        auc = _binary_auc(binary, proba[:, index]) if binary.any() and (~binary).any() else float("nan")
        aucs[label] = float(auc) if np.isfinite(auc) else None
    recalls = _recall_per_class(y, pred, len(class_names))
    return {
        "classes": list(class_names),
        "auc_ovr": aucs,
        "macro_f1": _macro_f1(y, pred, len(class_names)),
        "balanced_accuracy": _balanced_accuracy(y, pred, len(class_names)),
        "recall": {label: float(recalls[index]) for index, label in enumerate(class_names)},
        "ece": _ece(y, proba, ECE_BINS),
        "ece_bins": ECE_BINS,
        "confusion_matrix": _confusion_matrix(y, pred, len(class_names)).tolist(),
    }


def _digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest(root: Path):
    protocol = json.loads((root / "config/clinical-development-protocol-v1.json").read_text())
    manifest = root / "inputs/pad-adapt-image-only.jsonl"
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != protocol["cohort"]["manifest_sha256"]:
        raise ValueError("Canonical PAD adaptation manifest hash mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if (len(rows) != protocol["cohort"]["images"]
            or len({row["image_id"] for row in rows}) != len(rows)
            or len({row["identity_component_id"] for row in rows}) != protocol["cohort"]["patients"]):
        raise ValueError("PAD manifest image or identity-component support mismatch")
    audit_path = root / "evidence/pad-adapt-content-audit.json"
    if not audit_path.is_file():
        raise ValueError("PAD content-identity audit missing")
    audit = json.loads(audit_path.read_text())
    if any(group["cross_patient_groups"] for group in audit["aliases"].values()):
        raise ValueError("Cross-patient PAD content aliases prevent grouped scoring")
    return protocol, rows


def load_authorized_labels(root: Path, rows: list[dict], protocol: dict, metadata_path: Path) -> np.ndarray:
    if not metadata_path.is_file() or hashlib.sha256(metadata_path.read_bytes()).hexdigest() != protocol["cohort"]["metadata_sha256"]:
        raise ValueError("Authorized PAD diagnostic metadata hash mismatch")
    wanted = {row["image_id"] for row in rows}
    diagnoses = {}
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"img_id", "diagnostic"} <= set(reader.fieldnames or ()):
            raise ValueError("PAD metadata requires img_id and diagnostic")
        for row in reader:
            image_id = Path(row["img_id"]).stem
            if image_id not in wanted:
                continue
            if image_id in diagnoses:
                raise ValueError("duplicate PAD diagnostic identity")
            diagnoses[image_id] = row["diagnostic"].strip()
    if set(diagnoses) != wanted:
        raise ValueError("missing PAD diagnostic identity")
    labels = np.asarray([diagnoses[row["image_id"]] for row in rows])
    if set(labels) != set(CLASSES):
        raise ValueError("PAD diagnostic labels do not cover the fixed native class ontology")
    return labels


def _read_variant(directory: Path, variant: str, expected_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    blocks, hashes = [], []
    for model in MODELS:
        path = directory / f"{model}_{variant}.npz"
        if not path.is_file():
            raise ValueError(f"missing complete preprocessing feature file {path.name}")
        with np.load(path, allow_pickle=False) as data:
            if not {"features", "image_ids"} <= set(data.files):
                raise ValueError(f"feature archive missing identities or features: {path.name}")
            features = align_features(data["features"], data["image_ids"], expected_ids)
            if features.shape[1] != 768:
                raise ValueError(f"unexpected feature dimensions in {path.name}")
            blocks.append(features)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    normalized = [block / np.linalg.norm(block, axis=1, keepdims=True) for block in blocks]
    return np.concatenate(normalized, axis=1), hashes


def _read_multilayer(directory: Path, expected_ids: np.ndarray) -> tuple[dict[str, np.ndarray], list[str]]:
    cls_layers, patch_layers, hashes = [], [], []
    for model in MODELS:
        path = directory / f"{model}.npz"
        if not path.is_file():
            raise ValueError(f"missing complete multilayer feature file {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            if not {"cls", "mean_patch", "image_ids"} <= set(archive.files):
                raise ValueError(f"multilayer archive missing identities or feature blocks: {path.name}")
            identities = _strings(archive["image_ids"], "multilayer image")
            if len(identities) != len(expected_ids) or set(identities) != set(expected_ids):
                raise ValueError(f"multilayer identity set mismatch in {path.name}")
            cls = align_features(archive["cls"], identities, expected_ids)
            patch = align_features(archive["mean_patch"], identities, expected_ids)
            if cls.shape[1] != 3072 or patch.shape[1] != 768:
                raise ValueError(f"unexpected multilayer feature dimensions in {path.name}")
            cls_layers.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
            patch_layers.append(patch / np.linalg.norm(patch, axis=1, keepdims=True))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return {"multilayer_cls": np.concatenate(cls_layers, axis=1),
            "multilayer_patch": np.concatenate(patch_layers, axis=1),
            "multilayer_concat": np.concatenate(cls_layers + patch_layers, axis=1)}, hashes


def _fit(x: np.ndarray, y: np.ndarray, c: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER, solver="lbfgs").fit(x, y)


def _select_pair(x_by_variant: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
                 seed: int, variants: Sequence[str] = VARIANTS):
    from sklearn.metrics import confusion_matrix

    folds = grouped_folds(y, groups, INNER_FOLDS, seed)
    scores = {(variant, c): [] for variant in variants for c in C_GRID}
    for train, valid in folds:
        for variant in variants:
            for c in C_GRID:
                predicted = _fit(x_by_variant[variant][train], y[train], c).predict(x_by_variant[variant][valid])
                cm = confusion_matrix(y[valid], predicted, labels=CLASSES)
                support = cm.sum(axis=1)
                f1 = np.divide(2 * np.diag(cm), support + cm.sum(axis=0), out=np.zeros(len(CLASSES)),
                               where=(support + cm.sum(axis=0)) != 0)
                scores[(variant, c)].append(float(f1.mean()))
    selected = max(scores, key=lambda candidate: np.mean(scores[candidate]))
    return selected, max(float(np.mean(values)) for values in scores.values())


def run_probe(root: Path = ROOT, output_path: Path | None = None) -> dict:
    destination = output_path or root / "results/preprocessing-readout-v1.json"
    if destination.exists():
        raise FileExistsError("Preserve existing readout evidence; choose a new output path")
    protocol, rows = _manifest(root)
    metadata_path = Path(json.loads((root / "config/paths.json").read_text())["metadata_csv"])
    if not metadata_path.is_file():
        raise ValueError(f"authorized PAD metadata is unavailable: {metadata_path}")
    labels = load_authorized_labels(root, rows, protocol, metadata_path)
    ids = np.asarray([row["image_id"] for row in rows])
    groups = np.asarray([row["identity_component_id"] for row in rows])
    validate_fold_support(labels, CLASSES, OUTER_FOLDS)
    directory = root / "results/preprocessing-variants-v1"
    x_by_variant, feature_hashes, available_variants = {}, {}, []
    for variant in VARIANTS[:-1]:
        x_by_variant[variant], feature_hashes[variant] = _read_variant(directory, variant, ids)
        available_variants.append(variant)
    try:
        multilayer, multilayer_hashes = _read_multilayer(root / "results/multilayer-features-v1", ids)
        for name, matrix in multilayer.items():
            x_by_variant[name] = matrix
            feature_hashes[name] = multilayer_hashes
            available_variants.append(name)
    except ValueError as error:
        if "missing complete multilayer" not in str(error):
            raise
    fits_per_inner_cv = len(available_variants) * len(C_GRID) * INNER_FOLDS
    expected_fits = len(SEEDS) * OUTER_FOLDS * (fits_per_inner_cv + 1)
    print(f"joint_candidates={len(available_variants)}x{len(C_GRID)}x{INNER_FOLDS}; "
          f"fits_per_outer_fold={fits_per_inner_cv + 1}; expected_total_fits={expected_fits}", flush=True)
    runs = []
    for seed in SEEDS:
        seed_started = time.monotonic()
        outer_folds = grouped_folds(labels, groups, OUTER_FOLDS, seed)
        oof = np.zeros((len(ids), len(CLASSES)), dtype=np.float64)
        seen = np.zeros(len(ids), dtype=bool)
        fold_records = []
        for fold_number, (train, test) in enumerate(outer_folds):
            fold_started = time.monotonic()
            selection_started = time.monotonic()
            selected, inner_f1 = _select_pair(
                {name: x[train] for name, x in x_by_variant.items()}, labels[train], groups[train],
                seed + fold_number, available_variants)
            selected_variant, selected_c = selected
            selection_seconds = time.monotonic() - selection_started
            model = _fit(x_by_variant[selected_variant][train], labels[train], selected_c)
            probabilities = model.predict_proba(x_by_variant[selected_variant][test])
            aligned_probabilities = np.zeros((len(test), len(CLASSES)), dtype=np.float64)
            for column, label in enumerate(model.classes_):
                aligned_probabilities[:, CLASSES.index(label)] = probabilities[:, column]
            oof[test] = aligned_probabilities
            seen[test] = True
            fold_records.append({
                "fold": fold_number,
                "train_id_sha256": _digest_json(sorted(ids[train].tolist())),
                "test_id_sha256": _digest_json(sorted(ids[test].tolist())),
                "train_group_sha256": _digest_json(sorted(set(groups[train].tolist()))),
                "test_group_sha256": _digest_json(sorted(set(groups[test].tolist()))),
                "selected_variant": selected_variant,
                "selected_C": selected_c,
                "inner_macro_f1": inner_f1,
                "selection_seconds": selection_seconds,
                "fold_seconds": time.monotonic() - fold_started,
                "train_images": len(train),
                "test_images": len(test),
                "train_groups": len(np.unique(groups[train])),
                "test_groups": len(np.unique(groups[test])),
            })
            print(f"seed={seed} fold={fold_number + 1}/{OUTER_FOLDS} selected={selected_variant},C={selected_c} "
                  f"inner_macro_f1={inner_f1:.4f} selection_seconds={selection_seconds:.2f} "
                  f"fold_seconds={time.monotonic() - fold_started:.2f}", flush=True)
        if not seen.all() or (oof.sum(axis=1) <= 0).any():
            raise ValueError("outer folds must yield exactly one class-aligned OOF prediction per image")
        runs.append({"seed": seed, "elapsed_seconds": time.monotonic() - seed_started,
                     "metrics": metrics(labels, oof, CLASSES), "oof_probabilities": oof.tolist(), "folds": fold_records})
        print(f"seed={seed} elapsed_seconds={time.monotonic() - seed_started:.2f}", flush=True)
    metric_names = ("macro_f1", "balanced_accuracy", "ece")
    summaries = {}
    for key in metric_names:
        values = [record["metrics"][key] for record in runs]
        summaries[key] = {"mean": float(np.mean(values)), "sd_ddof_0": float(np.std(values, ddof=0))}
    summaries["auc_ovr"] = {
        label: {
            "mean": float(np.mean([run["metrics"]["auc_ovr"][label] for run in runs])),
            "sd_ddof_0": float(np.std([run["metrics"]["auc_ovr"][label] for run in runs], ddof=0)),
        } for label in CLASSES
    }
    summaries["recall"] = {
        label: {
            "mean": float(np.mean([run["metrics"]["recall"][label] for run in runs])),
            "sd_ddof_0": float(np.std([run["metrics"]["recall"][label] for run in runs], ddof=0)),
        } for label in CLASSES
    }
    result = {
        "scope": "Exploratory transductive PAD target_adapt development only; not independent, unseen-patient encoder, or clinical validation.",
        "representation_exposure": protocol["representation_exposure"],
        "forbidden_outcomes_used": [],
        "selection": "Joint preprocessing variant and C selected by inner grouped CV only; outer folds estimate this selection procedure, never rank or select variants.",
        "feature_views": {"squash_variant": "Per-image unit-L2 S0 and refit CLS features concatenated; no separate S0-only or refit-only probe was run.",
                          "multilayer": "S0/refit CLS features from DINOv2 layers 9–12 concatenated; separate mean-patch features from the final (12th) layer; blockwise unit-L2 normalization."},
        "preprocessing_variants_are_exploratory_not_winners": list(available_variants),
        "multilayer_included": any(name.startswith("multilayer_") for name in available_variants),
        "classes": list(CLASSES),
        "settings": {"readout": "balanced multinomial logistic regression", "C_grid": list(C_GRID),
                     "solver": "lbfgs", "max_iter": MAX_ITER, "seeds": list(SEEDS),
                     "outer_splits": OUTER_FOLDS, "inner_splits": INNER_FOLDS,
                     "inner_selection_metric": "macro-F1", "ece_bins": ECE_BINS,
                     "summary_sd_ddof": 0,
                     "sd_interpretation": "Descriptive sensitivity among three seeded partitioning runs, not an inferential confidence interval.",
                     "normalization": "Per-image unit-L2 on each encoder embedding before concatenation."},
        "cohort": {"images": len(ids), "identity_components": len(np.unique(groups)),
                   "manifest_sha256": protocol["cohort"]["manifest_sha256"],
                   "image_ids_sha256": _digest_json(ids.tolist()), "identity_component_ids_sha256": _digest_json(groups.tolist()),
                   "label_support_images": {label: int(np.sum(labels == label)) for label in CLASSES},
                   "label_support_identity_components": {label: int(len(set(groups[labels == label]))) for label in CLASSES}},
        "feature_archives_sha256": feature_hashes,
        "summaries": summaries,
        "runs": runs,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_probe(args.root, args.output)
    print(json.dumps({"output": str(args.output or args.root / "results/preprocessing-readout-v1.json"),
                      "scope": result["scope"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "2")
    main()
