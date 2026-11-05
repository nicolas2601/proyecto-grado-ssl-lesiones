"""Fixed-224 paired frozen-encoder controls; no further feature search."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from probe_preprocessing_readout import (
    CLASSES, SEEDS, C_GRID, OUTER_FOLDS, INNER_FOLDS, ROOT,
    _manifest, load_authorized_labels, align_features, grouped_folds, metrics,
)

TAUS = (0., .5, 1.)
METHODS = ("balanced_lr", "unweighted_lr_logit", "ncm_cl2")


def adjust_probabilities(probabilities, counts, tau):
    probabilities, counts = np.asarray(probabilities), np.asarray(counts)
    if (probabilities.ndim != 2 or counts.shape != (probabilities.shape[1],)
            or np.any(counts <= 0) or not np.isfinite(counts).all()
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Expected normalized finite probabilities and positive training class counts")
    logits = np.log(np.maximum(probabilities, np.finfo(float).tiny)) - tau * np.log(counts / counts.sum())
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def unit(features):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), np.finfo(float).tiny)


def ncm_probabilities(train, labels, query, classes):
    center = train.mean(axis=0, keepdims=True)
    train, query = unit(train - center), unit(query - center)
    prototypes = np.stack([train[labels == label].mean(axis=0) for label in classes])
    scores = -np.square(query[:, None, :] - prototypes[None, :, :]).sum(axis=2)
    values = np.exp(scores - scores.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fit_lr(x, y, c, balanced):
    model = LogisticRegression(C=c, class_weight="balanced" if balanced else None,
                               max_iter=2000).fit(unit(x), y)
    if tuple(model.classes_) != CLASSES:
        raise ValueError("Training fold must contain all fixed classes")
    return model


def counts(y):
    return np.array([np.count_nonzero(y == label) for label in CLASSES])


def inner_selection(x, y, groups, seed):
    scores = {method: {} for method in METHODS[:2]}
    for train, valid in grouped_folds(y, groups, INNER_FOLDS, seed):
        for c in C_GRID:
            for method in scores:
                balanced = method == "balanced_lr"
                p = fit_lr(x[train], y[train], c, balanced).predict_proba(unit(x[valid]))
                for tau in ((0.,) if balanced else TAUS):
                    adjusted = adjust_probabilities(p, counts(y[train]), tau)
                    score = f1_score(y[valid], np.array(CLASSES)[adjusted.argmax(1)],
                                     labels=CLASSES, average="macro", zero_division=0)
                    scores[method].setdefault((c, tau), []).append(float(score))
    return {method: max(grid, key=lambda key: np.mean(grid[key])) for method, grid in scores.items()}


def fold_digest(indices, groups):
    return hashlib.sha256(json.dumps(sorted(set(groups[indices]))).encode()).hexdigest()


def evaluate_encoder(x, y, groups):
    runs = {method: [] for method in METHODS}
    for seed in SEEDS:
        oof = {method: np.zeros((len(y), len(CLASSES))) for method in METHODS}
        traces = {method: [] for method in METHODS}
        for fold, (train, test) in enumerate(grouped_folds(y, groups, OUTER_FOLDS, seed)):
            selected = inner_selection(x[train], y[train], groups[train], seed + fold)
            for method in METHODS:
                c, tau = selected.get(method, (None, 0.))
                if method == "ncm_cl2":
                    p = ncm_probabilities(x[train], y[train], x[test], CLASSES)
                else:
                    p = fit_lr(x[train], y[train], c, method == "balanced_lr").predict_proba(unit(x[test]))
                    p = adjust_probabilities(p, counts(y[train]), tau)
                oof[method][test] = p
                traces[method].append({"fold": fold, "C": c, "tau": tau,
                                      "train_groups_sha256": fold_digest(train, groups),
                                      "test_groups_sha256": fold_digest(test, groups)})
        for method in METHODS:
            runs[method].append({"seed": seed, "metrics": metrics(y, oof[method], CLASSES),
                                 "folds": traces[method], "oof_probabilities": oof[method].tolist()})
    return runs


def summarize(runs):
    result = {}
    for name in ("macro_f1", "balanced_accuracy", "ece"):
        values = [run["metrics"][name] for run in runs]
        result[name] = {"mean": float(np.mean(values)), "sd": float(np.std(values))}
    for name in ("recall", "auc_ovr"):
        result[name] = {}
        for label in CLASSES:
            values = [run["metrics"][name][label] for run in runs]
            result[name][label] = {"mean": float(np.mean(values)), "sd": float(np.std(values))}
    return result


def main(root):
    destination = root / "results/final-reader-controls-v1.json"
    if destination.exists():
        raise FileExistsError("Preserve existing reader-control evidence")
    started = time.monotonic()
    protocol, rows = _manifest(root)
    ids = np.array([row["image_id"] for row in rows])
    groups = np.array([row["identity_component_id"] for row in rows])
    metadata = Path(json.loads((root / "config/paths.json").read_text())["metadata_csv"])
    y = load_authorized_labels(root, rows, protocol, metadata)
    arms, hashes = {}, {}
    for encoder in ("S0", "refit"):
        source = root / f"results/preprocessing-variants-v1/{encoder}_squash224.npz"
        hashes[encoder] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source, allow_pickle=False) as data:
            x = align_features(data["features"], data["image_ids"], ids)
        for method, runs in evaluate_encoder(x, y, groups).items():
            arms[f"{encoder}_{method}"] = {"summary": summarize(runs), "runs": runs}
        print(f"completed {encoder}, seconds={time.monotonic() - started:.1f}", flush=True)
    differences = {}
    for method in METHODS:
        values = [b["metrics"]["macro_f1"] - a["metrics"]["macro_f1"]
                  for a, b in zip(arms[f"S0_{method}"]["runs"], arms[f"refit_{method}"]["runs"])]
        differences[method] = {"per_seed": values, "mean": float(np.mean(values)), "sd": float(np.std(values))}
    report = {"scope": "Exploratory paired frozen-encoder readout development on PAD target_adapt, not clinical validation",
              "settings": {"feature": "fixed squash224 CLS; no further preprocessing search",
                           "seeds": SEEDS, "outer": OUTER_FOLDS, "inner": INNER_FOLDS,
                           "C_grid": C_GRID, "tau_grid": TAUS,
                           "logit_rule": "Unweighted LR only: log p minus tau*log training image prior; never balanced plus adjustment",
                           "NCM": "Subtract training-only raw embedding mean, L2 each image, class means, negative squared Euclidean logits, fixed temperature1; no validation centering",
                           "selection": "Each LR selects C/tau internally; no method selected on outer scores",
                           "sd": "ddof0, descriptive repeated-partition sensitivity, not CI"},
              "manifest_sha256": protocol["cohort"]["manifest_sha256"], "features_sha256": hashes,
              "image_ids": ids.tolist(), "groups": groups.tolist(), "arms": arms,
              "paired_refit_minus_S0_f1": differences, "duration_seconds": time.monotonic() - started,
              "limitations": [protocol["representation_exposure"], "NCM softmax scores are not probability-calibrated",
                              "Hyperparameters chosen separately inside each matched encoder/reader pipeline; this is not a reader-weight swap test",
                              "Six arms on the same cohort do not establish an encoder ceiling or independent replication"]}
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(destination), "paired_refit_minus_S0_f1": differences,
                      "duration_seconds": report["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    main(parser.parse_args().root)
