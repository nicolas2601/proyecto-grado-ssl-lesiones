"""Grouped development-only fitting for the frozen MoCo feature reader."""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np

CV_SEEDS = (20261025, 20261026, 20261027)
C_GRID = (.3, 1., 3., 10., 30., 100.)
N_SPLITS = 3


def normalized_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Features must be a nonempty finite matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Features must have finite nonzero row norms")
    return x / norms


def _validate_inputs(x, labels, groups, classes, c_grid):
    y, g = np.asarray(labels).astype(str), np.asarray(groups).astype(str)
    if y.ndim != 1 or g.ndim != 1 or len(y) != len(x) or len(g) != len(x):
        raise ValueError("Features, labels and groups must be aligned")
    if len(classes) < 3 or len(set(classes)) != len(classes) or set(y) != set(classes):
        raise ValueError("Training labels must match the declared native classes")
    if tuple(sorted(classes)) != tuple(classes):
        raise ValueError("Declared classes must preserve the registered sorted order")
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("C grid must contain finite positive values")
    if any(len(set(g[y == c])) < N_SPLITS for c in classes):
        raise ValueError("Insufficient component support for grouped selection")
    return y, g


def _estimator(x, y, c, max_iter):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter,
                                  solver="lbfgs", random_state=42).fit(x, y)


def _group_hash(groups):
    return hashlib.sha256(json.dumps(sorted(set(groups))).encode()).hexdigest()


def _selection(x, y, groups, classes, c_grid, cv_seeds, max_iter):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    scores = {c: [] for c in c_grid}
    traces = []
    for seed in cv_seeds:
        splitter = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train, valid) in enumerate(splitter.split(x, y, groups)):
            if set(groups[train]) & set(groups[valid]):
                raise ValueError("Identity component leaks across grouped selection")
            if any(set(y[index]) != set(classes) for index in (train, valid)):
                raise ValueError("Every grouped fold must contain all native classes")
            for c in c_grid:
                model = _estimator(x[train], y[train], c, max_iter)
                scores[c].append(float(f1_score(y[valid], model.predict(x[valid]),
                                                labels=classes, average="macro", zero_division=0)))
            traces.append({"seed": int(seed), "fold": fold, "train_groups_sha256": _group_hash(groups[train]),
                           "valid_groups_sha256": _group_hash(groups[valid])})
    means = {c: float(np.mean(values)) for c, values in scores.items()}
    selected = max(c_grid, key=lambda c: (means[c], -c))
    return selected, {"C_grid": list(c_grid), "CV_seeds": list(cv_seeds), "splits": N_SPLITS,
                      "scores": {str(c): scores[c] for c in c_grid}, "mean_macro_f1": means,
                      "folds": traces, "tie_rule": "smaller C", "metric": "macro_f1"}


def fit_reader(features, labels, groups, classes, *, c_grid=C_GRID, cv_seeds=CV_SEEDS, max_iter=2000):
    """Select C within grouped development data, then fit all allowed rows; not a test score."""
    x = normalized_features(features)
    y, group = _validate_inputs(x, labels, groups, classes, c_grid)
    selected, selection = _selection(x, y, group, classes, tuple(sorted(set(c_grid))), cv_seeds, max_iter)
    model = _estimator(x, y, selected, max_iter)
    if tuple(model.classes_) != tuple(classes):
        raise ValueError("Estimator class order differs from registered ontology")
    return {"classes": list(classes), "feature_dim": int(x.shape[1]),
            "weight": model.coef_.tolist(), "bias": model.intercept_.tolist(), "C": float(selected),
            "class_weight": "balanced", "solver": "lbfgs", "max_iter": max_iter,
            "n_iter": model.n_iter_.tolist(), "n_images": len(x), "n_groups": len(set(group)),
            "selection": selection, "normalization": "per-image L2, no StandardScaler",
            "scope": "Final development fit; do not interpret fitting scores as held-out evaluation"}


def reader_probabilities(features, artifact):
    x = normalized_features(features)
    weight, bias = np.asarray(artifact["weight"]), np.asarray(artifact["bias"])
    if weight.shape != (len(artifact["classes"]), x.shape[1]) or bias.shape != (len(weight),):
        raise ValueError("Reader dimensions do not match features/classes")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Reader parameters must be finite")
    logits = x @ weight.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
