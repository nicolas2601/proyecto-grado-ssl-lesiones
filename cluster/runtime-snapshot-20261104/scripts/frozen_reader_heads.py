"""Supervised readers on frozen SSL features; no encoder or image gradient exists here."""
import time

import numpy as np

from evaluate_overlay_readout import group_sha, validate_support
from probe_preprocessing_readout import SEEDS, OUTER_FOLDS, INNER_FOLDS, align_features, grouped_folds, metrics

SVM_C = (.3, 3., 30.)
WEIGHT_DECAYS = (.01, .001)


def unit(features):
    values = np.asarray(features, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise ValueError("Expected finite nonzero frozen features")
    return values / norms


def concat_features(adapted, ids, public, public_ids):
    adapted = align_features(adapted, ids, ids)
    public = align_features(public, public_ids, ids)
    return unit(np.concatenate((unit(public), unit(adapted)), axis=1))


def fit_calibrated_svm(x, y, groups, c, seed):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    folds = grouped_folds(y, groups, INNER_FOLDS, seed)
    model = CalibratedClassifierCV(
        SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", cache_size=256),
        method="sigmoid", cv=folds, ensemble=False, n_jobs=1)
    return model.fit(x, y)


def svm_fold(x, labels, groups, train, test, seed):
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    scores = {c: [] for c in SVM_C}
    for inner_train, valid in grouped_folds(labels[train], groups[train], INNER_FOLDS, seed):
        for c in SVM_C:
            model = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", cache_size=256)
            model.fit(x[train[inner_train]], labels[train[inner_train]])
            scores[c].append(f1_score(labels[train[valid]], model.predict(x[train[valid]]),
                                     average="macro", zero_division=0))
    selected = max(SVM_C, key=lambda c: np.mean(scores[c]))
    model = fit_calibrated_svm(x[train], labels[train], groups[train], selected, seed)
    return model.predict_proba(x[test]), {"C": selected, "calibration": "Patient-group OOF sigmoid, outer-train only", "C_selection": "Uncalibrated prediction macro-F1 inside inner folds"}


def mlp_fold(x, labels, groups, train, test, seed, device):
    from sklearn.metrics import f1_score
    from small_mlp_reader import fit_head
    n_classes = len(np.unique(labels))
    scores, epochs = {}, {}
    for decay in WEIGHT_DECAYS:
        scores[decay], epochs[decay] = [], []
        for number, (inner_train, valid) in enumerate(grouped_folds(labels[train], groups[train], INNER_FOLDS, seed)):
            result = fit_head(x[train[inner_train]], labels[train[inner_train]], x[train[valid]],
                              labels[train[valid]], n_classes=n_classes, seed=seed + number,
                              weight_decay=decay, device=device)
            scores[decay].append(f1_score(labels[train[valid]], result["probabilities"].argmax(1),
                                          labels=np.arange(n_classes), average="macro", zero_division=0))
            epochs[decay].append(result["best_epoch"])
    selected = max(WEIGHT_DECAYS, key=lambda decay: np.mean(scores[decay]))
    fixed_epochs = max(1, int(np.median(epochs[selected])))
    result = fit_head(x[train], labels[train], x[test], None, n_classes=n_classes,
                      seed=seed, weight_decay=selected, fixed_epochs=fixed_epochs, device=device)
    return result["probabilities"], {"weight_decay": selected, "fixed_outer_epochs": fixed_epochs,
                                    "inner_best_epochs": epochs[selected],
                                    "outer_epoch_rule": "Median of inner best epochs; outer labels never passed"}


def evaluate(method, features, truth, groups, classes, device="cuda:0"):
    if method not in ("svm-reader", "small-mlp-reader"):
        raise ValueError("Unsupported frozen reader")
    validate_support(truth, groups, classes)
    x = unit(features)
    labels = np.asarray([classes.index(label) for label in truth])
    runs = []
    for seed in SEEDS:
        started = time.monotonic()
        oof, seen, traces = np.zeros((len(labels), len(classes))), np.zeros(len(labels), int), []
        for number, (train, test) in enumerate(grouped_folds(truth, groups, OUTER_FOLDS, seed)):
            if method == "svm-reader":
                probabilities, selected = svm_fold(x, labels, groups, train, test, seed + number)
            else:
                probabilities, selected = mlp_fold(x, labels, groups, train, test, seed + number, device)
            if probabilities.shape != (len(test), len(classes)):
                raise ValueError("Reader probabilities have wrong class order or shape")
            oof[test] = probabilities
            seen[test] += 1
            traces.append({"fold": number, "train_groups_sha256": group_sha(train, groups),
                           "test_groups_sha256": group_sha(test, groups), **selected})
        if not np.all(seen == 1):
            raise ValueError("Expected one OOF prediction per allowed image")
        runs.append({"seed": seed, "metrics": metrics(truth, oof, classes), "folds": traces,
                     "oof_probabilities": oof.tolist(), "seconds": time.monotonic() - started})
        print(f"{method} CV_seed={seed} macro_f1={runs[-1]['metrics']['macro_f1']:.6f}", flush=True)
    return runs
