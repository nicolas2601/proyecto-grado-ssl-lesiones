"""Nested patient-grouped CV: inner folds pick C, outer folds score. Dev only, no encoder gradients."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(sys.argv[1])
CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
SEEDS, OUTER, INNER, GRID = (20261025, 20261026, 20261027), 5, 3, (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fit(x, y, c):
    return LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(x, y)


def pick_c(x, y, g, seed):
    scores = {c: [] for c in GRID}
    for tr, va in StratifiedGroupKFold(INNER, shuffle=True, random_state=seed).split(x, y, g):
        for c in GRID:
            scores[c].append(f1_score(y[va], fit(x[tr], y[tr], c).predict(x[va]), average="macro", zero_division=0))
    return max(GRID, key=lambda c: np.mean(scores[c]))


out = {}
for model in ("S0", "refit"):
    for view in ("cls", "unit_concat"):
        name = f"{model}_{view}"
        d = np.load(ROOT / f"{name}.npz", allow_pickle=True)
        x, g = l2(d["features"].astype(np.float64)), d["groups"].astype(str)
        t = np.load(ROOT / f"{name}_scores.npz", allow_pickle=True)
        y = t[next(k for k in t.files if "truth" in k)]
        y = np.array([CLASSES[i] for i in y]) if y.dtype.kind in "iu" else y.astype(str)
        runs = []
        for seed in SEEDS:
            pred = np.empty(len(y), dtype=object)
            for tr, te in StratifiedGroupKFold(OUTER, shuffle=True, random_state=seed).split(x, y, g):
                pred[te] = fit(x[tr], y[tr], pick_c(x[tr], y[tr], g[tr], seed)).predict(x[te])
            pred = pred.astype(str)
            runs.append({
                "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
                "balanced_acc": float(balanced_accuracy_score(y, pred)),
                "recall": dict(zip(CLASSES, map(float, recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)))),
            })
        f1 = [r["macro_f1"] for r in runs]
        out[name] = {"macro_f1_mean": float(np.mean(f1)), "macro_f1_sd": float(np.std(f1)), "runs": runs}
print(json.dumps(out, indent=1))
