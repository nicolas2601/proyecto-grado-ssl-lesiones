"""Development probe: patient-grouped CV linear readout on frozen features (no encoder gradients)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(sys.argv[1])
CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
SEED, FOLDS, GRID = 20261025, 5, (0.1, 1.0, 10.0)


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def truth_of(name):
    d = np.load(ROOT / f"{name}_scores.npz", allow_pickle=True)
    key = next(k for k in d.files if "truth" in k)
    t = d[key]
    return np.array([CLASSES[i] for i in t]) if t.dtype.kind in "iu" else t.astype(str)


out = {}
for model in ("S0", "refit"):
    for view in ("cls", "unit_concat"):
        name = f"{model}_{view}"
        d = np.load(ROOT / f"{name}.npz", allow_pickle=True)
        x, groups = l2(d["features"].astype(np.float64)), d["groups"].astype(str)
        y = truth_of(name)
        for c in GRID:
            pred = np.empty(len(y), dtype=object)
            for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(x, y, groups):
                clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
                pred[te] = clf.fit(x[tr], y[tr]).predict(x[te])
            pred = pred.astype(str)
            out[f"{name}|C={c}"] = {
                "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
                "balanced_acc": float(balanced_accuracy_score(y, pred)),
                "recall": dict(zip(CLASSES, map(float, recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)))),
            }
print(json.dumps(out, indent=1))
