"""Full metric table (repo-table metrics) from stored out-of-fold probabilities.

Usage: python scripts/full_metrics_20261030.py OUT.json NAME=EVAL_DIR[,EVAL_DIR...] ...
Per dataset (PAD, HAM) and per name: accuracy, balanced accuracy, macro F1/precision/recall, macro AUC OvR,
ECE and per-class precision/recall/F1, each averaged over every CV-seed run of every listed evaluation dir.
"""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate_overlay_readout import load_labels

ROOT = Path(__file__).resolve().parents[1]
FILES = {"PAD": "PAD-readout.json", "HAM": "HAM-readout-06aee5.json"}
META = {"PAD": ("/home/jovyan/work/pad_ufes_20/metadata.csv", "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"),
        "HAM": ("/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv", "06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5")}
ECE_BINS = 15


def ece(y, proba, bins=ECE_BINS):
    confidence, predicted = proba.max(axis=1), proba.argmax(axis=1)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            total += inside.mean() * abs((predicted[inside] == y[inside]).mean() - confidence[inside].mean())
    return float(total)


def run_metrics(y, proba):
    n_classes = proba.shape[1]
    pred = proba.argmax(axis=1)
    per_class = []
    for k in range(n_classes):
        tp = int(np.sum((y == k) & (pred == k)))
        predicted, support = int(np.sum(pred == k)), int(np.sum(y == k))
        precision, recall = (tp / predicted if predicted else 0.0), (tp / support if support else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1))
    aucs = [roc_auc_score(y == k, proba[:, k]) for k in range(n_classes) if (y == k).any() and (y != k).any()]
    p, r, f = (np.mean([c[i] for c in per_class]) for i in range(3))
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(r), "macro_f1": float(f),
            "macro_precision": float(p), "macro_recall": float(r), "auc_ovr_macro": float(np.mean(aucs)),
            "ece": ece(y, proba), "per_class": per_class}


def dataset_metrics(kind, directories):
    runs = []
    for directory in directories:
        readout = json.loads((ROOT / "results" / directory / FILES[kind]).read_text())
        classes = readout["classes"]
        metadata, sha = META[kind]
        truth = load_labels(kind, [{"image_id": i} for i in readout["image_ids"]], Path(metadata), sha)
        y = np.asarray([classes.index(v) for v in truth])
        runs += [run_metrics(y, np.asarray(r["oof_probabilities"])) for r in readout["runs"]]
    keys = [k for k in runs[0] if k != "per_class"]
    mean = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    mean["per_class"] = {name: {m: float(np.mean([r["per_class"][i][j] for r in runs])) for j, m in enumerate(("precision", "recall", "f1"))}
                         for i, name in enumerate(classes)}
    return mean


def main(argv):
    result = {}
    for spec in argv[2:]:
        name, directories = spec.split("=", 1)
        result[name] = {kind: dataset_metrics(kind, directories.split(",")) for kind in FILES
                        if all((ROOT / "results" / d / FILES[kind]).exists() for d in directories.split(","))}
    Path(argv[1]).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"names": len(result)}))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv)
