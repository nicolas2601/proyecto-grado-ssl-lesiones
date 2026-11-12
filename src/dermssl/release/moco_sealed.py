"""Scoring contract for the one-shot sealed evaluation (ham_holdout, pad_test). No I/O of sealed rows here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLINICAL_TARGETS = {"HAM": ("mel",), "PAD": ("MEL", "SCC")}
MAX_MACRO_F1_DROP = 0.05
SEALED_ROLES = {"HAM": "ham_holdout", "PAD": "pad_test"}


def class_metrics(y_true, probabilities, classes):
    """Macro-F1, balanced accuracy, per-class recall and confusion matrix from probabilities (argmax)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    truth = np.asarray(y_true).astype(str)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape != (len(truth), len(classes)) or not np.isfinite(probs).all():
        raise ValueError("Probabilities must be a finite (n, classes) matrix aligned with labels")
    predicted = np.asarray(classes)[probs.argmax(1)]
    recall = {}
    for index, name in enumerate(classes):
        support = int(np.sum(truth == name))
        hits = int(np.sum((truth == name) & (predicted == name)))
        recall[name] = float(hits / support) if support else None
    matrix = [[int(np.sum((truth == t) & (predicted == p))) for p in classes] for t in classes]
    return {"n": int(len(truth)),
            "macro_f1": float(f1_score(truth, predicted, labels=list(classes), average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "recall": recall, "confusion_matrix": matrix, "classes": list(classes)}


def clinical_score(ham_metrics, pad_metrics):
    """Mean of HAM mel recall, PAD MEL recall and PAD SCC recall (the registered release criterion)."""
    values = [ham_metrics["recall"][c] for c in CLINICAL_TARGETS["HAM"]]
    values += [pad_metrics["recall"][c] for c in CLINICAL_TARGETS["PAD"]]
    if any(v is None for v in values):
        raise ValueError("Clinical targets need support in the evaluated role")
    return float(np.mean(values))


def release_rule(candidate, reference):
    """Pass iff the clinical score improves and macro-F1 drops at most 0.05 on each dataset."""
    drops = {task: reference[task]["macro_f1"] - candidate[task]["macro_f1"] for task in ("HAM", "PAD")}
    clinical_candidate = clinical_score(candidate["HAM"], candidate["PAD"])
    clinical_reference = clinical_score(reference["HAM"], reference["PAD"])
    passed = (clinical_candidate > clinical_reference
              and all(drop <= MAX_MACRO_F1_DROP for drop in drops.values()))
    return {"passed": bool(passed), "clinical_candidate": clinical_candidate,
            "clinical_reference": clinical_reference, "clinical_delta": clinical_candidate - clinical_reference,
            "macro_f1_drop": drops, "max_macro_f1_drop": MAX_MACRO_F1_DROP}


def claim_once(path):
    """Create the one-shot receipt; a second sealed evaluation must fail closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            stream.write("claimed\n")
    except FileExistsError as exc:
        raise ValueError("Sealed evaluation already consumed; refusing a second run") from exc


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
