"""MCC and macro metrics recomputed from a confusion matrix match scikit-learn and the JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from dermssl.bench import metrics

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "results" / "baselines_isic2019" / "results_dinov2_vitb14_isic2019.json"


def _expand(cm):
    y_true, y_pred = [], []
    for t, row in enumerate(cm):
        for p, count in enumerate(row):
            y_true += [t] * count
            y_pred += [p] * count
    return y_true, y_pred


def test_mcc_matches_sklearn_on_an_expanded_label_vector_of_a_committed_json():
    probe = json.loads(BASELINE.read_text(encoding="utf-8"))["probes"]["logreg"]
    y_true, y_pred = _expand(probe["matriz_confusion"])
    assert len(y_true) == 5067
    assert metrics.mcc_from_confusion(probe["matriz_confusion"]) == pytest.approx(
        matthews_corrcoef(y_true, y_pred), abs=1e-12)


def test_macro_precision_recall_reproduce_the_committed_numbers():
    probe = json.loads(BASELINE.read_text(encoding="utf-8"))["probes"]["logreg"]
    precision, recall = metrics.macro_precision_recall(probe["matriz_confusion"])
    assert precision == pytest.approx(probe["macro_precision"], abs=1e-12)
    assert recall == pytest.approx(probe["macro_recall"], abs=1e-12)


def test_degenerate_matrix_gives_zero_like_sklearn():
    cm = [[6538, 0], [122, 0]]
    y_true, y_pred = _expand(cm)
    assert metrics.mcc_from_confusion(cm) == 0.0 == matthews_corrcoef(y_true, y_pred)


def test_perfect_and_inverted_two_class_matrices():
    assert metrics.mcc_from_confusion([[5, 0], [0, 7]]) == pytest.approx(1.0)
    assert metrics.mcc_from_confusion([[0, 5], [7, 0]]) == pytest.approx(-1.0)


def test_non_square_matrix_is_rejected():
    with pytest.raises(ValueError):
        metrics.mcc_from_confusion([[1, 2, 3], [4, 5, 6]])


def test_confusion_from_probabilities_and_row_normalization():
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    cm = metrics.confusion_from_probabilities(np.array([0, 1, 1, 1]), proba)
    assert cm == [[1, 0], [1, 2]]
    assert metrics.row_normalized(cm) == [[1.0, 0.0], [pytest.approx(1 / 3), pytest.approx(2 / 3)]]
    assert metrics.row_normalized([[0, 0], [1, 1]])[0] == [0.0, 0.0]
