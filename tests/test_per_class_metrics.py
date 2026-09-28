"""Per-class metrics behind resultados/clasificacion/TABLA_CLASIFICADOR.md.

scikit-learn appears here only as an oracle: the module under test is numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import per_class_metrics as pcm  # noqa: E402

CLASSES = ("a", "b", "c")


def _onehot(pred: list[int], n: int = 3) -> np.ndarray:
    proba = np.full((len(pred), n), 0.1)
    proba[np.arange(len(pred)), pred] = 0.8
    return proba


def test_confusion_matrix_on_a_hand_example():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    report = pcm.class_report(y_true, _onehot([0, 1, 1, 1, 0, 2]), CLASSES, positive="c")
    expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    np.testing.assert_array_equal(report["confusion"], expected)
    np.testing.assert_allclose(report["recall"], [0.5, 1.0, 0.5])


def test_a_class_never_predicted_scores_f1_zero_not_nan():
    y_true = np.array([0, 1, 2, 2])
    report = pcm.class_report(y_true, _onehot([0, 1, 1, 1]), CLASSES, positive="c")
    assert report["f1"][2] == 0.0
    assert not np.isnan(report["f1"]).any()
    assert report["positive_sensitivity"] == 0.0


def test_macro_f1_is_the_mean_of_the_per_class_f1():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    report = pcm.class_report(y_true, _onehot([0, 1, 1, 1, 0, 2]), CLASSES, positive="a")
    assert report["macro_f1"] == pytest.approx(float(np.mean(report["f1"])))


def test_positive_auc_matches_sklearn_with_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 3, size=400)
    proba = np.round(rng.random((400, 3)), 1)  # coarse grid: many tied scores
    report = pcm.class_report(y_true, proba, CLASSES, positive="b")
    oracle = metrics.roc_auc_score((y_true == 1).astype(int), proba[:, 1])
    assert report["positive_auc"] == pytest.approx(oracle, abs=1e-12)


def test_per_class_f1_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 3, size=300)
    proba = rng.random((300, 3))
    report = pcm.class_report(y_true, proba, CLASSES, positive="a")
    oracle = metrics.f1_score(y_true, proba.argmax(1), average=None, labels=[0, 1, 2])
    np.testing.assert_allclose(report["f1"], oracle, atol=1e-12)


def test_unknown_positive_class_is_rejected():
    with pytest.raises(ValueError, match="mel"):
        pcm.class_report(np.array([0]), _onehot([0]), CLASSES, positive="mel")


def test_class_order_guard():
    pcm.check_class_order(list(pcm.HAM10000_CLASSES))
    with pytest.raises(ValueError, match="orden de clases"):
        pcm.check_class_order(["bcc", "akiec", "bkl", "df", "mel", "nv", "vasc"])
