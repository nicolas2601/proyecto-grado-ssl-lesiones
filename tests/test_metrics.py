"""Metrics are checked against values computed by hand, not against a library."""

import numpy as np
import pytest

from dermssl import metrics as m


def test_accuracy():
    assert m.accuracy([0, 1, 2, 2], [0, 1, 2, 0]) == pytest.approx(0.75)


def test_confusion_matrix_orientation():
    cm = m.confusion_matrix([0, 0, 1], [0, 1, 1], n_classes=2)
    # row = truth, column = prediction
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_macro_f1_hand_computed():
    """y=[0,0,1,1], pred=[0,1,1,1] -> f1(class0)=2/3, f1(class1)=0.8."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    f1 = m.per_class_f1(y, p, 2)
    assert f1[0] == pytest.approx(2 / 3)
    assert f1[1] == pytest.approx(0.8)
    assert m.macro_f1(y, p, 2) == pytest.approx((2 / 3 + 0.8) / 2)


def test_class_never_predicted_scores_zero_not_nan():
    y = np.array([0, 0, 1])
    p = np.array([0, 0, 0])
    f1 = m.per_class_f1(y, p, 3)
    assert f1[1] == 0.0
    assert f1[2] == 0.0  # class absent from both -> 0, never NaN
    assert not np.isnan(m.macro_f1(y, p, 3))


def test_recall_per_class():
    r = m.recall_per_class([0, 0, 1, 1], [0, 1, 1, 1], n_classes=2)
    assert r.tolist() == pytest.approx([0.5, 1.0])


def test_auc_perfect_separation_is_one():
    assert m.binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_auc_inverted_separation_is_zero():
    assert m.binary_auc([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.0)


def test_auc_all_scores_tied_is_half():
    """Without average-rank tie handling this would come out wrongly optimistic."""
    assert m.binary_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_partial_hand_computed():
    """y=[0,1,0,1] with increasing scores -> positive ranks 2 and 4 -> 0.75."""
    assert m.binary_auc([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.75)


def test_auc_is_nan_for_single_class():
    assert np.isnan(m.binary_auc([1, 1, 1], [0.1, 0.5, 0.9]))


def test_auc_ovr_macro_skips_undefined_classes():
    y = np.array([0, 0, 1, 1])
    proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    # class 2 never appears -> undefined -> skipped, not counted as 0.5
    assert m.auc_ovr_macro(y, proba) == pytest.approx(1.0)


def test_ece_zero_when_perfectly_calibrated_and_certain():
    y = np.array([0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert m.ece(y, proba) == pytest.approx(0.0)


def test_ece_is_large_when_confidently_wrong():
    y = np.array([1, 1, 1, 1])
    proba = np.tile([0.99, 0.01], (4, 1))  # 99 % confident, always wrong
    assert m.ece(y, proba) > 0.9


def test_summarize_returns_every_headline_metric():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]])
    out = m.summarize(y, proba, n_classes=2)
    assert set(out) == {"macro_f1", "accuracy", "auc", "ece"}
    assert out["accuracy"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    proba = np.zeros((200, 2))
    # informative but imperfect scores
    proba[np.arange(200), y] = 0.7
    proba[np.arange(200), 1 - y] = 0.3
    noise = rng.rand(200) < 0.2
    proba[noise] = proba[noise][:, ::-1]

    point, lo, hi = m.bootstrap_ci(
        y, proba, lambda yy, pp: m.auc_ovr_macro(yy, pp), n_resamples=200, seed=1
    )
    assert lo <= point <= hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_mean_std_ignores_nan():
    out = m.mean_std([0.5, 0.7, float("nan")])
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(0.6)
    assert out["std"] == pytest.approx(0.1)


def test_mean_std_all_nan_is_safe():
    out = m.mean_std([float("nan")])
    assert out["n"] == 0
    assert np.isnan(out["mean"])
