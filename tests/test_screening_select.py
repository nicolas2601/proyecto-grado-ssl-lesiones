import numpy as np
import pytest

from dermssl.release import screening_select as ss


def test_zero_delta_keeps_the_argmax():
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    assert ss.screening_predictions(probs, target=1, delta=0.0).tolist() == [0, 1]


def test_delta_moves_a_close_call_to_the_target():
    probs = np.array([[0.55, 0.45, 0.0]])
    # log(0.45) + 0.25 = -0.549 > log(0.55) = -0.598
    assert ss.screening_predictions(probs, target=1, delta=0.25).tolist() == [1]
    assert ss.screening_predictions(probs, target=1, delta=0.1).tolist() == [0]


def test_zero_probability_never_wins_through_the_offset():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert ss.screening_predictions(probs, target=1, delta=3.0).tolist() == [0]


def test_negative_delta_is_rejected():
    with pytest.raises(ValueError):
        ss.screening_predictions(np.array([[0.5, 0.5]]), target=0, delta=-0.1)


def test_select_delta_takes_the_largest_value_within_tolerance():
    scores = {0.0: 0.614, 0.5: 0.615, 1.0: 0.605, 1.5: 0.603, 2.0: 0.580}
    assert ss.select_delta(scores, tolerance=0.01) == 1.0


def test_select_delta_keeps_a_passing_value_after_a_gap():
    # 1.5 fails, 2.0 passes again: the rule keeps the largest passing grid value.
    scores = {0.0: 0.60, 1.0: 0.595, 1.5: 0.58, 2.0: 0.595}
    assert ss.select_delta(scores, tolerance=0.01) == 2.0


def test_select_delta_requires_zero_in_the_grid():
    with pytest.raises(ValueError):
        ss.select_delta({0.5: 0.6}, tolerance=0.01)


def test_operating_metrics_counts_target_and_nevus_errors():
    classes = ["mel", "nv", "bkl"]
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 2, 1, 0, 2])
    metrics = ss.operating_metrics(y, pred, classes, target="mel", nevus="nv")
    assert metrics["target_recall"] == pytest.approx(0.5)
    assert metrics["target_to_nevus"] == pytest.approx(0.25)
    assert metrics["nevus_recall"] == pytest.approx(0.5)
    assert metrics["target_precision"] == pytest.approx(2 / 3)
    assert metrics["target_alert_rate"] == pytest.approx(3 / 7)
    assert set(metrics) >= {"macro_f1", "balanced_accuracy"}


def test_delta_grid_is_the_declared_one():
    grid = ss.DELTA_GRID
    assert grid[0] == 0.0 and grid[-1] == 3.0 and len(grid) == 13
    assert np.allclose(np.diff(grid), 0.25)


def test_operating_curve_covers_the_declared_grid():
    classes = ["mel", "nv", "bkl"]
    y = np.array([0, 0, 1, 1, 2])
    probs = np.array([[0.4, 0.6, 0.0], [0.7, 0.3, 0.0], [0.1, 0.9, 0.0], [0.3, 0.7, 0.0], [0.0, 0.1, 0.9]])

    curve = ss.operating_curve(y, probs, classes, target="mel", nevus="nv")

    assert list(curve) == list(ss.DELTA_GRID)
    assert curve[0.0]["target_recall"] == pytest.approx(0.5)
    assert curve[3.0]["target_recall"] == pytest.approx(1.0)
