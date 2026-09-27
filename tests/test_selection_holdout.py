"""docs/28 selects pilot variants on a holdout carved out of TRAIN.

The validation set is scored once, at the end. Everything here guards that:
the holdout never touches validation, never splits a lesion across the two
sides, and is the same holdout every time it is rebuilt.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from dermssl import selection_holdout as sh  # noqa: E402


def _data(n_lesions: int = 200, per_lesion: int = 2, n_classes: int = 4):
    rng = np.random.default_rng(0)
    ids, labels, groups = [], [], []
    for lesion in range(n_lesions):
        cls = int(rng.integers(n_classes))
        for k in range(per_lesion):
            ids.append(f"ISIC_{lesion:05d}_{k}")
            labels.append(cls)
            groups.append(f"HAM_{lesion:05d}")
    return ids, np.array(labels), groups


def test_the_two_sides_partition_the_train_ids():
    ids, labels, groups = _data()
    fit, holdout = sh.grouped_holdout(ids, labels, groups)
    assert set(fit).isdisjoint(holdout)
    assert sorted(fit + holdout) == sorted(ids)


def test_the_holdout_is_about_a_fifth():
    ids, labels, groups = _data()
    _, holdout = sh.grouped_holdout(ids, labels, groups)
    assert 0.15 <= len(holdout) / len(ids) <= 0.25


def test_no_lesion_sits_on_both_sides():
    ids, labels, groups = _data()
    fit, holdout = sh.grouped_holdout(ids, labels, groups)
    lesion = dict(zip(ids, groups))
    assert {lesion[i] for i in fit}.isdisjoint({lesion[i] for i in holdout})


def test_it_is_the_same_holdout_every_time():
    ids, labels, groups = _data()
    assert sh.grouped_holdout(ids, labels, groups) == sh.grouped_holdout(ids, labels, groups)


def test_every_class_reaches_the_holdout():
    ids, labels, groups = _data()
    _, holdout = sh.grouped_holdout(ids, labels, groups)
    label = dict(zip(ids, labels))
    assert {label[i] for i in holdout} == set(labels.tolist())


def test_a_validation_id_in_the_input_is_refused():
    """The whole point: selection must never see the validation set."""
    ids, labels, groups = _data()
    with pytest.raises(ValueError, match="validacion"):
        sh.grouped_holdout(ids, labels, groups, forbidden=[ids[3]])


def test_misaligned_inputs_are_refused():
    ids, labels, groups = _data()
    with pytest.raises(ValueError):
        sh.grouped_holdout(ids, labels[:-1], groups)


class TestSelectionRule:
    """docs/28 §4.1: max macro-F1; within 0.005 of the best, lowest ECE."""

    def test_the_best_macro_f1_wins_when_the_gap_is_clear(self):
        results = {"A1": {"macro_f1": 0.62, "ece": 0.20}, "A2": {"macro_f1": 0.60, "ece": 0.05}}
        assert sh.select_winner(results) == "A1"

    def test_a_near_tie_goes_to_the_lower_ece(self):
        results = {"A1": {"macro_f1": 0.620, "ece": 0.20}, "A2": {"macro_f1": 0.617, "ece": 0.05}}
        assert sh.select_winner(results) == "A2"

    def test_exactly_the_tie_width_is_not_a_tie(self):
        results = {"A1": {"macro_f1": 0.620, "ece": 0.20}, "A2": {"macro_f1": 0.615, "ece": 0.05}}
        assert sh.select_winner(results) == "A1"

    def test_the_reference_is_never_a_candidate(self):
        results = {
            "public": {"macro_f1": 0.99, "ece": 0.0},
            "A1": {"macro_f1": 0.6, "ece": 0.1},
        }
        assert sh.select_winner(results, candidates=["A1"]) == "A1"

    def test_no_candidates_refuses(self):
        with pytest.raises(ValueError):
            sh.select_winner({}, candidates=[])


class TestBootstrapCI:
    """Reported next to the selection, never used by it (docs/28 4.1)."""

    def _data(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        proba = rng.random((300, 3))
        proba[np.arange(300), y] += 0.8
        return y, proba / proba.sum(axis=1, keepdims=True)

    def test_the_interval_contains_the_point_and_is_reproducible(self):
        from dermssl import paired_bootstrap as pb

        y, proba = self._data()
        low, high = sh.bootstrap_ci(y, proba, pb.metric_macro_f1)
        point = pb.metric_macro_f1(y, proba)
        assert low <= point <= high
        assert (low, high) == sh.bootstrap_ci(y, proba, pb.metric_macro_f1)

    def test_the_interval_has_width(self):
        from dermssl import paired_bootstrap as pb

        y, proba = self._data()
        low, high = sh.bootstrap_ci(y, proba, pb.metric_ece)
        assert high > low
