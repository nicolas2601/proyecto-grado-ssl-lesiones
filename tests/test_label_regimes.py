"""Regimes must nest, must never starve a class, and must be reproducible."""

import numpy as np
import pytest

from dermssl import label_regimes as lr


def imbalanced_labels() -> np.ndarray:
    """Shape of the real problem: one huge class, one nearly empty one.

    Proportions mirror HAM10000: nv dominates, df is the starvation candidate.
    """
    return np.concatenate(
        [
            np.zeros(50, dtype=int),  # akiec-like
            np.ones(5000, dtype=int),  # nv-like majority
            np.full(8, 2, dtype=int),  # df-like: tiny
            np.full(300, 3, dtype=int),
        ]
    )


def test_every_class_survives_the_1_percent_regime():
    """1 % of 5358 samples naively sampled can drop the 8-sample class."""
    y = imbalanced_labels()
    idx = lr.stratified_regime_indices(y, 0.01, seed=42, min_per_class=2)
    present = set(np.unique(y[idx]).tolist())
    assert present == {0, 1, 2, 3}
    assert (y[idx] == 2).sum() >= 2


def test_regimes_are_nested():
    y = imbalanced_labels()
    regimes = lr.build_regimes(y, fractions=(0.01, 0.05, 0.1, 0.5, 1.0), seed=7)
    fractions = sorted(regimes)
    for small, large in zip(fractions, fractions[1:]):
        assert set(regimes[small]).issubset(set(regimes[large])), (
            f"regime {small} is not contained in {large}"
        )


def test_full_regime_returns_everything_exactly_once():
    y = imbalanced_labels()
    idx = lr.stratified_regime_indices(y, 1.0, seed=1)
    assert len(idx) == len(y)
    assert np.array_equal(idx, np.arange(len(y)))


def test_indices_are_sorted_unique_and_in_range():
    y = imbalanced_labels()
    idx = lr.stratified_regime_indices(y, 0.2, seed=3)
    assert np.array_equal(idx, np.sort(idx))
    assert len(set(idx.tolist())) == len(idx)
    assert idx.min() >= 0 and idx.max() < len(y)


def test_same_seed_is_deterministic_and_different_seed_differs():
    y = imbalanced_labels()
    a = lr.stratified_regime_indices(y, 0.1, seed=42)
    b = lr.stratified_regime_indices(y, 0.1, seed=42)
    c = lr.stratified_regime_indices(y, 0.1, seed=1234)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_proportions_are_respected_for_large_classes():
    y = imbalanced_labels()
    idx = lr.stratified_regime_indices(y, 0.1, seed=42, min_per_class=2)
    # 10 % of the 5000-sample majority class, allowing rounding.
    assert abs((y[idx] == 1).sum() - 500) <= 1


def test_min_per_class_cannot_exceed_class_size():
    y = np.array([0, 0, 1])  # class 1 has a single sample
    idx = lr.stratified_regime_indices(y, 0.01, seed=0, min_per_class=5)
    assert (y[idx] == 1).sum() == 1


def test_regime_size_bounds():
    assert lr.regime_size(100, 0.1, min_per_class=2) == 10
    assert lr.regime_size(8, 0.01, min_per_class=2) == 2
    assert lr.regime_size(8, 1.0, min_per_class=2) == 8


def test_invalid_inputs_raise():
    y = imbalanced_labels()
    with pytest.raises(lr.RegimeError):
        lr.stratified_regime_indices(y, 0.0)
    with pytest.raises(lr.RegimeError):
        lr.stratified_regime_indices(y, 1.5)
    with pytest.raises(lr.RegimeError):
        lr.stratified_regime_indices(np.array([]), 0.5)


def test_describe_regimes_audit_flags_min_class():
    y = imbalanced_labels()
    regimes = lr.build_regimes(y, fractions=(0.01, 1.0), seed=42)
    rows = lr.describe_regimes(y, regimes)
    assert [r["fraction"] for r in rows] == [0.01, 1.0]
    assert rows[0]["min_class_count"] >= 2
    assert rows[-1]["n_labels"] == len(y)
    assert rows[-1]["pct_of_train"] == 100.0
