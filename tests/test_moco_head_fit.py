"""Final-reader fitting contracts; development CV is not a deployment test."""
from __future__ import annotations

import numpy as np
import pytest

from dermssl.release import moco_head_fit as fit


def test_features_are_normalized_per_image_without_a_dataset_scaler():
    x = np.array([[3, 4], [0, 2]], dtype=np.float32)
    result = fit.normalized_features(x)
    np.testing.assert_allclose(result, [[.6, .8], [0, 1]], atol=1e-7)
    assert result.dtype == np.float32


@pytest.mark.parametrize("x", [np.zeros((2, 3)), np.array([[np.nan, 2]]), np.array([[1, np.inf]])])
def test_invalid_features_fail_closed(x):
    with pytest.raises(ValueError, match="finite|norm"):
        fit.normalized_features(x)


def test_grouped_selection_refits_all_development_rows_and_breaks_ties_by_smaller_c():
    labels = np.repeat(np.array(["a", "b", "c"]), 18)
    x = np.eye(3, dtype=np.float32)[np.repeat(np.arange(3), 18)]
    groups = np.array([f"patient-{i}" for i in range(len(x))])
    artifact = fit.fit_reader(x, labels, groups, ("a", "b", "c"), c_grid=(1.0, .3), cv_seeds=(1, 2))
    assert artifact["C"] == .3
    assert artifact["n_images"] == 54
    assert artifact["n_groups"] == 54
    assert np.asarray(artifact["weight"]).shape == (3, 3)
    assert len(artifact["selection"]["folds"]) == 6
    probabilities = fit.reader_probabilities(x, artifact)
    assert np.array_equal(probabilities.argmax(1), np.repeat(np.arange(3), 18))
    np.testing.assert_allclose(probabilities.sum(1), 1)


def test_reader_rejects_misaligned_groups_and_unknown_classes():
    x = np.ones((12, 4))
    labels = np.array(["a", "b", "c"] * 4)
    with pytest.raises(ValueError, match="aligned"):
        fit.fit_reader(x, labels, ["one"], ("a", "b", "c"))
    with pytest.raises(ValueError, match="classes"):
        fit.fit_reader(x, labels, [f"g{i}" for i in range(12)], ("a", "b", "d"))


def test_reader_rejects_nonfinite_or_nonpositive_c_grid():
    labels = np.array(["a", "b", "c"] * 12)
    x = np.ones((len(labels), 4))
    for grid in ((float("nan"),), (0,), (-1,), (float("inf"),)):
        with pytest.raises(ValueError, match="C grid"):
            fit.fit_reader(x, labels, [f"g{i}" for i in range(len(x))], ("a", "b", "c"), c_grid=grid)
