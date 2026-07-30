"""Probes are validated on synthetic features with a known correct answer.

Separable clusters must be classified perfectly; noise must not be. Together
those two assertions catch both broken maths and accidentally leaked labels.
"""

import numpy as np
import pytest

from dermssl import metrics as m
from dermssl import probes as pr

N_CLASSES = 3
DIM = 8


def clustered_features(n_per_class: int, seed: int, spread: float = 0.05):
    """Three tight, well-separated clusters - trivially separable by design."""
    rng = np.random.RandomState(seed)
    centres = np.eye(N_CLASSES, DIM) * 5.0
    x, y = [], []
    for cls in range(N_CLASSES):
        x.append(centres[cls] + rng.randn(n_per_class, DIM) * spread)
        y.append(np.full(n_per_class, cls))
    return np.concatenate(x).astype(np.float32), np.concatenate(y)


def test_l2_normalize_gives_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = pr.l2_normalize(x)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert np.all(np.isfinite(out[1]))  # zero row must not become NaN


@pytest.mark.parametrize("probe", ["knn", "prototype"])
def test_separable_clusters_are_classified_perfectly(probe):
    x_tr, y_tr = clustered_features(20, seed=0)
    x_va, y_va = clustered_features(10, seed=1)
    proba = pr.run_probe(probe, x_tr, y_tr, x_va, N_CLASSES)
    assert proba.shape == (len(x_va), N_CLASSES)
    assert m.accuracy(y_va, proba.argmax(1)) == pytest.approx(1.0)
    assert m.auc_ovr_macro(y_va, proba) == pytest.approx(1.0)


@pytest.mark.parametrize("probe", ["knn", "prototype"])
def test_pure_noise_is_near_chance(probe):
    """Guards against a probe that somehow sees the validation labels."""
    rng = np.random.RandomState(3)
    x_tr = rng.randn(120, DIM).astype(np.float32)
    y_tr = rng.randint(0, N_CLASSES, 120)
    x_va = rng.randn(60, DIM).astype(np.float32)
    y_va = rng.randint(0, N_CLASSES, 60)
    proba = pr.run_probe(probe, x_tr, y_tr, x_va, N_CLASSES)
    assert m.accuracy(y_va, proba.argmax(1)) < 0.6


@pytest.mark.parametrize("probe", ["knn", "prototype"])
def test_probabilities_are_valid_distributions(probe):
    x_tr, y_tr = clustered_features(15, seed=4)
    x_va, _ = clustered_features(5, seed=5)
    proba = pr.run_probe(probe, x_tr, y_tr, x_va, N_CLASSES)
    assert np.all(proba >= 0.0)
    assert proba.sum(axis=1) == pytest.approx(np.ones(len(x_va)))


def test_knn_clips_k_to_the_training_set_size():
    """The 1 % regime can leave fewer samples than the requested k."""
    x_tr, y_tr = clustered_features(1, seed=6)  # only 3 training samples
    x_va, y_va = clustered_features(4, seed=7)
    proba = pr.knn_predict_proba(x_tr, y_tr, x_va, N_CLASSES, k=20)
    assert proba.shape == (len(x_va), N_CLASSES)
    assert m.accuracy(y_va, proba.argmax(1)) == pytest.approx(1.0)


def test_knn_chunking_matches_single_pass(monkeypatch):
    """A validation set larger than CHUNK must give identical results."""
    x_tr, y_tr = clustered_features(10, seed=8)
    x_va, _ = clustered_features(40, seed=9)  # 120 val rows
    reference = pr.knn_predict_proba(x_tr, y_tr, x_va, N_CLASSES, k=5)
    monkeypatch.setattr(pr, "CHUNK", 7)
    chunked = pr.knn_predict_proba(x_tr, y_tr, x_va, N_CLASSES, k=5)
    assert chunked == pytest.approx(reference)


def test_prototype_handles_a_class_missing_from_train():
    x_tr, y_tr = clustered_features(10, seed=10)
    keep = y_tr != 2  # class 2 has no labels at all
    x_va, y_va = clustered_features(5, seed=11)
    proba = pr.prototype_predict_proba(x_tr[keep], y_tr[keep], x_va, N_CLASSES)
    assert proba.shape[1] == N_CLASSES
    # classes 0 and 1 are still recovered correctly
    mask = y_va != 2
    assert m.accuracy(y_va[mask], proba.argmax(1)[mask]) == pytest.approx(1.0)


def test_knn_rejects_empty_training_set():
    with pytest.raises(ValueError):
        pr.knn_predict_proba(np.zeros((0, DIM)), np.array([]), np.zeros((2, DIM)), N_CLASSES)


def test_unknown_probe_name_is_an_error():
    x_tr, y_tr = clustered_features(5, seed=12)
    with pytest.raises(ValueError, match="unknown probe"):
        pr.run_probe("magic", x_tr, y_tr, x_tr, N_CLASSES)


def test_logreg_probe_when_sklearn_available():
    pytest.importorskip("sklearn")
    x_tr, y_tr = clustered_features(20, seed=13)
    x_va, y_va = clustered_features(8, seed=14)
    proba = pr.logreg_predict_proba(x_tr, y_tr, x_va, N_CLASSES)
    assert proba.shape == (len(x_va), N_CLASSES)
    assert m.accuracy(y_va, proba.argmax(1)) == pytest.approx(1.0)
