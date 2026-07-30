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


def test_effective_k_is_only_a_crash_guard():
    """Shrinking k was measured and rejected: it trades AUC away. See effective_k."""
    assert pr.effective_k(20, n_train=648, n_classes=7) == 20
    assert pr.effective_k(20, n_train=8001, n_classes=7) == 20
    assert pr.effective_k(20, n_train=37, n_classes=7) == 20
    assert pr.effective_k(20, n_train=3, n_classes=7) == 3  # clipped to the set size
    assert pr.effective_k(20, n_train=1, n_classes=7) == 1
    with pytest.raises(ValueError):
        pr.effective_k(20, n_train=0, n_classes=7)


# Real HAM10000 class counts, from the run log: nv, mel, bkl, bcc, akiec, vasc, df.
HAM_COUNTS = np.array([6705, 1113, 1099, 514, 327, 142, 115], dtype=float)
HAM_WEIGHTS = HAM_COUNTS / HAM_COUNTS.sum()


def ham_shaped_split(n_train=648, n_val=169, seed=42, dim=384, separation=0.10):
    """Synthetic features with HAM10000's exact class proportions and overlap.

    At n_train=648 this gives roughly 440 ``nv`` against 10 ``df`` - the ratio
    that makes an unweighted 20-neighbour vote unable to ever pick ``df``.
    ``separation`` is low on purpose: real dermatoscopic classes overlap, and a
    cleanly separable fixture would hide the effect entirely. ``dim=384`` matches
    DINOv2 ViT-S/14, the backbone whose run this reproduces - dimensionality
    changes the geometry, so it is part of the fixture, not an arbitrary choice.
    """
    rng = np.random.RandomState(seed)
    centres = rng.randn(7, dim) * separation

    def make(n):
        y = rng.choice(7, size=n, p=HAM_WEIGHTS)
        y[:7] = np.arange(7)  # every class present
        return (centres[y] + rng.randn(n, dim)).astype(np.float32), y

    return make(n_train), make(n_val)


def test_plain_knn_reproduces_the_kaggle_majority_collapse():
    """Regression pin for the 2026-07-29 symptom: accuracy 0.663 vs macro-F1 0.181.

    A class with ~10 labelled samples cannot out-vote one with ~440, so the argmax
    only ever names a couple of classes even though the ranking (AUC) is fine.
    """
    (x_tr, y_tr), (x_va, y_va) = ham_shaped_split()
    proba = pr.knn_predict_proba(x_tr, y_tr, x_va, 7, k=20)
    pred = proba.argmax(1)

    assert len(set(pred.tolist())) <= 3, "the collapse means most classes are never predicted"
    assert m.accuracy(y_va, pred) > 3 * m.macro_f1(y_va, pred, 7), (
        "accuracy tracks the majority prior while macro-F1 sits at the floor"
    )
    assert m.auc_ovr_macro(y_va, proba) > 0.65, "ranking survives; only the argmax collapses"


def test_lowering_k_trades_auc_for_macro_f1():
    """Pins the measured trade-off so nobody 'fixes' the collapse by shrinking k."""
    (x_tr, y_tr), (x_va, y_va) = ham_shaped_split()
    wide = pr.knn_predict_proba(x_tr, y_tr, x_va, 7, k=20)
    narrow = pr.knn_predict_proba(x_tr, y_tr, x_va, 7, k=3)

    assert m.macro_f1(y_va, narrow.argmax(1), 7) > m.macro_f1(y_va, wide.argmax(1), 7)
    assert m.auc_ovr_macro(y_va, narrow) < m.auc_ovr_macro(y_va, wide), (
        "small k destroys the ranking: that is why k is not shrunk"
    )


def test_balanced_knn_recovers_every_class():
    """Dividing votes by class frequency removes the prior, at a cost in accuracy."""
    (x_tr, y_tr), (x_va, y_va) = ham_shaped_split()
    plain = pr.knn_predict_proba(x_tr, y_tr, x_va, 7, k=20)
    balanced = pr.knn_balanced_predict_proba(x_tr, y_tr, x_va, 7, k=20)

    assert len(set(balanced.argmax(1).tolist())) > len(set(plain.argmax(1).tolist()))
    assert m.macro_f1(y_va, balanced.argmax(1), 7) > m.macro_f1(y_va, plain.argmax(1), 7)
    # Not a free win, and the report must show both rather than swap one in.
    assert m.accuracy(y_va, balanced.argmax(1)) < m.accuracy(y_va, plain.argmax(1))


def test_balancing_barely_moves_the_ranking():
    """AUC is rank based, so re-weighting classes should not change it much."""
    (x_tr, y_tr), (x_va, y_va) = ham_shaped_split(seed=7)
    plain = pr.knn_predict_proba(x_tr, y_tr, x_va, 7, k=20)
    balanced = pr.knn_balanced_predict_proba(x_tr, y_tr, x_va, 7, k=20)
    assert m.auc_ovr_macro(y_va, balanced) == pytest.approx(
        m.auc_ovr_macro(y_va, plain), abs=0.08
    )


def test_balanced_variants_are_registered_as_separate_probes():
    """Different estimators must be reported side by side, never swapped in."""
    for name in ("knn", "knn_balanced", "prototype", "logreg", "logreg_balanced"):
        assert name in pr.PROBES
    assert pr.PROBES["knn"] is not pr.PROBES["knn_balanced"]
    assert pr.PROBES["logreg"] is not pr.PROBES["logreg_balanced"]


def test_uncalibrated_probes_are_flagged():
    """Their softmax temperature is arbitrary, so their ECE is not a claim."""
    assert set(pr.UNCALIBRATED_PROBES) == {"knn", "knn_balanced", "prototype"}
    assert "logreg" not in pr.UNCALIBRATED_PROBES


def test_headline_metric_forbids_quoting_knn_macro_f1():
    """The reporting rule that follows from the measurement above."""
    assert pr.headline_metric("knn") == "auc"
    assert pr.headline_metric("prototype") == "auc"
    assert pr.headline_metric("logreg") == "macro_f1"
    assert pr.headline_metric("logreg_balanced") == "macro_f1"
    assert pr.headline_metric("knn_balanced") == "macro_f1"  # prior removed by design


def test_balanced_knn_tolerates_classes_absent_from_the_labels():
    x_tr, y_tr = clustered_features(10, seed=30)
    keep = y_tr != 2
    x_va, _ = clustered_features(5, seed=31)
    proba = pr.knn_balanced_predict_proba(x_tr[keep], y_tr[keep], x_va, N_CLASSES)
    assert proba.shape == (len(x_va), N_CLASSES)
    assert proba.sum(axis=1) == pytest.approx(np.ones(len(x_va)))


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
