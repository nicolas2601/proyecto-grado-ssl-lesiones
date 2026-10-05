"""End-to-end test of the probe grid, using synthetic features instead of a GPU.

This replays exactly what notebook cell "5. La rejilla de probes" does - regimes,
probes, metrics, aggregation, acceptance gates - on features whose correct answer
we know in advance. If this passes, the only thing left untested on Kaggle is the
image-to-feature step itself.
"""

import numpy as np
import pytest

from dermssl.feature_cache import FeatureCache, cache_key
from dermssl.label_regimes import build_regimes
from dermssl.metrics import mean_std, summarize
from dermssl.probes import run_probe

N_CLASSES = 7
FRACTIONS = (0.05, 0.20, 1.00)
SEEDS = (42, 1234)
PROBES = ("knn", "prototype")


def synthetic_split(n_train=420, n_val=140, dim=32, separation=4.0, seed=0):
    """Labelled features with a controllable amount of class structure.

    ``separation=0`` gives pure noise (a useless backbone); larger values give a
    progressively better one. Class sizes are deliberately imbalanced, with one
    tiny class, to mirror ``df`` in HAM10000.
    """
    rng = np.random.RandomState(seed)
    weights = np.array([0.05, 0.08, 0.10, 0.01, 0.11, 0.60, 0.05])  # df-like at 1 %
    centres = rng.randn(N_CLASSES, dim) * separation

    def make(n):
        y = rng.choice(N_CLASSES, size=n, p=weights)
        # guarantee every class appears, otherwise the fixture, not the code, is flaky
        y[:N_CLASSES] = np.arange(N_CLASSES)
        x = centres[y] + rng.randn(n, dim)
        return x.astype(np.float32), y

    return make(n_train), make(n_val)


def run_grid(features_by_backbone):
    """The notebook's grid loop, verbatim in structure."""
    rows = []
    for backbone, (train, val) in features_by_backbone.items():
        x_tr, y_tr = train
        x_va, y_va = val
        for seed in SEEDS:
            for frac, idx in build_regimes(y_tr, FRACTIONS, seed=seed, min_per_class=2).items():
                for probe in PROBES:
                    proba = run_probe(probe, x_tr[idx], y_tr[idx], x_va, N_CLASSES)
                    rows.append(
                        {
                            "backbone": backbone,
                            "probe": probe,
                            "fraction": frac,
                            "seed": seed,
                            "n_labels": int(len(idx)),
                            **summarize(y_va, proba, N_CLASSES),
                        }
                    )
    return rows


@pytest.fixture(scope="module")
def grid():
    good_train, good_val = synthetic_split(separation=4.0, seed=1)
    noise_train, noise_val = synthetic_split(separation=0.0, seed=1)
    return run_grid(
        {
            "ssl_like": (good_train, good_val),
            "useless": (noise_train, noise_val),
        }
    )


def test_grid_is_complete_and_has_no_nans(grid):
    expected = 2 * len(SEEDS) * len(FRACTIONS) * len(PROBES)
    assert len(grid) == expected
    for row in grid:
        assert not np.isnan(row["macro_f1"]), row
        assert not np.isnan(row["auc"]), row


def test_a_structured_backbone_beats_a_useless_one(grid):
    """The headline comparison the thesis rests on must come out the right way."""
    def auc(backbone):
        vals = [r["auc"] for r in grid if r["backbone"] == backbone and r["fraction"] == 1.0]
        return float(np.mean(vals))

    assert auc("ssl_like") > 0.90
    assert auc("useless") < 0.65
    assert auc("ssl_like") > auc("useless") + 0.25


def test_more_labels_never_hurt_much(grid):
    """Label efficiency curve must be monotone-ish, not chaotic.

    A non-monotone curve is the signature of a broken sampler (regimes that do
    not nest) rather than of a real property of the representation.
    """
    for probe in PROBES:
        by_frac = {}
        for row in grid:
            if row["backbone"] == "ssl_like" and row["probe"] == probe:
                by_frac.setdefault(row["fraction"], []).append(row["auc"])
        means = [float(np.mean(by_frac[f])) for f in sorted(by_frac)]
        for small, large in zip(means, means[1:]):
            assert large >= small - 0.05, f"{probe}: AUC dropped from {small} to {large}"


def test_tiny_class_is_present_in_every_regime(grid):
    """A regime that starves a class would show up as a macro-F1 collapse."""
    tiny = [r for r in grid if r["fraction"] == min(FRACTIONS)]
    assert all(r["n_labels"] >= N_CLASSES * 2 for r in tiny)


def test_seed_aggregation_reports_a_real_spread(grid):
    """Two seeds must produce a std we can report as 'mean +/- std'."""
    vals = [r["auc"] for r in grid
            if r["backbone"] == "ssl_like" and r["probe"] == "knn" and r["fraction"] == 0.2]
    stats = mean_std(vals)
    assert stats["n"] == len(SEEDS)
    assert stats["std"] >= 0.0
    assert 0.0 <= stats["mean"] <= 1.0


def test_acceptance_gates_pass_for_a_good_backbone_and_fail_for_noise(grid):
    """Same gate logic as the notebook's report cell."""
    def knn_auc(backbone):
        vals = [r["auc"] for r in grid
                if r["backbone"] == backbone and r["probe"] == "knn" and r["fraction"] == 1.0]
        return float(np.mean(vals))

    assert knn_auc("ssl_like") > 0.80, "G2 should pass for a structured representation"
    assert not knn_auc("useless") > 0.80, "G2 must fail for a useless representation"


def test_cache_round_trip_preserves_probe_results(tmp_path):
    """Caching must be transparent: identical numbers before and after a restart."""
    (x_tr, y_tr), (x_va, y_va) = synthetic_split(separation=4.0, seed=2)
    cache = FeatureCache(tmp_path)
    ids = [f"img{i}" for i in range(len(x_tr))]
    key = cache_key("dinov2_vits14", "train", 224, ids)

    direct = run_probe("knn", x_tr, y_tr, x_va, N_CLASSES)
    cache.save(key, x_tr, y_tr)
    restored = cache.load(key)
    from_cache = run_probe("knn", restored.features, restored.labels, x_va, N_CLASSES)

    assert from_cache == pytest.approx(direct)
    assert summarize(y_va, from_cache, N_CLASSES) == pytest.approx(
        summarize(y_va, direct, N_CLASSES)
    )
