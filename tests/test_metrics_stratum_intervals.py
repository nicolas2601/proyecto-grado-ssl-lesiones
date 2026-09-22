"""A stratum gap is not a result until it carries its uncertainty.

The first bank reported the worst tone stratum as the LIGHT one in all three
backbones - contradicting the segmentation finding - but that stratum had 183
samples against 820 and 957, spread over seven classes. With only a point
estimate there was no way to say whether that was an effect or sampling
noise, so the gap could not be reported at all. A bootstrap interval per
stratum answers it: overlapping intervals mean the ordering is not
established.
"""

import numpy as np

from dermssl.metrics import bootstrap_interval, stratified_by_label


def _proba(y_true, n_classes, correct_ratio, seed=0):
    """Predicted probabilities where a known fraction of rows is right."""
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0.0, 0.2, size=(len(y_true), n_classes))
    for row, label in enumerate(y_true):
        winner = label if rng.random() < correct_ratio else (label + 1) % n_classes
        proba[row, winner] = 0.9
    return proba / proba.sum(axis=1, keepdims=True)


def test_the_interval_brackets_the_point_estimate():
    values = np.array([0.6, 0.7, 0.65, 0.72, 0.58, 0.61, 0.69, 0.63])
    low, high = bootstrap_interval(values)
    assert low <= values.mean() <= high
    assert low < high


def test_a_single_value_has_no_interval():
    low, high = bootstrap_interval(np.array([0.5]))
    assert np.isnan(low) and np.isnan(high)


def test_an_empty_vector_has_no_interval():
    low, high = bootstrap_interval(np.array([]))
    assert np.isnan(low) and np.isnan(high)


def test_the_interval_is_reproducible_for_a_given_seed():
    values = np.linspace(0.4, 0.9, 40)
    assert bootstrap_interval(values) == bootstrap_interval(values)


def test_a_smaller_stratum_gets_a_wider_interval():
    """The whole point: 183 samples cannot claim what 957 can."""
    rng = np.random.default_rng(7)
    big = rng.normal(0.6, 0.1, size=900)
    small = rng.normal(0.6, 0.1, size=180)
    big_low, big_high = bootstrap_interval(big)
    small_low, small_high = bootstrap_interval(small)
    assert (small_high - small_low) > (big_high - big_low)


def test_every_qualifying_stratum_reports_an_interval_of_its_macro_f1():
    n_classes = 3
    labels = np.array(["dark"] * 60 + ["light"] * 60)
    y_true = np.array([i % n_classes for i in range(120)])
    proba = _proba(y_true, n_classes, correct_ratio=0.8)

    strata = stratified_by_label(y_true, proba, labels, n_classes, min_n=30)

    for key in ("dark", "light"):
        entry = strata[key]
        assert entry["status"] == "OK"
        assert "macro_f1_ci95_low" in entry and "macro_f1_ci95_high" in entry
        assert entry["macro_f1_ci95_low"] <= entry["macro_f1"] <= entry["macro_f1_ci95_high"]


def test_a_skipped_stratum_carries_no_interval():
    labels = np.array(["dark"] * 60 + ["light"] * 5)
    y_true = np.array([i % 3 for i in range(65)])
    proba = _proba(y_true, 3, correct_ratio=0.8)

    strata = stratified_by_label(y_true, proba, labels, 3, min_n=30)

    assert strata["light"]["status"] == "SKIPPED"
    assert "macro_f1_ci95_low" not in strata["light"]


def test_overlapping_intervals_are_visible_to_the_caller():
    """Two strata drawn from the same distribution must not look separated."""
    n_classes = 3
    labels = np.array(["dark"] * 90 + ["light"] * 90)
    y_true = np.array([i % n_classes for i in range(180)])
    proba = _proba(y_true, n_classes, correct_ratio=0.75, seed=3)

    strata = stratified_by_label(y_true, proba, labels, n_classes, min_n=30)
    dark, light = strata["dark"], strata["light"]

    overlap = (
        dark["macro_f1_ci95_low"] <= light["macro_f1_ci95_high"]
        and light["macro_f1_ci95_low"] <= dark["macro_f1_ci95_high"]
    )
    assert overlap, "same distribution: the intervals have to overlap"


# --------------------------------------------- the gap has to say if it holds
def test_a_gap_whose_intervals_overlap_is_reported_as_not_established():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cluster_frozen_bank as bank

    strata = {
        "dark": {"status": "OK", "macro_f1": 0.50,
                 "macro_f1_ci95_low": 0.44, "macro_f1_ci95_high": 0.56},
        "light": {"status": "OK", "macro_f1": 0.55,
                  "macro_f1_ci95_low": 0.49, "macro_f1_ci95_high": 0.61},
    }
    assert bank.gap_is_established(strata, "macro_f1") is False


def test_a_gap_whose_intervals_are_separated_is_established():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cluster_frozen_bank as bank

    strata = {
        "dark": {"status": "OK", "macro_f1": 0.40,
                 "macro_f1_ci95_low": 0.36, "macro_f1_ci95_high": 0.44},
        "light": {"status": "OK", "macro_f1": 0.60,
                  "macro_f1_ci95_low": 0.56, "macro_f1_ci95_high": 0.64},
    }
    assert bank.gap_is_established(strata, "macro_f1") is True


def test_without_intervals_the_gap_cannot_be_established():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cluster_frozen_bank as bank

    strata = {
        "dark": {"status": "OK", "macro_f1": 0.40},
        "light": {"status": "OK", "macro_f1": 0.60},
    }
    assert bank.gap_is_established(strata, "macro_f1") is None
