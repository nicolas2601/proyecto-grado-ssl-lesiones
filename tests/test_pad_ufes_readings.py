"""docs/30 2.3-3: the PAD-UFES readings, strata and verdicts, on synthetic probabilities."""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_readings as pr


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(pr, "BOOTSTRAP", {"n_resamples": 30, "rng_seed": 20260923})


def _proba7(n: int, seed: int, truth=None, skill: float = 0.0) -> np.ndarray:
    raw = np.random.default_rng(seed).random((n, 7))
    if truth is not None:
        raw[np.arange(n), [pr.HAM_CLASSES.index(t) for t in truth]] += skill
    return raw / raw.sum(axis=1, keepdims=True)


def _labels(n: int) -> np.ndarray:
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    return np.array([cycle[i % len(cycle)] for i in range(n)])


class TestRestriction:
    def test_the_probe_columns_are_hams_sorted_classes(self):
        assert list(pr.HAM_CLASSES) == sorted(pr.HAM_CLASSES)
        assert set(pu.PAD_TO_HAM.values()) <= set(pr.HAM_CLASSES)

    def test_restrict_keeps_the_reading_columns_and_renormalizes(self):
        proba7 = np.array([[0.1, 0.2, 0.05, 0.05, 0.1, 0.4, 0.1]])
        out = pr.restrict(proba7, "primary")  # bcc, mel, nv
        assert out.sum() == pytest.approx(1.0)
        assert out[0] == pytest.approx(np.array([0.2, 0.1, 0.4]) / 0.7)

    def test_a_row_with_no_mass_in_the_reading_is_refused(self):
        proba7 = np.array([[1.0, 0, 0, 0, 0, 0, 0]])
        with pytest.raises(ValueError, match="sin masa"):
            pr.restrict(proba7, "primary")

    def test_targets_drop_classes_outside_the_reading(self):
        mask, y = pr.targets(["bcc", "akiec", "nv", "bkl", "mel"], "primary")
        assert mask.tolist() == [True, False, True, False, True]
        assert y.tolist() == [0, 2, 1]

    def test_outside_rate_counts_seven_way_argmax_off_the_reading(self):
        proba7 = np.eye(7)[[0, 1, 4, 3]] * 0.9 + 0.1 / 7
        assert pr.outside_rate(proba7, "primary") == pytest.approx(0.5)


class TestContrasts:
    def test_roles_and_pairs_per_seed(self):
        assert pr.roles() == ["public", "plain_s42", "proposed_s42", "plain_s43", "proposed_s43"]
        assert ("proposed_s43", "plain_s43") in pr.contrast_pairs()
        assert len(pr.contrast_pairs()) == 6

    def test_evaluate_reading_uses_patient_groups_and_reports_everything(self):
        n = 60
        labels = _labels(n)
        groups = np.arange(n) // 3
        probas = {r: _proba7(n, i, labels, skill=0.5 * i) for i, r in enumerate(pr.roles())}
        result = pr.evaluate_reading("secondary", labels, groups, probas, pr.contrast_pairs())
        assert result["classes"] == ["akiec", "bcc", "mel", "nv"]
        assert result["n_images"] == 48 and result["n_patients"] == len(set(groups[labels != "bkl"]))
        assert len(result["paired"]) == 6 * 4
        assert set(result["verdicts"]) >= {"PAD-P1", "PAD-P2", "PAD-P3", "PAD-P4", "mejora_robusta"}
        low, high = result["arms"]["public"]["ci95"]["macro_f1"]
        assert low <= result["arms"]["public"]["macro_f1"] <= high

    def test_the_exploratory_reading_has_no_verdict(self):
        n = 50
        labels = _labels(n)
        probas = {r: _proba7(n, i) for i, r in enumerate(pr.roles())}
        result = pr.evaluate_reading("exploratory", labels, np.arange(n), probas, pr.contrast_pairs())
        assert "verdicts" not in result


def _row(arm, control, metric, delta, excludes=False, direction="indistinguishable"):
    return {"arm": arm, "control": control, "metric": metric, "point_delta": delta,
            "ci95_low": delta - 0.1, "ci95_high": delta + 0.1, "excludes_zero": excludes,
            "direction": direction}


def _rows(f1_prop_pub=(-0.01, -0.02), f1_plain_pub=(-0.01, 0.01), ece_prop_pub=(-0.05, -0.02),
          prop_plain_excludes=(False, False)):
    rows = []
    for i, seed in enumerate(pr.SEEDS):
        plain, prop = pr.role("plain", seed), pr.role("proposed", seed)
        for metric in ("macro_f1", "ece", "balanced_accuracy", "auc"):
            rows.append(_row(prop, "public", metric,
                             {"macro_f1": f1_prop_pub[i], "ece": ece_prop_pub[i]}.get(metric, 0.0)))
            rows.append(_row(plain, "public", metric, f1_plain_pub[i] if metric == "macro_f1" else 0.0))
            rows.append(_row(prop, plain, metric, 0.0, excludes=metric == "macro_f1" and prop_plain_excludes[i]))
    return rows


class TestVerdicts:
    def test_the_registered_prediction_met(self):
        v = pr.verdicts(_rows())
        assert v["PAD-P1"] == "cumplida"
        assert v["PAD-P3"] == "cumplida"
        assert v["PAD-P4"] == "cumplida"

    def test_one_seed_against_is_enough_to_miss(self):
        assert pr.verdicts(_rows(f1_plain_pub=(-0.01, 0.01)))["PAD-P2"] == "no cumplida"
        assert pr.verdicts(_rows(f1_prop_pub=(0.001, -0.02)))["PAD-P1"] == "no cumplida"
        assert pr.verdicts(_rows(ece_prop_pub=(-0.05, 0.0)))["PAD-P3"] == "no cumplida"
        assert pr.verdicts(_rows(prop_plain_excludes=(False, True)))["PAD-P4"] == "no cumplida"

    def test_mejora_needs_every_seed(self):
        rows = [_row("proposed_s42", "public", "macro_f1", 0.05, True, "arm_better"),
                _row("proposed_s43", "public", "macro_f1", 0.04, True, "arm_better"),
                _row("plain_s42", "public", "macro_f1", 0.05, True, "arm_better"),
                _row("plain_s43", "public", "macro_f1", 0.01, False)]
        wins = pr.improvements(rows)
        assert wins["proposed-public/macro_f1"] is True
        assert wins["plain-public/macro_f1"] is False


class TestFstStrata:
    def test_bins_below_the_floor_are_flagged_without_power(self, monkeypatch):
        monkeypatch.setattr(pr, "MIN_BIN_N", 30)
        n = 150
        labels = np.array(["bcc", "mel", "nv"] * 50)
        bins = np.array(["II"] * 100 + ["I"] * 20 + [pu.FST_MISSING] * 30)
        probas = {r: _proba7(n, i) for i, r in enumerate(pr.roles())}
        strata = pr.fst_strata(labels, np.arange(n) // 2, bins, probas, pr.contrast_pairs())
        assert strata["II"]["power"] == "ok" and strata["II"]["n_images"] == 100
        assert strata["I"]["power"] == pr.NO_POWER and strata["I"]["n_images"] == 20
        assert strata[pu.FST_MISSING]["n_images"] == 30
        assert strata["III-VI"] == {"n_images": 0, "power": pr.NO_POWER}
        assert "verdicts" not in strata["II"]

    def test_the_registered_floor_is_one_hundred(self):
        assert pr.MIN_BIN_N == 100
