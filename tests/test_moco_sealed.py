"""Sealed scoring contract: synthetic checks only; no sealed rows are touched by these tests."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from dermssl.release import moco_sealed as sealed

HAM = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
PAD = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")


def _probs(predicted, classes):
    index = [classes.index(p) for p in predicted]
    return np.eye(len(classes))[index]


def test_class_metrics_match_sklearn_and_report_recalls_and_confusion():
    truth = ["mel", "nv", "mel", "akiec", "nv", "vasc", "mel"]
    predicted = ["mel", "nv", "nv", "akiec", "nv", "mel", "mel"]
    metrics = sealed.class_metrics(truth, _probs(predicted, list(HAM)), HAM)
    assert metrics["macro_f1"] == pytest.approx(f1_score(truth, predicted, labels=list(HAM), average="macro", zero_division=0))
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, predicted))
    assert metrics["recall"]["mel"] == pytest.approx(2 / 3)
    assert metrics["recall"]["bcc"] is None
    assert metrics["confusion_matrix"][HAM.index("mel")][HAM.index("nv")] == 1
    assert metrics["n"] == 7


def test_class_metrics_reject_misaligned_or_nonfinite_probabilities():
    with pytest.raises(ValueError, match="aligned"):
        sealed.class_metrics(["mel"], np.ones((2, 7)), HAM)
    bad = np.ones((1, 7))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sealed.class_metrics(["mel"], bad, HAM)


def _metrics(ham_mel, pad_mel, pad_scc, ham_f1=.6, pad_f1=.6):
    ham = {"macro_f1": ham_f1, "recall": {"mel": ham_mel}}
    pad = {"macro_f1": pad_f1, "recall": {"MEL": pad_mel, "SCC": pad_scc}}
    return {"HAM": ham, "PAD": pad}


def test_clinical_score_is_the_mean_of_three_registered_recalls():
    m = _metrics(.5, .6, .9)
    assert sealed.clinical_score(m["HAM"], m["PAD"]) == pytest.approx((.5 + .6 + .9) / 3)


def test_release_rule_requires_clinical_gain_and_bounded_macro_f1_drop_per_dataset():
    reference = _metrics(.5, .5, .5)
    assert sealed.release_rule(_metrics(.6, .6, .6, .58, .58), reference)["passed"] is True
    assert sealed.release_rule(_metrics(.6, .6, .6, .54, .6), reference)["passed"] is False
    assert sealed.release_rule(_metrics(.5, .5, .5), reference)["passed"] is False


def test_sealed_receipt_is_claimed_once(tmp_path):
    receipt = tmp_path / "sealed.claim"
    sealed.claim_once(receipt)
    with pytest.raises(ValueError, match="already consumed"):
        sealed.claim_once(receipt)
