"""docs/30 5: the FST estimator vs PAD's fitspatrick, on synthetic rows and images only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from dermssl import pad_ufes_guard as pg
from dermssl import pad_ufes_readings as pr
from scripts import pad_ufes_fst_validation as fv

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(pr, "BOOTSTRAP", {"n_resamples": 25, "rng_seed": 20260923})


def _row(actual: int, whole: int | None, masked: int | None, patient: str) -> dict:
    return {"actual_fst": actual, "patient_id": patient,
            "whole_image": {"predicted_bucket": whole}, "masked": {"predicted_bucket": masked}}


class TestBaselineAndKappa:
    def test_majority_takes_the_most_frequent_type(self):
        base = fv.majority_baseline([2, 2, 3, 2, 5])
        assert base["label"] == 2
        assert base["accuracy"] == pytest.approx(0.6)
        assert base["off_by_one_accuracy"] == pytest.approx(0.8)

    def test_a_tie_goes_to_the_lighter_type(self):
        assert fv.majority_baseline([3, 2, 3, 2])["label"] == 2

    def test_no_rows_no_baseline(self):
        assert fv.majority_baseline([])["accuracy"] is None

    def test_linear_kappa_perfect_and_reversed(self):
        assert fv.linear_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
        assert fv.linear_kappa([1, 2, 3, 4], [4, 3, 2, 1]) < 0

    def test_linear_weights_punish_far_misses_more(self):
        near = fv.linear_kappa([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 5])
        far = fv.linear_kappa([1, 2, 3, 4, 5, 6], [6, 6, 1, 1, 1, 1])
        assert near > far


class TestConditionReport:
    def test_unavailable_rows_are_counted_not_scored(self):
        rows = [_row(2, 2, None, "a"), _row(2, 3, 2, "b"), _row(3, None, 3, "c"), _row(2, 2, 2, "d")]
        report = fv.condition_report(rows, "whole_image")
        assert report["n_evaluable"] == 3 and report["n_unavailable"] == 1
        assert report["accuracy"] == pytest.approx(2 / 3)
        assert report["majority_baseline"]["accuracy"] == pytest.approx(1.0)
        assert report["beats_majority"] is False
        low, high = report["ci95"]["accuracy"]
        assert low <= high

    def test_an_estimator_better_than_the_majority_is_reported_as_such(self):
        rows = [_row(t, t, t, f"p{i}") for i, t in enumerate([1, 2, 3, 4, 5, 6, 2, 3])]
        report = fv.condition_report(rows, "masked")
        assert report["beats_majority"] is True
        assert report["linear_kappa"] == pytest.approx(1.0)


class TestVerdicts:
    def _reports(self, beats, kappas):
        return {c: {"beats_majority": b, "linear_kappa": k}
                for c, b, k in zip(fv.CONDITIONS, beats, kappas)}

    def test_prediction_met_when_neither_condition_beats_the_majority(self):
        v = fv.verdicts(self._reports((False, False), (0.05, 0.1)))
        assert v == {"FST-P1": "cumplida", "FST-P2": "cumplida"}

    def test_one_condition_beating_it_misses(self):
        v = fv.verdicts(self._reports((False, True), (0.05, 0.25)))
        assert v == {"FST-P1": "no cumplida", "FST-P2": "no cumplida"}

    def test_a_condition_without_rows_is_not_a_met_prediction(self):
        assert fv.verdicts(self._reports((None, False), (None, 0.1)))["FST-P1"] == "no cumplida"


def test_evaluate_rows_skips_missing_types_and_uses_the_docs20_estimator(tmp_path):
    rng = np.random.default_rng(0)
    image = np.clip(0.7 + 0.05 * rng.standard_normal((64, 64, 3)), 0, 1).astype(np.float32)
    pad = {"ids": ["a", "b", "c"], "fst": [2, None, 3], "groups": ["P1", "P1", "P2"],
           "paths": {"a": tmp_path / "a.png", "b": tmp_path / "b.png", "c": tmp_path / "c.png"}}
    rows, skipped = fv.evaluate_rows(pad, loader=lambda path: image)
    assert skipped == 1
    assert [r["img_id"] for r in rows] == ["a", "c"]
    assert {"whole_image", "masked", "actual_fst"} <= set(rows[0])


def test_it_refuses_when_the_fst_block_is_not_registered_first(tmp_path, monkeypatch):
    def refuse(repo, **kwargs):
        assert kwargs["start"] == fv.FST_BLOCK[0]
        raise pg.RegistrationError("el bloque registrado cambio")
    monkeypatch.setattr(pg, "check_registration", refuse)
    argv = ["--manifest", str(tmp_path / "none.json"), "--pad-root", str(tmp_path),
            "--out", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
        fv.main(argv)


def test_the_real_fst_registration_precedes_this_script():
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow != "false":
        pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
    got = pg.check_registration(REPO, commit=fv.FST_REGISTRATION_COMMIT, doc=pg.REGISTRATION_DOC,
                                script=fv.THIS_SCRIPT, start=fv.FST_BLOCK[0], end=fv.FST_BLOCK[1])
    assert got == fv.FST_REGISTRATION_COMMIT


def test_the_document_carries_the_fst_note():
    rows = [_row(3, 3, 3, "a"), _row(4, 2, 2, "b")]
    document = fv.build_document(rows, {"fst": [3, 4, None], "ids": ["a", "b", "c"],
                                        "manifest_sha256": "x"}, skipped=1, provenance={})
    assert "50 % tipo III (1/2)" in document["fst_note"]
