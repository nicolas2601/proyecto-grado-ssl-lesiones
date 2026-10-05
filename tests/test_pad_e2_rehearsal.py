"""E2 plumbing rehearsal (docs/33 errata): D0 and D1 on target_adapt, never touching target_test."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from test_pad_e2 import _manifest_rows, _split

from dermssl import pad_e2 as e2
from dermssl import pad_target_split as ts

N_RES = 30
GOLDEN_REAL_EVALUATE_SHA256 = "ca801b065adcb8479214b0cb56777af99fa4ff0c1bbad30e59ea6d15a5fb5739"


def test_the_real_mode_evaluate_output_is_byte_for_byte_what_it_was():
    rng = np.random.RandomState(11)
    labels = np.array([["bcc", "mel", "nv", "akiec"][(p + i) % 4] for p in range(30) for i in range(4)])
    groups = np.array([f"P{p}" for p in range(30) for i in range(4)])
    probas = {a: rng.dirichlet(np.ones(7), size=len(labels)) for a in e2.ARMS}
    doc = e2.evaluate(labels, groups, probas, n_resamples=N_RES)
    assert hashlib.sha256(json.dumps(doc, ensure_ascii=False).encode()).hexdigest() == GOLDEN_REAL_EVALUATE_SHA256


class TestEvaluateTwoArms:
    def _doc(self):
        rng = np.random.RandomState(3)
        labels = np.array([["bcc", "mel", "nv"][(p + i) % 3] for p in range(30) for i in range(3)])
        groups = np.array([f"P{p}" for p in range(30) for i in range(3)])
        probas = {a: rng.dirichlet(np.ones(7), size=len(labels)) for a in e2.REHEARSAL_ARMS}
        return e2.evaluate(labels, groups, probas, n_resamples=N_RES, arms=e2.REHEARSAL_ARMS,
                           contrasts=e2.REHEARSAL_CONTRASTS)

    def test_only_d0_and_d1_rows_and_the_d1_minus_d0_contrast(self):
        doc = self._doc()
        assert list(doc["arms"]) == ["D0", "D1"]
        assert {(r["arm"], r["control"]) for r in doc["paired"]} == {("D1", "D0")}

    def test_no_d3_minus_d2_contrast_or_prediction_exists(self):
        doc = self._doc()
        assert "primary_contrast" not in doc and "prediction_d3_minus_d2_met" not in doc


class TestOutputRule:
    @pytest.mark.parametrize("path", ["/w/pad_rehearsal/e2_rehearsal.json", "rehearsal_x/out_rehearsal.json"])
    def test_a_rehearsal_directory_and_name_pass(self, path):
        e2.check_rehearsal_output(path)

    @pytest.mark.parametrize("path", ["/w/pad_da/e2_rehearsal.json", "/w/pad_rehearsal/pad_e2_eval.json",
                                      "resultados/banco_completo/pad_rehearsal/e2_rehearsal.json",
                                      "/repo/resultados/pad_rehearsal/e2_rehearsal.json"])
    def test_anything_else_is_refused(self, path):
        with pytest.raises(e2.RehearsalOutputError):
            e2.check_rehearsal_output(path)


class TestRunRehearsal:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        rows = _manifest_rows(tmp_path)
        split = _split(tmp_path, rows)
        doc = json.loads(split.read_text())
        test_ids = set(doc["target_test"]["img_ids"])
        for img_id in test_ids:  # a tampered test image must not matter: it is never hashed
            (tmp_path / "images" / f"{img_id}.png").write_bytes(b"tampered")
        real_sha = ts.ids_sha256

        def sentinel_sha(ids):
            if set(ids) == test_ids:
                raise AssertionError("target_test list hashed")
            return real_sha(ids)

        def boom(*a, **k):
            raise AssertionError("target_test / lock touched")

        monkeypatch.setattr(ts, "ids_sha256", sentinel_sha)
        monkeypatch.setattr(ts, "read_target_test_ids", boom)
        monkeypatch.setattr(ts, "TargetTestLock", boom)
        return tmp_path, rows, split, doc, test_ids

    def _predict(self, seen):
        def predict(ids, paths):
            seen.append(list(ids))
            rng = np.random.RandomState(0)
            return {a: rng.dirichlet(np.ones(7), size=len(ids)) for a in e2.REHEARSAL_ARMS}
        return predict

    def _run(self, setup, seen):
        tmp, rows, split, *_ = setup
        out = tmp / "pad_rehearsal" / "e2_rehearsal.json"
        out.parent.mkdir()
        return e2.run_rehearsal(split, rows, tmp, self._predict(seen), out, n_resamples=N_RES), out

    def test_it_predicts_only_target_adapt_images(self, setup):
        seen = []
        self._run(setup, seen)
        assert sorted(seen[0]) == sorted(setup[3]["target_adapt"]["img_ids"])
        assert not set(seen[0]) & setup[4]

    def test_it_never_opens_the_test_list_hashes_it_or_creates_the_lock(self, setup):
        self._run(setup, [])
        assert not (setup[0] / ts.LOCK_NAME).exists()

    def test_the_document_is_marked_as_discarded(self, setup):
        doc, out = self._run(setup, [])
        assert doc["rehearsal"] is True and doc["split"] == "target_adapt" and doc["discard"] is True
        assert "NO se reporta" in doc["warning"] and "elegir" in doc["warning"]
        assert doc["target_adapt_sha256"] == setup[3]["sha256"]["target_adapt"]
        assert "target_test_sha256" not in doc and "n_target_test_images" not in doc
        assert "rehearsal" in out.with_suffix(".probas.npz").name

    def test_it_scores_d0_d1_and_the_paired_contrast_only(self, setup):
        doc, _ = self._run(setup, [])
        assert list(doc["arms"]) == ["D0", "D1"] and {(r["arm"], r["control"]) for r in doc["paired"]} == {("D1", "D0")}

    def test_a_changed_adapt_image_aborts_before_predicting(self, setup):
        seen = []
        tmp, _, _, doc, _ = setup
        (tmp / "images" / f"{doc['target_adapt']['img_ids'][0]}.png").write_bytes(b"changed")
        with pytest.raises(e2.ImageChangedError):
            self._run(setup, seen)
        assert not seen

    def test_a_forbidden_output_path_aborts_before_predicting(self, setup):
        tmp, rows, split, *_ = setup
        seen = []
        with pytest.raises(e2.RehearsalOutputError):
            e2.run_rehearsal(split, rows, tmp, self._predict(seen), tmp / "pad_e2_eval.json", n_resamples=N_RES)
        assert not seen
