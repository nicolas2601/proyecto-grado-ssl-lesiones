"""docs/30 7: the descriptive A71/s44 PAD-UFES addendum, on synthetic probabilities.

The readings must be the first pass's (docs/30 2) number for number, with no
verdict and every contrast labelled only "excludes 0" or "indistinguishable".
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_addendum as pa
from dermssl import pad_ufes_readings as pr


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(pr, "BOOTSTRAP", {"n_resamples": 25, "rng_seed": 20260923})


def _pad(n: int = 60) -> dict:
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    return {"ham_labels": np.array([cycle[i % 5] for i in range(n)]),
            "groups": np.array([f"P{i // 3}" for i in range(n)]),
            "fst_bins": np.array([pu.FST_BINS[i % 4] for i in range(n)]),
            "fst": [[1, 2, 3, 4, None][i % 5] for i in range(n)]}


def _probas(roles, n: int = 60, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {role: (lambda raw: raw / raw.sum(axis=1, keepdims=True))(rng.random((n, 7))) for role in roles}


class TestRolesAndContrasts:
    def test_the_models_of_the_addendum(self):
        assert pa.roles() == ["public", "plain_s44", "proposed_s44", "A71", "public_A71"]

    def test_a71_against_its_same_protocol_public_and_s44_as_in_section_2(self):
        assert pa.contrast_pairs() == [("A71", "public_A71"), ("plain_s44", "public"),
                                       ("proposed_s44", "public"), ("proposed_s44", "plain_s44")]

    def test_the_frozen_v71_protocols(self):
        assert pa.A71_CONFIG == "A_soup_a1.0@224|cls_attn|la0.5|tta"
        assert pa.PUBLIC_A71_CONFIG == "public@224|cls_attn|la0.5|tta"
        assert pa.A71_CELL == ("cls_attn", "la0.5", True)


class TestSameNumbersAsTheFirstPass:
    """Same probabilities, same roles: the addendum reading is the first pass's, minus the verdicts."""

    def _first_pass_rows(self, rows):
        return [{k: v for k, v in row.items() if k != "direction"} for row in rows]

    @pytest.mark.parametrize("reading", list(pu.READINGS))
    def test_readings(self, reading):
        pad, probas = _pad(), _probas(pr.roles())
        pairs = pr.contrast_pairs()
        first = pr.evaluate_reading(reading, pad["ham_labels"], pad["groups"], probas, pairs)
        ours = pa.evaluate_reading(reading, pad["ham_labels"], pad["groups"], probas, pairs)
        assert ours["arms"] == first["arms"]
        assert ours["outside_rate"] == first["outside_rate"]
        assert (ours["n_images"], ours["n_patients"], ours["classes"]) == \
            (first["n_images"], first["n_patients"], first["classes"])
        assert [{k: v for k, v in r.items() if k != "reading"} for r in ours["paired"]] == \
            self._first_pass_rows(first["paired"])

    def test_fst_strata(self):
        pad, probas = _pad(), _probas(pr.roles())
        pairs = pr.contrast_pairs()
        first = pr.fst_strata(pad["ham_labels"], pad["groups"], pad["fst_bins"], probas, pairs)
        ours = pa.fst_strata(pad["ham_labels"], pad["groups"], pad["fst_bins"], probas, pairs)
        assert set(ours) == set(first)
        for name, entry in first.items():
            assert ours[name]["arms"] == entry["arms"] and ours[name]["power"] == entry["power"]
            assert [{k: v for k, v in r.items() if k != "reading"} for r in ours[name]["paired"]] == \
                self._first_pass_rows(entry["paired"])


class TestDescriptiveOnly:
    def test_each_contrast_is_labelled_by_its_interval_only(self):
        rows = [{"excludes_zero": True, "direction": "arm_better"},
                {"excludes_zero": False, "direction": "indistinguishable"}]
        assert [r["reading"] for r in pa.describe(rows)] == [pa.EXCLUDES_ZERO, pa.INDISTINGUISHABLE]
        assert all("direction" not in r for r in pa.describe(rows))

    def test_the_document_has_no_verdict_anywhere(self):
        document = pa.build_document(_probas(pa.roles()), _pad(), {"code_revision": "abc"})
        text = json.dumps(document, ensure_ascii=False)
        for word in ("verdict", "PAD-P", "mejora", "gana", "direction", "arm_better", "control_better"):
            assert word not in text
        labels = {r["reading"] for reading in document["readings"].values() for r in reading["paired"]}
        assert labels <= {pa.EXCLUDES_ZERO, pa.INDISTINGUISHABLE}

    def test_the_document_has_every_reading_and_the_strata_with_the_note(self):
        pad = _pad()
        document = pa.build_document(_probas(pa.roles()), pad, {"code_revision": "abc"})
        assert set(document["readings"]) == set(pu.READINGS)
        assert set(document["fst_strata"]["bins"]) == set(pu.FST_BINS)
        assert document["fst_strata"]["note"] == pu.fst_note(pu.fst_composition(pad["fst"]))
        assert document["bootstrap"] == {**pr.BOOTSTRAP, "paired": True, "resampling_unit": "patient_id"}
        assert document["contrasts"] == [list(p) for p in pa.contrast_pairs()]
        assert document["code_revision"] == "abc"

    @pytest.mark.parametrize("bad", [{"verdicts": {}}, {"x": [{"direction": "arm_better"}]},
                                     {"x": {"y": "B* mejora"}}, {"PAD-P1": "cumplida"}])
    def test_the_guard_refuses_a_verdict_smuggled_in(self, bad):
        with pytest.raises(ValueError, match="descriptiv"):
            pa.assert_descriptive(bad)
