"""docs/28 5: the three registered predictions are decided by code, not by reading a table."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dermssl import final_eval as fe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_final_eval_table as bt  # noqa: E402


def _rec(delta, low, high):
    return {"point_delta": delta, "ci95_low": low, "ci95_high": high,
            "excludes_zero": bool(low > 0 or high < 0)}


def _paired(over=None):
    base = {
        ("plain", "public"): {"macro_f1": _rec(0.02, -0.01, 0.05), "ece": _rec(-0.1, -0.12, -0.08)},
        ("proposed", "plain"): {"macro_f1": _rec(0.0, -0.03, 0.03), "ece": _rec(-0.02, -0.03, -0.01)},
        ("proposed", "public"): {"macro_f1": _rec(0.02, -0.02, 0.06), "ece": _rec(-0.12, -0.14, -0.1)},
    }
    for pair in base.values():
        pair.setdefault("balanced_accuracy", _rec(0.0, -0.01, 0.01))
        pair.setdefault("auc", _rec(0.0, -0.01, 0.01))
    for pair, metrics in (over or {}).items():
        base[pair] = {**base[pair], **metrics}
    return base


class TestP1:
    def test_plain_not_worse_than_minus_001_is_met(self):
        p = fe.predictions(_paired())
        assert p["P1"]["verdict"] == "cumplida"

    def test_boundary_is_inclusive(self):
        p = fe.predictions(_paired({("plain", "public"): {"macro_f1": _rec(-0.01, -0.05, 0.03), "ece": _rec(0, -1, 1)}}))
        assert p["P1"]["verdict"] == "cumplida"

    def test_below_minus_001_fails(self):
        p = fe.predictions(_paired({("plain", "public"): {"macro_f1": _rec(-0.011, -0.05, 0.03), "ece": _rec(0, -1, 1)}}))
        assert p["P1"]["verdict"] == "no cumplida"

    def test_reports_improvement_only_when_ci_excludes_zero_upward(self):
        up = fe.predictions(_paired({("plain", "public"): {"macro_f1": _rec(0.08, 0.03, 0.13), "ece": _rec(0, -1, 1)}}))
        assert up["P1"]["improves_public"] is True
        assert fe.predictions(_paired())["P1"]["improves_public"] is False


class TestP2:
    def test_ece_better_with_ci_and_f1_not_worse_is_met(self):
        assert fe.predictions(_paired())["P2"]["verdict"] == "cumplida"

    def test_ece_ci_touching_zero_fails(self):
        p = fe.predictions(_paired({("proposed", "plain"): {"macro_f1": _rec(0.0, -0.03, 0.03), "ece": _rec(-0.01, -0.03, 0.001)}}))
        assert p["P2"]["verdict"] == "no cumplida"

    def test_f1_drop_fails_even_with_better_ece(self):
        p = fe.predictions(_paired({("proposed", "plain"): {"macro_f1": _rec(-0.02, -0.05, 0.0), "ece": _rec(-0.02, -0.03, -0.01)}}))
        assert p["P2"]["verdict"] == "no cumplida"


class TestP3:
    def test_f1_at_least_public_and_ece_advantage_is_met(self):
        assert fe.predictions(_paired())["P3"]["verdict"] == "cumplida"

    def test_negative_f1_point_fails(self):
        p = fe.predictions(_paired({("proposed", "public"): {"macro_f1": _rec(-0.001, -0.04, 0.03), "ece": _rec(-0.12, -0.14, -0.1)}}))
        assert p["P3"]["verdict"] == "no cumplida"

    def test_lost_ece_advantage_fails(self):
        p = fe.predictions(_paired({("proposed", "public"): {"macro_f1": _rec(0.02, -0.02, 0.06), "ece": _rec(-0.01, -0.03, 0.01)}}))
        assert p["P3"]["verdict"] == "no cumplida"


def test_missing_contrast_refuses():
    paired = _paired()
    del paired[("proposed", "plain")]
    with pytest.raises(KeyError):
        fe.predictions(paired)


def _document():
    results = {name: {m: 0.5 for m in fe.METRICS} | {"ci95": {m: [0.4, 0.6] for m in fe.METRICS}}
               for name in ("plain", "proposed", "public")}
    paired = _paired()
    return {
        "split": "holdout", "code_revision": "abc", "probe": "logreg", "n_eval": 10,
        "arms": {"plain": "A*", "proposed": "B*", "public": "DINOv2 público"},
        "results": results,
        "paired": [{"arm": a, "control": c, "metric": m, **r}
                   for (a, c), ms in paired.items() for m, r in ms.items()],
        "predictions": fe.predictions(paired),
    }


def test_table_is_generated_and_check_detects_drift(tmp_path):
    doc = tmp_path / "final_eval.json"
    out = tmp_path / "README.md"
    import json
    doc.write_text(json.dumps(_document()), encoding="utf-8")
    assert bt.main([str(doc), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "P1" in text and "cumplida" in text and "No editar a mano" in text
    assert bt.main([str(doc), "--out", str(out), "--check"]) == 0
    out.write_text(text + "edit\n", encoding="utf-8")
    assert bt.main([str(doc), "--out", str(out), "--check"]) == 1


def test_paired_rows_roundtrip():
    doc = _document()
    assert fe.paired_from_rows(doc["paired"]) == _paired()


class TestSecondaryReadings:
    """Reported beside the primary verdicts, never replacing them."""

    def test_p1_ni_needs_the_lower_bound_at_or_above_minus_001(self):
        ok = fe.predictions(_paired({("plain", "public"): {"macro_f1": _rec(0.02, -0.01, 0.05)}}))
        bad = fe.predictions(_paired({("plain", "public"): {"macro_f1": _rec(0.02, -0.011, 0.05)}}))
        assert ok["P1-NI"]["verdict"] == "cumplida" and ok["P1-NI"]["reading"] == "secundaria"
        assert bad["P1-NI"]["verdict"] == "no cumplida"
        assert bad["P1"]["verdict"] == "cumplida"  # the primary does not move

    def test_p3_ic_needs_the_lower_bound_strictly_above_zero(self):
        wins = fe.predictions(_paired({("proposed", "public"): {"macro_f1": _rec(0.05, 0.001, 0.1)}}))
        touches = fe.predictions(_paired({("proposed", "public"): {"macro_f1": _rec(0.05, 0.0, 0.1)}}))
        assert wins["P3-IC"]["verdict"] == "cumplida"
        assert touches["P3-IC"]["verdict"] == "no cumplida"

    def test_primary_ones_are_labelled_primary(self):
        p = fe.predictions(_paired())
        assert {p[k]["reading"] for k in ("P1", "P2", "P3")} == {"primaria"}


class TestValProtocol:
    def test_matching_fingerprints_pass(self, monkeypatch):
        from dermssl.feature_cache import fingerprint_ids
        monkeypatch.setattr(fe, "C1_TRAIN_FINGERPRINT", fingerprint_ids(["a", "b"]))
        monkeypatch.setattr(fe, "C1_VAL_FINGERPRINT", fingerprint_ids(["c"]))
        fe.check_val_protocol(["a", "b"], ["c"])

    def test_a_different_fit_set_refuses(self):
        with pytest.raises(ValueError, match="train"):
            fe.check_val_protocol(["a", "b"], ["c"])

    def test_the_published_fingerprints_are_pinned(self):
        assert fe.C1_TRAIN_FINGERPRINT == "d347edd52d0d7648"
        assert fe.C1_VAL_FINGERPRINT == "6ff4c17f04a9eae2"


class TestRunConsistency:
    def _run(self, rev="f1c2053", sha="95d2"):
        return {"code_revision": rev, "exclude_ids_sha256": sha}

    def test_same_revision_and_exclude_pass(self):
        fe.check_runs_consistent({"plain": self._run(), "proposed": self._run()})

    def test_different_revision_refuses(self):
        with pytest.raises(ValueError, match="code_revision"):
            fe.check_runs_consistent({"plain": self._run(), "proposed": self._run(rev="x")})

    def test_different_exclude_refuses(self):
        with pytest.raises(ValueError, match="exclude"):
            fe.check_runs_consistent({"plain": self._run(), "proposed": self._run(sha="y")})

    @pytest.mark.parametrize("missing", [None, ""])
    def test_a_missing_revision_is_not_equal_to_another_missing_one(self, missing):
        with pytest.raises(ValueError, match="code_revision"):
            fe.check_runs_consistent({"plain": self._run(rev=missing), "proposed": self._run(rev=missing)})


class TestSameSeed:
    def test_one_seed_passes(self):
        fe.check_same_seed({"plain": {"seed": 44}, "proposed": {"seed": 44}})

    def test_finals_of_different_seeds_refuse(self):
        with pytest.raises(ValueError, match="semilla"):
            fe.check_same_seed({"plain": {"seed": 44}, "proposed": {"seed": 43}})

    def test_a_run_without_seed_refuses(self):
        with pytest.raises(ValueError, match="semilla"):
            fe.check_same_seed({"plain": {"seed": 44}, "proposed": {}})


def test_val_guard_refuses_finals_of_two_seeds(monkeypatch):
    import c1_final_eval as ev
    from dermssl.feature_cache import fingerprint_ids

    monkeypatch.setattr(fe, "C1_TRAIN_FINGERPRINT", fingerprint_ids(["a"]))
    monkeypatch.setattr(fe, "C1_VAL_FINGERPRINT", fingerprint_ids(["b"]))
    runs = {"plain": {"code_revision": "f1", "exclude_ids_sha256": "e", "seed": 44},
            "proposed": {"code_revision": "f1", "exclude_ids_sha256": "e", "seed": 43}}
    with pytest.raises(ValueError, match="semilla"):
        ev.guard_val({"train_ids": ["a"], "val_ids": ["b"]}, runs, "e")


def test_val_guard_refuses_a_final_trained_with_another_exclusion(monkeypatch):
    import c1_final_eval as ev
    from dermssl.feature_cache import fingerprint_ids

    monkeypatch.setattr(fe, "C1_TRAIN_FINGERPRINT", fingerprint_ids(["a"]))
    monkeypatch.setattr(fe, "C1_VAL_FINGERPRINT", fingerprint_ids(["b"]))
    runs = {r: {"code_revision": "f1", "exclude_ids_sha256": "old", "seed": 42} for r in ("plain", "proposed")}
    with pytest.raises(ValueError, match="exclude"):
        ev.guard_val({"train_ids": ["a"], "val_ids": ["b"]}, runs, "new")
    ev.guard_val({"train_ids": ["a"], "val_ids": ["b"]}, runs, "old")


class TestEceBasicInterval:
    """Secondary ECE interval: the percentile one is biased for ECE (docs/28 5)."""

    def test_basic_interval_reflects_the_percentile_bounds_around_the_point(self):
        assert fe.basic_interval(0.0135, 0.0138, 0.0336) == pytest.approx((-0.0066, 0.0132))

    def test_document_gets_arms_and_contrasts_labelled_secondary(self):
        doc = _document()
        doc["results"]["plain"]["ece"] = 0.0135
        doc["results"]["plain"]["ci95"]["ece"] = [0.0138, 0.0336]
        extra = fe.ece_basic_ci(doc)
        assert extra["reading"] == "secundaria" and "sesgo" in extra["note"]
        plain = extra["arms"]["plain"]
        assert plain["ci95_low"] == pytest.approx(-0.0066) and plain["ci95_high"] == pytest.approx(0.0132)
        assert plain["point_outside_percentile"] is True
        assert extra["arms"]["public"]["point_outside_percentile"] is False
        c = extra["contrasts"]["proposed-plain"]
        # _paired(): proposed-plain ECE delta -0.02 in [-0.03, -0.01] -> basic [-0.03, -0.01]
        assert (c["ci95_low"], c["ci95_high"]) == pytest.approx((-0.03, -0.01))
        assert c["excludes_zero"] is True

    def test_p2_under_basic_is_reported_but_the_primary_verdict_does_not_move(self):
        doc = _document()
        doc["paired"] = [r if not (r["arm"] == "proposed" and r["control"] == "plain" and r["metric"] == "ece")
                         else {**r, **_rec(0.019, 0.0003, 0.027)} for r in doc["paired"]]
        doc["predictions"] = fe.predictions(fe.paired_from_rows(doc["paired"]))
        extra = fe.ece_basic_ci(doc)
        assert extra["P2_with_basic"] == "no cumplida"
        assert doc["predictions"]["P2"]["verdict"] == "no cumplida"

    def test_table_shows_the_secondary_section_only_when_present(self):
        doc = _document()
        assert "IC basic" not in bt.render(doc)
        doc["secondary_ece_basic_ci"] = fe.ece_basic_ci(doc)
        text = bt.render(doc)
        assert "IC basic" in text and "secundaria" in text


def test_augment_adds_the_secondary_block_once_and_keeps_everything_else(tmp_path):
    import json

    import add_ece_basic_ci as aug

    path = tmp_path / "final_eval.json"
    doc = _document()
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert aug.main([str(path)]) == 0
    new = json.loads(path.read_text(encoding="utf-8"))
    assert new["secondary_ece_basic_ci"]["reading"] == "secundaria"
    assert {k: v for k, v in new.items() if k != "secondary_ece_basic_ci"} == json.loads(json.dumps(doc))
    assert aug.main([str(path)]) == 1  # never overwrites
