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
