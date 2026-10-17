"""docs/28 4.1: both seeds are reported, per seed and as mean with the s43-s42 difference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_seed_summary as bs  # noqa: E402


def _doc(f1_plain, f1_prop, f1_ic_low, p3_ic):
    results = {role: {"macro_f1": f, "balanced_accuracy": 0.6, "auc": 0.95, "ece": 0.02, "ci95": {}}
               for role, f in (("plain", f1_plain), ("proposed", f1_prop), ("public", 0.6))}
    paired = [{"arm": a, "control": c, "metric": m, "point_delta": 0.0, "ci95_low": -0.1, "ci95_high": 0.1,
               "excludes_zero": False} for a, c in bs.fe.CONTRASTS for m in bs.fe.METRICS]
    paired.append({"arm": "proposed", "control": "public", "metric": "macro_f1", "point_delta": f1_prop - 0.6,
                   "ci95_low": f1_ic_low, "ci95_high": 0.1, "excludes_zero": f1_ic_low > 0})
    predictions = {name: {"verdict": "cumplida"} for name in ("P1", "P2", "P3", "P1-NI")}
    predictions["P3-IC"] = {"verdict": p3_ic}
    return {"arms": {"plain": "A*", "proposed": "B*", "public": "DINOv2 público"},
            "results": results, "paired": paired, "predictions": predictions}


def test_mean_and_difference_per_arm_and_metric():
    rows = bs.arm_rows({42: _doc(0.64, 0.66, 0.01, "cumplida"), 43: _doc(0.68, 0.66, 0.01, "cumplida")})
    plain = rows["plain"]["macro_f1"]
    assert plain["s42"] == pytest.approx(0.64) and plain["s43"] == pytest.approx(0.68)
    assert plain["mean"] == pytest.approx(0.66) and plain["diff"] == pytest.approx(0.04)


def test_p3_ic_is_robust_only_when_both_seeds_meet_it():
    both = bs.robustness({42: _doc(0.6, 0.66, 0.01, "cumplida"), 43: _doc(0.6, 0.66, 0.01, "cumplida")})
    split = bs.robustness({42: _doc(0.6, 0.66, 0.01, "cumplida"), 43: _doc(0.6, 0.66, -0.01, "no cumplida")})
    assert both == "robusto a la semilla"
    assert split == "no robusto a la semilla"


def test_three_seeds_give_mean_and_range_not_a_difference():
    docs = {42: _doc(0.64, 0.66, 0.01, "cumplida"), 43: _doc(0.68, 0.66, 0.01, "cumplida"),
            44: _doc(0.62, 0.66, 0.01, "cumplida")}
    plain = bs.arm_rows(docs)["plain"]["macro_f1"]
    assert plain["s44"] == pytest.approx(0.62) and plain["mean"] == pytest.approx((0.64 + 0.68 + 0.62) / 3)
    assert plain["range"] == pytest.approx(0.06) and "diff" not in plain
    text = bs.render(docs)
    assert "# v7: las tres semillas" in text and "Rango (máx − mín)" in text
    assert "| repetición 42 | repetición 43 | repetición 44 |" in text and "(semilla 44)" in text
    assert "Predicción de objetivo (intervalo de confianza) con las tres semillas: robusto a la semilla." in text


def test_one_seed_that_misses_p3_ic_breaks_robustness_across_three():
    docs = {42: _doc(0.6, 0.66, 0.01, "cumplida"), 43: _doc(0.6, 0.66, 0.01, "cumplida"),
            44: _doc(0.6, 0.66, -0.01, "no cumplida")}
    assert bs.robustness(docs) == "no robusto a la semilla"


def test_render_lists_verdicts_per_seed_and_marks_generated():
    text = bs.render({42: _doc(0.64, 0.66, 0.01, "cumplida"), 43: _doc(0.68, 0.66, 0.01, "cumplida")})
    assert "No editar a mano" in text and "Predicción de objetivo (intervalo de confianza)" in text and "robusto a la semilla" in text
    assert "repetición 43 − repetición 42" in text
