"""C1 thread-count reproduction diagnostic: the pure parts, no GPU, no pod."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dermssl import c1_thread_repro as ctr
from dermssl import probes

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "resultados/clasificacion/c1_v7/diagnostico_hilos"


def _toy(seed: int = 0, n: int = 120, dim: int = 16, n_classes: int = 3):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % n_classes
    x = rng.normal(size=(n, dim)).astype(np.float32) + y[:, None] * 0.4
    return x, y, rng.normal(size=(40, dim)).astype(np.float32)


def _fit(x, y, v):
    return probes.run_probe("logreg", x, y, v, 3)


class TestPinnedFit:
    def test_a_probe_refit_with_the_same_thread_count_is_bit_equal(self):
        x, y, v = _toy()
        first, _ = ctr.fit_with_threads(_fit, x, y, v, threads=2)
        second, _ = ctr.fit_with_threads(_fit, x, y, v, threads=2)
        assert np.array_equal(first, second) and first.dtype == second.dtype

    def test_the_resolved_pools_report_the_requested_cap(self):
        x, y, v = _toy()
        _, pools = ctr.fit_with_threads(_fit, x, y, v, threads=1)
        assert pools and all(p["num_threads"] == 1 for p in pools)

    def test_default_applies_no_cap_and_still_reports_the_pools(self):
        x, y, v = _toy()
        proba, pools = ctr.fit_with_threads(_fit, x, y, v, threads=None)
        assert proba.shape == (40, 3) and pools and all(p["num_threads"] >= 1 for p in pools)


class TestCompare:
    def test_identical_probas_are_bit_equal_with_zero_deltas(self):
        p = np.full((4, 3), 1 / 3)
        out = ctr.compare_proba(p, p.copy(), refit_macro_f1=0.5, published_macro_f1=0.5)
        assert out == {"bit_equal": True, "max_abs_delta": 0.0, "argmax_flips": 0, "abs_delta_macro_f1": 0.0,
                       "refit_macro_f1": 0.5, "published_macro_f1": 0.5}

    def test_one_ulp_is_not_bit_equal_and_its_size_is_reported(self):
        p = np.array([[0.2, 0.8], [0.6, 0.4]])
        q = p.copy()
        q[0, 0] = np.nextafter(q[0, 0], 1.0)
        out = ctr.compare_proba(q, p, refit_macro_f1=0.5, published_macro_f1=0.5)
        assert not out["bit_equal"] and out["max_abs_delta"] == pytest.approx(abs(q[0, 0] - p[0, 0]))
        assert out["argmax_flips"] == 0

    def test_argmax_flips_and_macro_f1_delta_are_counted(self):
        p = np.array([[0.6, 0.4], [0.6, 0.4], [0.1, 0.9]])
        q = np.array([[0.4, 0.6], [0.6, 0.4], [0.9, 0.1]])
        out = ctr.compare_proba(q, p, refit_macro_f1=0.40, published_macro_f1=0.45)
        assert out["argmax_flips"] == 2 and out["abs_delta_macro_f1"] == pytest.approx(0.05)
        assert out["max_abs_delta"] == pytest.approx(0.8)

    def test_a_dtype_change_is_not_bit_equal_even_when_values_match(self):
        p = np.array([[0.5, 0.5]])
        assert not ctr.compare_proba(p.astype(np.float32), p, refit_macro_f1=0, published_macro_f1=0)["bit_equal"]

    def test_a_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError):
            ctr.compare_proba(np.zeros((2, 3)), np.zeros((3, 3)), refit_macro_f1=0, published_macro_f1=0)


class TestSettings:
    def test_labels_map_to_thread_caps(self):
        got = ctr.parse_settings(["default", "recorded", "4", "1"], recorded=8)
        assert got == [("default", None), ("recorded", 8), ("4", 4), ("1", 1)]

    def test_recorded_is_dropped_when_nothing_was_recorded(self):
        assert ctr.parse_settings(["default", "recorded", "4"], recorded=None) == [("default", None), ("4", 4)]

    @pytest.mark.parametrize("bad", ["0", "-2", "many"])
    def test_invalid_counts_are_refused(self, bad):
        with pytest.raises(ValueError):
            ctr.parse_settings([bad], recorded=None)


def _fit_row(setting, threads, *, bit_equal, delta=0.0, flips=0, df1=0.0):
    return {"setting": setting, "threads": threads, "resolved_threads": [48 if threads is None else threads],
            "seconds": 1.0, "bit_equal": bit_equal, "max_abs_delta": delta, "argmax_flips": flips,
            "abs_delta_macro_f1": df1, "refit_macro_f1": 0.5, "published_macro_f1": 0.5 + df1}


def _cell(name, fits, *, passed=True):
    return {"name": name, "gate": {"passed": passed, "problems": [] if passed else ["x"]},
            "repeat_bit_equal": True, "fits": fits}


def _cells(four_equal=(True, True)):
    return [
        _cell(f"c{i}", [_fit_row("default", None, bit_equal=False, delta=1e-3, flips=12, df1=0.004),
                        _fit_row("4", 4, bit_equal=eq, delta=0.0 if eq else 2e-4, flips=0 if eq else 3),
                        _fit_row("8", 8, bit_equal=False, delta=5e-3, flips=40, df1=0.01)])
        for i, eq in enumerate(four_equal)
    ]


class TestSummary:
    def test_a_count_bit_equal_in_every_cell_is_the_one_to_pin(self):
        s = ctr.summarize(_cells())
        assert s["reproducing_settings"] == ["4"] and s["pin_to"] == 4 and s["verdict"] == "pin"
        assert s["per_setting"]["8"] == {"threads": 8, "n_cells": 2, "n_bit_equal": 0, "max_abs_delta": 5e-3,
                                         "max_argmax_flips": 40, "max_abs_delta_macro_f1": 0.01}

    def test_bit_equal_in_only_some_cells_does_not_reproduce(self):
        s = ctr.summarize(_cells(four_equal=(True, False)))
        assert s["reproducing_settings"] == [] and s["pin_to"] is None and s["verdict"] == "declare"
        assert s["per_setting"]["4"]["n_bit_equal"] == 1

    def test_the_smallest_reproducing_explicit_count_is_pinned_and_default_never_is(self):
        cells = [_cell("c", [_fit_row("default", None, bit_equal=True), _fit_row("48", 48, bit_equal=True),
                             _fit_row("12", 12, bit_equal=True)])]
        s = ctr.summarize(cells)
        assert s["reproducing_settings"] == ["default", "48", "12"] and s["pin_to"] == 12

    def test_only_default_reproducing_gives_no_explicit_pin(self):
        s = ctr.summarize([_cell("c", [_fit_row("default", None, bit_equal=True), _fit_row("4", 4, bit_equal=False)])])
        assert s["pin_to"] is None and s["verdict"] == "declare"

    def test_a_failed_gate_makes_the_verdict_inconclusive(self):
        cells = _cells()
        cells[1]["gate"] = {"passed": False, "problems": ["val features differ"]}
        cells[1]["fits"] = []
        s = ctr.summarize(cells)
        assert s["verdict"] == "inconclusive" and not s["all_gates_passed"]


def _doc(cells):
    return {"code_revision": "abc", "settings": [{"label": "default", "threads": None}, {"label": "4", "threads": 4},
                                                  {"label": "8", "threads": 8}],
            "recorded_threads": {"value": None, "source": "no environment sidecar next to the grid"},
            "cells": cells, "summary": ctr.summarize(cells)}


class TestReadme:
    def test_it_states_the_decision_rule_and_the_verdict(self):
        text = ctr.render_readme(_doc(_cells()))
        assert "do not edit" in text and "Decision rule" in text
        assert "pinned to that explicit N" in text and "measured |delta|" in text
        assert "**pin C1 to 4 BLAS threads**" in text
        assert "| c0 | 8 | 8 | no | 5.000e-03 | 40 | 1.000e-02 |" in text

    def test_the_declare_verdict_is_written_out(self):
        text = ctr.render_readme(_doc(_cells(four_equal=(True, False))))
        assert "no thread count reproduces" in text


def test_committed_readme_matches_its_json():
    path = OUT_DIR / ctr.OUT_JSON
    if not path.exists():
        pytest.skip("the pod run has not been fetched yet")
    import json

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert (OUT_DIR / "README.md").read_text(encoding="utf-8") == ctr.render_readme(doc)
