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
