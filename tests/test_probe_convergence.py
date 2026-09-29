"""Diagnostic: does the v7.1 logreg probe converge within max_iter? (synthetic data only)."""

from pathlib import Path

import numpy as np
import pytest

from dermssl import probe_convergence as pc
from dermssl import v71_grid as vg

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "resultados/clasificacion/t4/diagnostico_convergencia"


def _hard_problem(seed=0, n=240, dim=20, k=4):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    x = rng.normal(size=(n, dim)) + 0.3 * y[:, None]
    return x.astype(np.float32), y


def _separable_problem(n=60):
    y = np.arange(n) % 3
    x = np.zeros((n, 3), dtype=np.float32)
    x[np.arange(n), y] = 1.0
    return x + 0.01 * np.arange(n)[:, None] / n, y


def test_a_single_iteration_hits_the_cap_and_warns():
    x, y = _hard_problem()
    _, record = pc.fit_views_recorded(x, y, x[None], 4, None, max_iter=1)
    assert record["hit_max_iter"] is True
    assert record["convergence_warning"] is True
    assert record["max_iter"] == 1 and max(record["n_iter"]) == 1


def test_a_trivially_separable_problem_converges_without_warning():
    x, y = _separable_problem()
    _, record = pc.fit_views_recorded(x, y, x[None], 3, None, max_iter=2000)
    assert record["hit_max_iter"] is False
    assert record["convergence_warning"] is False
    assert max(record["n_iter"]) < 2000


def test_the_default_max_iter_is_the_published_probe_one():
    assert pc.MAX_ITER == 2000


@pytest.mark.parametrize("weight", [None, "balanced"])
def test_the_instrumented_fit_returns_the_published_code_path_bit_for_bit(weight):
    x, y = _hard_problem(seed=1)
    views = np.stack([x, x[::-1]])
    ours, _ = pc.fit_views_recorded(x, y, views, 5, weight)
    assert np.array_equal(ours, vg.fit_predict_views(x, y, views, 5, weight))


def _parts(seed, n, n_views):
    rng = np.random.default_rng(seed)
    return {p: rng.normal(size=(n_views, n, 4)).astype(np.float32) for p in ("cls", "mean")}


def test_unit_cells_equal_the_published_unit_probas_and_every_fit_is_recorded():
    y = np.arange(80) % 3
    fit, hold = _parts(2, 80, 1), _parts(3, 30, 2)
    fit["cls"][0] += y[:, None]
    cells, records = pc.unit_fits("u@224", fit, y, hold, 3, features=("cls", "cls_mean"), n_jobs=1)
    ref = vg.unit_probas(fit, y, hold, 3, features=("cls", "cls_mean"))
    assert cells.keys() == ref.keys() and all(np.array_equal(cells[k], ref[k]) for k in ref)
    assert [(r["unit"], r["feature"], r["weighting"]) for r in records] == [
        ("u@224", f, w) for f in ("cls", "cls_mean") for w in ("none", "balanced")]
    assert all(r["n_views_scored"] == 2 for r in records)


def test_probas_equal_only_when_every_cell_is_bit_equal():
    a = {("cls", "none", False): np.full((3, 2), 0.5)}
    same = {"u@224|cls|none|id": np.full((3, 2), 0.5, dtype=np.float32)}
    report = pc.compare_probas("u", 224, a, same)
    assert report["passed"] and report["n_cells"] == 1
    moved = {"u@224|cls|none|id": np.array([[0.5, 0.5], [0.5, 0.5], [0.6, 0.4]], dtype=np.float32)}
    bad = pc.compare_probas("u", 224, a, moved)
    assert not bad["passed"] and bad["differing_cells"] == ["u@224|cls|none|id"]


def test_a_missing_reference_cell_fails_the_check():
    report = pc.compare_probas("u", 224, {("cls", "none", False): np.zeros((1, 2))}, {})
    assert not report["passed"] and report["missing_cells"] == ["u@224|cls|none|id"]


def _record(unit, feature, weighting, hit, warn):
    return {"unit": unit, "feature": feature, "weighting": weighting, "n_iter": [2000 if hit else 50],
            "max_iter": 2000, "hit_max_iter": hit, "convergence_warning": warn}


def test_summary_counts_totals_and_breakdowns():
    records = [_record("a", "cls", "none", False, False), _record("a", "cls4", "none", True, True),
               _record("b", "cls4", "balanced", True, False)]
    s = pc.summarize(records)
    assert (s["n_fits"], s["n_hit_max_iter"], s["n_convergence_warning"]) == (3, 2, 1)
    assert s["by_feature"]["cls4"] == {"n_fits": 2, "n_hit_max_iter": 2, "n_convergence_warning": 1}
    assert s["by_unit"]["a"]["n_hit_max_iter"] == 1
    assert s["by_weighting"]["balanced"]["n_fits"] == 1


def test_the_readme_is_generated_from_the_document():
    doc = {"max_iter": 2000, "blas_threads": 4, "code_revision": "abc",
           "summary": pc.summarize([_record("a", "cls", "none", True, True)]),
           "fits": [_record("a", "cls", "none", True, True)],
           "sanity": {"passed": True, "probas": {}, "metrics": {}}}
    text = pc.render_readme(doc)
    assert text == pc.render_readme(doc)
    assert "| a | cls | none | 2000 | 2000 | yes | yes |" in text
    assert "1 of 1" in text


@pytest.mark.parametrize("inside", ["pub/units/x", "pub"])
def test_the_output_may_not_land_inside_a_published_directory(tmp_path, inside):
    with pytest.raises(SystemExit):
        pc.check_out_dir(tmp_path / inside, protected=[tmp_path / "pub"])
    pc.check_out_dir(tmp_path / "diag", protected=[tmp_path / "pub"])


def test_committed_readme_matches_its_json():
    path = OUT_DIR / "probe_convergence.json"
    if not path.exists():
        pytest.skip("the pod run has not been fetched yet")
    import json

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert (OUT_DIR / "README.md").read_text(encoding="utf-8") == pc.render_readme(doc)
