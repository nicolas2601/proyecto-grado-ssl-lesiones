"""scripts/t4_repro_compare.py: two identity-only runs against each other and against v7.1."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from env_sidecar import sidecar_env  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import t4_repro_compare as cmp  # noqa: E402

UNIT = "public@224"
CELLS = ["public@224|cls|none|id", "public@224|cls4|la0.5|tta"]


def _run(root: Path, *, shift: float = 0.0, digest: str = "a" * 64, f1: float = 0.7) -> Path:
    udir = root / "identity" / "units" / UNIT
    udir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), 50)
    p1 = p.copy()
    p1[0] = [0.34 + shift, 0.33 - shift, 0.33]
    np.savez_compressed(udir / "probas.npz", y=rng.integers(0, 3, 50), p0=p, p1=p1)
    rows = {c: {"macro_f1": f1, "mel_recall": 0.5} for c in CELLS}
    (udir / "result.json").write_text(json.dumps({"cells": CELLS, "rows": rows,
                                                  "feature_sha256": {"fit_cls": digest}}))
    (root / "t4_holdout_grid.json").write_text(json.dumps({"inputs": {"identity": rows}}))
    return root


def test_two_identical_runs_are_reported_identical(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    report = cmp.compare_runs(a, b)
    assert report["identical"] is True
    assert report["units"][UNIT]["features_equal"] is True
    assert report["max_abs_delta"]["macro_f1"] == 0.0


def test_a_different_feature_hash_and_a_flipped_image_are_reported(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b", shift=-0.02, digest="b" * 64, f1=0.7011)
    report = cmp.compare_runs(a, b)
    assert report["identical"] is False
    unit = report["units"][UNIT]
    assert unit["features_equal"] is False and unit["differing_parts"] == ["fit_cls"]
    assert unit["cells"][CELLS[1]]["argmax_flips"] == 1
    assert report["max_abs_delta"]["macro_f1"] == pytest.approx(0.0011)


def test_each_run_is_compared_with_the_published_grid(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b", f1=0.702)
    published = tmp_path / "v71.json"
    published.write_text(json.dumps({"rows": {c: {"macro_f1": 0.7, "mel_recall": 0.5} for c in CELLS}}))
    out = tmp_path / "report.json"
    assert cmp.main(["--run-a", str(a), "--run-b", str(b), "--v71-grid", str(published), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["a_vs_published"]["max_abs_delta"]["macro_f1"] == 0.0
    assert doc["b_vs_published"]["max_abs_delta"]["macro_f1"] == pytest.approx(0.002)
    assert doc["a_vs_b"]["identical"] is False
    assert sidecar_env(out)["schema"] == 1


def test_missing_units_are_an_error_not_a_silent_skip(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    shutil.rmtree(b / "identity" / "units" / UNIT)
    with pytest.raises(SystemExit, match="unidades"):
        cmp.compare_runs(a, b)
