"""The C1 table is generated, and a lower ECE is never printed as a loss."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_c1_readme as br  # noqa: E402

CELLS = {
    "b__none": {"backbone": "b", "condition": "none", "gate_passes": True,
                "metrics": {"macro_f1": 0.6, "balanced_accuracy": 0.55, "auc": 0.9, "ece": 0.13}},
    "b__mask_pool": {"backbone": "b", "condition": "mask_pool", "gate_passes": False,
                     "metrics": {"macro_f1": 0.62, "balanced_accuracy": 0.57, "auc": 0.91, "ece": 0.10}},
}
ECE_ROW = {
    "arm": "b__mask_pool", "control": "b__none", "metric": "ece", "point_delta": -0.03,
    "ci95_low": -0.05, "ci95_high": -0.01, "excludes_zero": True, "direction": "arm_better",
}
BOOT = {"n_resamples": 1000, "n_val": 2004, "results": [ECE_ROW], "not_contrasted": ["x/y: motivo"]}


def test_a_lower_ece_is_reported_as_a_win_for_the_arm():
    text = br.render(CELLS, BOOT)
    assert "gana `b__mask_pool`" in text


def test_a_failed_gate_is_visible_in_the_table():
    assert "FRENA" in br.render(CELLS, BOOT)


def test_what_was_not_contrasted_is_listed():
    assert "x/y: motivo" in br.render(CELLS, BOOT)


def test_check_fails_on_a_stale_file_and_passes_on_a_fresh_one(tmp_path):
    (tmp_path / "c1_cells.json").write_text(json.dumps(CELLS))
    (tmp_path / "c1_bootstrap.json").write_text(json.dumps(BOOT))
    (tmp_path / "README.md").write_text("editado a mano")

    assert br.main(["--dir", str(tmp_path), "--check"]) == 1
    assert br.main(["--dir", str(tmp_path)]) == 0
    assert br.main(["--dir", str(tmp_path), "--check"]) == 0


def test_the_published_readme_is_fresh_when_it_exists():
    """The real gate: once C1 results are committed, the table must match them."""
    if not (br.C1_DIR / "c1_cells.json").exists():
        import pytest

        pytest.skip("todavia no hay resultados de C1 publicados")
    assert br.main(["--check"]) == 0
