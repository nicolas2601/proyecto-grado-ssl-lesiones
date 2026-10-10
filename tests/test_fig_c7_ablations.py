from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_c7_ablations as figure  # noqa: E402


def test_a71_b71_uses_verified_paired_macro_f1_and_discloses_protocol_bundle():
    data = figure.a71_b71_data(REPO)

    assert data["support"] == 2004
    assert data["arms"]["A71"]["config"] == "A_soup_a1.0@224|cls_attn|la0.5|tta"
    assert data["arms"]["B71"]["config"] == "B_s42_a1.0@224|cls4|balanced|tta"
    assert data["contrast"]["arm"] == "B71"
    assert data["contrast"]["control"] == "A71"
    assert data["contrast"]["delta"] == pytest.approx(-0.008781341600129111)
    assert data["contrast"]["ci95"] == pytest.approx((-0.0456512816038645, 0.027023250794903274))
    assert data["resampling_unit"] == "image"
    assert data["bootstrap"]["n_resamples"] == 1000


def test_f4_contrast_reports_registered_interval_and_all_seed_variability():
    data = figure.f4_robustness_data(REPO)

    assert data["support"] == 2004
    assert "bbox crop" in data["f4_definition"] and "mask descriptors" in data["f4_definition"]
    assert data["registered"]["delta"] == pytest.approx(0.035990551766247325)
    assert data["registered"]["ci95"] == pytest.approx((0.002048624770841898, 0.07778518372136149))
    assert [row["seed"] for row in data["seeds"]] == [42, 43, 44]
    assert [row["delta"] for row in data["seeds"]] == pytest.approx(
        [0.03002125313500692, 0.009175389575558413, 0.026051973357511082]
    )
    assert data["reading"] == "none"
    assert data["n_excluding_zero"] == 0


def test_c7_build_exports_two_distinct_traceable_figures(tmp_path):
    outputs = figure.build(REPO, tmp_path)

    assert set(outputs) == {"a71_vs_b71", "b71_vs_b71_f4"}
    for name, paths in outputs.items():
        assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
        stem = f"fig_c7_{name}"
        trace = json.loads((tmp_path / f"{stem}.traceability.json").read_text())
        assert len(trace["script_sha256"]) == 64
        assert (tmp_path / f"{stem}.caption.txt").is_file()
        assert (tmp_path / f"{stem}.note.txt").is_file()
        note = (tmp_path / f"{stem}.note.txt").read_text()
        assert "no un efecto aislado del tono" in note if name == "a71_vs_b71" else "0/3 IC" in note
