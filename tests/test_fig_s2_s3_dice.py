"""S2/S3: Dice figures must read the committed per-image and summary evaluations consistently."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_s2_s3_dice as figure  # noqa: E402


@pytest.mark.parametrize("model", figure.MODELS)
@pytest.mark.parametrize("dataset,_label,n", figure.DATASETS)
def test_per_image_mean_matches_summary(model, dataset, _label, n):
    summary = figure.load_summary(REPO, model, dataset, n)
    values = figure.load_dice(REPO, model, dataset, n)
    assert abs(values.mean() - summary["mean"]) < 1e-6
    assert summary["lo"] <= summary["mean"] <= summary["hi"]


def test_wrong_n_aborts():
    with pytest.raises(ValueError):
        figure.load_dice(REPO, "student_r3", "isic2018", 10)


def test_builds_export_all_formats(tmp_path):
    for builder, stem in ((figure.build_s2, "fig_s2_dice_ci"), (figure.build_s3, "fig_s3_dice_distribution")):
        paths = builder(REPO, tmp_path)
        assert {p.suffix for p in paths} == {".pdf", ".svg", ".png"}
        trace = json.loads((tmp_path / f"{stem}.traceability.json").read_text(encoding="utf-8"))
        assert all(len(s["sha256"]) == 64 for s in trace["sources"])
