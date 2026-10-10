from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_c5_pr_roc as figure  # noqa: E402


def test_binary_curve_metrics_are_computed_from_labels_and_scores():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.4])

    result = figure.binary_metrics(labels, scores)

    assert result["prevalence"] == 0.5
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["precision"][-1] == 1.0
    assert result["recall"][0] == 1.0
    assert result["recall"][-1] == 0.0
    assert np.all(np.diff(result["recall"]) <= 0)


def test_binary_metrics_group_equal_scores_as_one_threshold():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.8, 0.8, 0.2, 0.1])

    result = figure.binary_metrics(labels, scores)

    assert result["average_precision"] == pytest.approx(7 / 12)
    assert result["roc_auc"] == pytest.approx(0.625)
    assert len(result["recall"]) == 4


def test_ham10000_loads_e3b_probability_rows_with_melanoma_positive():
    panel = figure.load_panel(REPO, "ham10000")

    assert panel["n"] == 2004
    assert panel["positive_count"] == 222
    assert "B71_tta" in panel["arms"]
    assert "dinov2_vitb14" in panel["arms"]
    assert panel["arms"]["B71_tta"]["scores"].shape == (2004,)
    summary = json.loads((REPO / "resultados/banco_completo/e3b_bs64/e3b_summary.json").read_text())
    assert panel["arms"]["B71_tta"]["roc_auc"] == pytest.approx(
        summary["proposed"]["rows"]["B71_tta"]["mel_auc"]
    )


def test_isic2019_uses_matching_b71_original_and_public_probabilities():
    panel = figure.load_panel(REPO, "isic2019")

    assert panel["n"] == 5067
    assert panel["positive_count"] == 905
    assert set(arm["label"] for arm in panel["arms"].values()) == {"B71 original", "B71 limpio", "B71 público"}
    assert all(arm["scores"].shape == (5067,) for arm in panel["arms"].values())


def test_isic2020_finetuning_is_separate_and_uses_npz_ids():
    panel = figure.load_panel(REPO, "isic2020_ft")

    assert panel["n"] == 6660
    assert panel["positive_count"] == 122
    assert panel["arms"]["dinov2_vitb14"]["scores"].shape == (6660,)


def test_build_exports_all_datasets_with_traceability(tmp_path):
    outputs = figure.build(REPO, tmp_path)

    assert set(outputs) == {"ham10000", "isic2019", "isic2020", "isic2020_ft"}
    for dataset, paths in outputs.items():
        assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
        stem = f"fig_c5_pr_roc_{dataset}"
        trace = json.loads((tmp_path / f"{stem}.traceability.json").read_text())
        assert trace["positive_class"] == "melanoma"
        assert trace["evaluation_unit"] == "image"
        assert trace["threshold_selection_on_target_test"] is False
        assert (tmp_path / f"{stem}.caption.txt").is_file()
        assert (tmp_path / f"{stem}.note.txt").is_file()


def test_unknown_dataset_fails_closed():
    with pytest.raises(ValueError, match="Unknown C5 dataset"):
        figure.load_panel(REPO, "pad_ufes")
