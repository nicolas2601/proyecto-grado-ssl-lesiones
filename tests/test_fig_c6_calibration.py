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

import fig_c6_calibration as figure  # noqa: E402


def test_reliability_bins_use_predicted_class_confidence_and_track_support():
    labels = np.array([0, 1, 0, 0])
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.35, 0.65], [0.2, 0.8]])

    result = figure.reliability_data(labels, probabilities, published_ece=0.2375, bins=4)

    assert result["confidence_definition"] == "maximum class probability"
    assert result["bins"][0]["n"] == 0
    assert result["bins"][2] == {"index": 2, "low": 0.5, "high": 0.75, "confidence": 0.625,
                                 "accuracy": 0.5, "n": 2, "low_support": True}
    assert result["bins"][3]["n"] == 2
    assert result["bins"][2]["low_support"] is True
    assert result["ece"] == pytest.approx(0.2375)
    assert result["ece_matches_published"] is True


def test_reliability_data_hides_ece_when_source_value_does_not_match():
    labels = np.array([0, 1, 0, 0])
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.35, 0.65], [0.2, 0.8]])

    result = figure.reliability_data(labels, probabilities, published_ece=0.2, bins=4)

    assert result["ece"] is None
    assert result["ece_matches_published"] is False
    assert sum(row["n"] for row in result["bins"]) == len(labels)


def test_all_calibration_panels_recompute_published_ece():
    for dataset in figure.SOURCES:
        panel = figure.load_panel(REPO, dataset)
        assert panel["n"] > 0
        assert all(arm["ece_matches_published"] for arm in panel["arms"].values())
        assert all(sum(row["n"] for row in arm["bins"]) == panel["n"]
                   for arm in panel["arms"].values())


def test_build_exports_traceable_calibration_panels(tmp_path):
    outputs = figure.build(REPO, tmp_path)

    assert set(outputs) == set(figure.SOURCES)
    for dataset, paths in outputs.items():
        assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
        trace = json.loads((tmp_path / f"fig_c6_calibration_{dataset}.traceability.json").read_text())
        assert trace["confidence_definition"] == "maximum class probability"
        assert trace["bins"] == 15
        assert trace["low_support_threshold_n"] == 10
        assert trace["support_visualization"]
        assert all("n" in row for arm in trace["arms"].values() for row in arm["bins"])


def test_calibration_plot_has_support_histogram_and_low_support_markers():
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt

    panel = figure.load_panel(REPO, "ham10000")
    fig = figure._draw(panel)

    assert len(fig.axes) == 2
    assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])
    assert fig.axes[1].get_ylabel() == "n por bin"
    assert len(fig.axes[1].patches) == sum(
        row["n"] > 0 for arm in panel["arms"].values() for row in arm["bins"]
    )
    assert len(fig.axes[0].collections) == 2 * len(panel["arms"])
    assert np.allclose(fig.axes[0].collections[0].get_facecolors()[0], to_rgba(figure.style.COLORS["background"]))
    plt.close(fig)


def test_reliability_data_rejects_misaligned_or_invalid_probabilities():
    with pytest.raises(ValueError, match="aligned"):
        figure.reliability_data(np.array([0]), np.array([[0.4, 0.6], [0.1, 0.9]]), 0.1)

    with pytest.raises(ValueError, match="probabilities"):
        figure.reliability_data(np.array([0]), np.array([[0.4, 0.4]]), 0.1)
