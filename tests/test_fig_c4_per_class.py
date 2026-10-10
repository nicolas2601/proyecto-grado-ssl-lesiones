from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_c4_per_class as figure  # noqa: E402


def test_class_metrics_use_true_support_and_predicted_support():
    rows = figure.class_metrics([[8, 2], [1, 9]], ["a", "b"])

    assert rows[0] == {
        "class": "a",
        "support": 10,
        "precision": pytest.approx(8 / 9),
        "sensitivity": pytest.approx(8 / 10),
        "f1": pytest.approx(16 / 19),
    }
    assert rows[1]["support"] == 10
    assert rows[1]["precision"] == pytest.approx(9 / 11)
    assert rows[1]["sensitivity"] == pytest.approx(9 / 10)
    assert rows[1]["f1"] == pytest.approx(18 / 21)


def test_class_metrics_use_zero_precision_when_class_is_never_predicted():
    rows = figure.class_metrics([[1, 0], [2, 0]], ["a", "b"])

    assert rows[1]["support"] == 2
    assert rows[1]["precision"] == 0
    assert rows[1]["sensitivity"] == 0
    assert rows[1]["f1"] == 0


@pytest.mark.parametrize(
    ("counts", "classes", "message"),
    [
        ([[1, 2]], ["a", "b"], "square"),
        ([[1, -1], [0, 1]], ["a", "b"], "non-negative"),
        ([[1, 0], [0, 0]], ["a", "b"], "support"),
        ([[1, 0], [0, 1]], ["a", "a"], "unique"),
    ],
)
def test_class_metrics_reject_invalid_matrices(counts, classes, message):
    with pytest.raises(ValueError, match=message):
        figure.class_metrics(counts, classes)


@pytest.mark.parametrize(
    ("dataset", "support"),
    [("ham10000", 2004), ("isic2019", 5067), ("isic2020", 6660)],
)
def test_c4_uses_matched_b71_a71_matrices_and_original_classes(dataset, support):
    panel = figure.figure_data(REPO)["datasets"][dataset]

    assert panel["n"] == support
    assert [model["key"] for model in panel["models"]] == ["b71", "a71"]
    assert panel["models"][0]["classes"] == panel["models"][1]["classes"]
    assert panel["models"][0]["support"] == panel["models"][1]["support"]
    assert sum(panel["models"][0]["support"]) == support
    assert all(
        0 <= item[metric] <= 1
        for model in panel["models"]
        for item in model["class_metrics"]
        for metric in ("precision", "sensitivity", "f1")
    )


def test_c4_figure_uses_three_explicit_metric_panels_and_traceable_sources(tmp_path):
    import matplotlib.pyplot as plt

    data = figure.figure_data(REPO)
    panel = data["datasets"]["ham10000"]
    fig = figure.draw(panel)
    assert len(fig.axes) == 3
    expected_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert all(list(axis.get_xticks()) == expected_ticks for axis in fig.axes)
    plt.close(fig)

    outputs = figure.build_all(REPO, tmp_path)
    assert set(outputs) == {"ham10000", "isic2019", "isic2020"}
    assert all({path.suffix for path in paths} == {".pdf", ".svg", ".png"} for paths in outputs.values())
    for dataset in outputs:
        stem = f"fig_c4_per_diagnosis_{dataset}"
        trace = json.loads((tmp_path / f"{stem}.traceability.json").read_text(encoding="utf-8"))
        assert trace["figure_id"] == "C4"
        assert trace["evaluation_unit"] == "image"
        assert len(trace["models"]) == 2
        assert (tmp_path / f"{stem}.caption.txt").is_file()
        assert (tmp_path / f"{stem}.note.txt").is_file()
    with pytest.raises(FileExistsError):
        figure.build_all(REPO, tmp_path)
