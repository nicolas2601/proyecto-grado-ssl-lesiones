from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_c2_paired_delta_f1 as figure  # noqa: E402


def _json(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def test_ham_c2_uses_all_eight_paired_canonical_e3b_contrasts():
    data = figure.figure_data(REPO)
    group = next(item for item in data["groups"] if item["dataset"] == "ham10000")
    source = _json("resultados/banco_completo/e3b_bs64/e3b_summary.json")
    assert len(group["comparisons"]) == 8
    assert group["n_val"] == 2004
    assert group["unit"] == "lesion_id"
    assert group["n_resamples"] == 1000
    assert group["seed"] == 20260923
    model_by_label = {name: model for model, name in figure.bench_rows.MODEL_NAMES.items()}
    for comparison in group["comparisons"]:
        record = next(
            row for row in source["contrasts"]
            if row["metric"] == "macro_f1" and row["control"] == "B71_tta"
            and row["arm"] == model_by_label[comparison["control"]]
        )
        assert comparison["delta"] == pytest.approx(-record["point_delta"])
        assert comparison["low"] == pytest.approx(-record["ci95_high"])
        assert comparison["high"] == pytest.approx(-record["ci95_low"])


@pytest.mark.parametrize(("dataset", "unit", "support"), [
    ("isic2019", "image", 5067),
    ("isic2020", "patient_id", 6660),
])
def test_isic_c2_uses_only_paired_original_b71_public_control(dataset, unit, support):
    data = figure.figure_data(REPO)
    group = next(item for item in data["groups"] if item["dataset"] == dataset)
    source = _json(
        f"resultados/banco_completo/isic_eval/limpio_canonico_bs64/{dataset}/contrasts/"
        "B71__original__vs__PUBLIC_B71.json"
    )
    raw = next(row for row in source["result"]["paired"] if row["metric"] == "macro_f1")
    assert len(group["comparisons"]) == 1
    assert group["n_val"] == support
    assert group["unit"] == unit
    assert group["comparisons"][0]["delta"] == raw["point_delta"]
    assert group["comparisons"][0]["low"] == raw["ci95_low"]
    assert group["comparisons"][0]["high"] == raw["ci95_high"]
    assert group["n_resamples"] == 1000
    assert group["seed"] == 20260923


def test_c2_footnotes_do_not_overlap_x_axis_label():
    import matplotlib.pyplot as plt

    fig = figure._draw(figure.figure_data(REPO))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel = fig.axes[0].xaxis.label.get_window_extent(renderer)
    footnotes = [text.get_window_extent(renderer) for text in fig.texts]
    assert footnotes
    assert max(box.y1 for box in footnotes) + 4 < xlabel.y0
    plt.close(fig)


def test_c2_build_exports_vector_and_preview_with_traceability(tmp_path):
    paths = figure.build(REPO, tmp_path)
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert "C2 pareado" in (tmp_path / "fig_c2_paired_delta_f1.caption.txt").read_text(encoding="utf-8")
    note = (tmp_path / "fig_c2_paired_delta_f1.note.txt").read_text(encoding="utf-8")
    assert "HAM10000 2.004" in note
    assert "ISIC 2019 5.067" in note
    trace = json.loads((tmp_path / "fig_c2_paired_delta_f1.traceability.json").read_text(encoding="utf-8"))
    assert trace["multiplicity_adjustment"] == "none"
    assert trace["comparison_count"] == 10
    script_sha = hashlib.sha256((REPO / "scripts/figures/fig_c2_paired_delta_f1.py").read_bytes()).hexdigest()
    assert trace["figure_script"] == {"path": "scripts/figures/fig_c2_paired_delta_f1.py", "sha256": script_sha}
    with pytest.raises(FileExistsError):
        figure.build(REPO, tmp_path)
