from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_s1_cases as figure  # noqa: E402


LEVELS = (("alto", "p90", 0), ("medio", "p50", 4), ("bajo", "p10", 8))


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(left.sum() + right.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.logical_and(left, right).sum()) / denominator


def _write_fixture(root: Path) -> Path:
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True)
    entries = []
    for level, criterion, offset in LEVELS:
        image_id = f"case-{level}"
        truth = np.zeros((16, 16), dtype=bool)
        truth[4:8, 4:8] = True
        student = np.zeros_like(truth)
        student[4:8, 4 + offset // 2:8 + offset // 2] = True
        s4 = np.zeros_like(truth)
        s4[4:8, 4 + offset // 2:8 + offset // 2] = True
        image = np.full((16, 16, 3), 180, dtype=np.uint8)
        image[truth] = (180, 80, 80)
        arrays = {
            "original.png": image,
            "reference_mask.png": truth.astype(np.uint8) * 255,
            "student_r3_mask.png": student.astype(np.uint8) * 255,
            "s4_a2_mask.png": s4.astype(np.uint8) * 255,
        }
        routes = {}
        for filename, pixels in arrays.items():
            path = cases_dir / f"{image_id}_{filename}"
            Image.fromarray(pixels).save(path)
            routes[filename] = path.relative_to(root).as_posix()
        entries.append({
            "image_id": image_id,
            "dataset": "ISIC 2018",
            "nivel": level,
            "criterio": criterion,
            "dice_student_r3": _dice(truth, student),
            "dice_s4a2": _dice(truth, s4),
            "rutas": routes,
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"cases": entries}, ensure_ascii=False), encoding="utf-8")
    return manifest


def test_s1_requires_high_medium_low_and_sorts_by_quality_level(tmp_path):
    manifest = _write_fixture(tmp_path / "S1")
    cases = figure.load_cases(manifest)

    assert [case["nivel"] for case in cases] == ["alto", "medio", "bajo"]
    assert [case["criterio"] for case in cases] == ["p90", "p50", "p10"]
    assert len({case["image_id"] for case in cases}) == 3
    assert all(case["resolution"] == (16, 16) for case in cases)
    assert all(set(np.unique(case["masks"]["reference_mask.png"])) <= {0, 1} for case in cases)


def test_s1_recomputes_both_dice_values_and_rejects_manifest_mismatch(tmp_path):
    manifest = _write_fixture(tmp_path / "S1")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["cases"][0]["dice_student_r3"] = 0.123
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Dice mismatch"):
        figure.load_cases(manifest)


def test_s1_requires_all_levels_and_rejects_resolution_mismatch(tmp_path):
    manifest = _write_fixture(tmp_path / "S1")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["cases"] = [case for case in data["cases"] if case["nivel"] != "bajo"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="alto.*medio.*bajo"):
        figure.load_cases(manifest)

    manifest = _write_fixture(tmp_path / "S1-mixed")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    case = data["cases"][1]
    route = case["rutas"]["original.png"]
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(manifest.parent / route)
    with pytest.raises(ValueError, match="resolution"):
        figure.load_cases(manifest)


def test_s1_build_fails_closed_without_manifest_and_draws_fixture_only_in_memory(tmp_path):
    import matplotlib.pyplot as plt

    missing_manifest = tmp_path / "S1" / "manifest.json"
    missing_output = tmp_path / "missing-outputs"
    with pytest.raises(FileNotFoundError, match="manifest"):
        figure.build(missing_manifest, outdir=missing_output)
    assert not missing_output.exists()

    manifest = _write_fixture(tmp_path / "S1-real-fixture")
    output = tmp_path / "temporary-test-exports"
    fig = figure.draw(figure.load_cases(manifest))
    assert len(fig.axes) == 15
    assert [axis.get_title() for axis in fig.axes[:5]] == list(figure.COLUMNS)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [text for text in fig.texts if text.get_text().startswith(LEVELS[0][0])
              or text.get_text().startswith(LEVELS[1][0]) or text.get_text().startswith(LEVELS[2][0])]
    assert len(labels) == 3
    for row, label in enumerate(labels):
        assert label.get_window_extent(renderer).x1 < fig.axes[row * 5].get_window_extent(renderer).x0 - 2
    plt.close(fig)
    assert not output.exists()
