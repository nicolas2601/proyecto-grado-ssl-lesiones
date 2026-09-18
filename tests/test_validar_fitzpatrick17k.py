"""Validacion externa provisional (sin GPU): compara estimate_fst_whole_image
y estimate_fst contra Fitzpatrick17k-C (version limpia y con splits
estandarizados de Abhishek, Jain & Hamarneh, Nature Scientific Data 2025) -
contraste con fototipo etiquetado por expertos, no verdad clinica."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import validar_fitzpatrick17k as vf  # noqa: E402


def _write_disk_image(path: Path, size=64, radius=15,
                       skin_rgb=(0.85, 0.65, 0.60), lesion_rgb=(0.30, 0.20, 0.10)):
    """Mismo generador sintetico que tests/test_fitzpatrick.py::disk_srgb_image
    y tests/test_diagnostico_fst_comparativa.py, escrito a disco como jpg."""
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.empty((size, size, 3))
    img[:] = skin_rgb
    img[dist <= radius] = lesion_rgb
    arr = (img * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_labels_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["md5hash", "fst"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_ground_truth_excludes_missing_fst_labels(tmp_path):
    csv_path = tmp_path / "labels.csv"
    _write_labels_csv(csv_path, [
        {"md5hash": "aaa", "fst": "0"},
        {"md5hash": "bbb", "fst": ""},
        {"md5hash": "ccc", "fst": "3"},
        {"md5hash": "ddd", "fst": "6"},
    ])

    gt = vf.load_ground_truth(csv_path)

    assert gt == {"ccc": 3, "ddd": 6}


def test_find_image_locates_file_by_md5hash(tmp_path):
    (tmp_path / "abc123.jpg").write_bytes(b"")

    found = vf.find_image(tmp_path, "abc123")

    assert found == tmp_path / "abc123.jpg"


def test_find_image_returns_none_when_missing(tmp_path):
    assert vf.find_image(tmp_path, "nope") is None


def test_accuracy_metrics_reports_exact_and_off_by_one_rates():
    pairs = [(1, 1), (1, 2), (3, 5), (6, None)]

    metrics = vf.accuracy_metrics(pairs)

    assert metrics["n_evaluable"] == 3
    assert metrics["n_unavailable"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["off_by_one_accuracy"] == pytest.approx(2 / 3)


def test_accuracy_metrics_confusion_matrix_only_counts_evaluable_pairs():
    pairs = [(1, 1), (1, 2), (3, 5), (6, None)]

    metrics = vf.accuracy_metrics(pairs)

    assert metrics["confusion"] == {"1": {"1": 1, "2": 1}, "3": {"5": 1}}


def test_accuracy_metrics_handles_all_unavailable():
    metrics = vf.accuracy_metrics([(1, None), (2, None)])

    assert metrics["n_evaluable"] == 0
    assert metrics["n_unavailable"] == 2
    assert metrics["accuracy"] is None
    assert metrics["off_by_one_accuracy"] is None
    assert metrics["confusion"] == {}


def test_evaluate_image_returns_predicted_bucket_for_both_methods():
    size, radius = 64, 15
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.empty((size, size, 3))
    img[:] = (0.85, 0.65, 0.60)
    img[dist <= radius] = (0.30, 0.20, 0.10)

    row = vf.evaluate_image(img, actual_fst=1)

    assert row["actual_fst"] == 1
    assert row["whole_image"]["status"] == "ok"
    assert row["masked"]["status"] == "ok"
    assert isinstance(row["whole_image"]["predicted_bucket"], int)
    assert isinstance(row["masked"]["predicted_bucket"], int)


def test_run_writes_report_matching_evaluated_images(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_disk_image(image_root / "aaa.jpg")
    _write_disk_image(image_root / "bbb.jpg")
    labels_csv = tmp_path / "Fitzpatrick17k-C.csv"
    _write_labels_csv(labels_csv, [
        {"md5hash": "aaa", "fst": "1"},
        {"md5hash": "bbb", "fst": "3"},
        {"md5hash": "ccc", "fst": "2"},  # imagen faltante a proposito
    ])
    out_json = tmp_path / "validacion.json"
    args = vf.parse_args([
        "--image-root", str(image_root),
        "--labels-csv", str(labels_csv),
        "--output-json", str(out_json),
    ])

    vf.run(args)

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.0"
    assert report["n_ground_truth"] == 3
    assert report["n_images_found"] == 2
    assert report["n_images_missing"] == 1
    assert len(report["records"]) == 3
    assert "whole_image" in report["metrics"]
    assert "masked" in report["metrics"]


def test_smoke_flag_caps_number_of_evaluated_rows(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for i in range(vf.SMOKE_MAX_IMAGES + 5):
        name = f"img{i:03d}"
        _write_disk_image(image_root / f"{name}.jpg")
        rows.append({"md5hash": name, "fst": "1"})
    labels_csv = tmp_path / "Fitzpatrick17k-C.csv"
    _write_labels_csv(labels_csv, rows)
    args = vf.parse_args([
        "--image-root", str(image_root),
        "--labels-csv", str(labels_csv),
        "--output-json", str(tmp_path / "out.json"),
        "--smoke",
    ])

    report_rows = vf.run(args)

    assert len(report_rows) == vf.SMOKE_MAX_IMAGES


def test_parse_args_defaults_color_constancy_to_identity():
    args = vf.parse_args([
        "--image-root", "x", "--labels-csv", "y", "--output-json", "z",
    ])

    assert args.color_constancy == "identity"


def test_invalid_color_constancy_choice_is_rejected():
    with pytest.raises(SystemExit):
        vf.parse_args([
            "--image-root", "x", "--labels-csv", "y", "--output-json", "z",
            "--color-constancy", "no-existe",
        ])


def test_color_constancy_option_shifts_the_whole_image_ita(tmp_path):
    """Una imagen con un sesgo de color fuerte (canales muy desbalanceados)
    debe dar un ITA distinto segun se aplique o no gray_world antes de
    estimar - si no cambia nada, el flag no esta realmente conectado."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    arr = np.full((64, 64, 3), (0.9, 0.5, 0.2))
    Image.fromarray((arr * 255).astype(np.uint8), mode="RGB").save(image_root / "aaa.jpg")
    labels_csv = tmp_path / "Fitzpatrick17k-C.csv"
    _write_labels_csv(labels_csv, [{"md5hash": "aaa", "fst": "1"}])

    identity_args = vf.parse_args([
        "--image-root", str(image_root), "--labels-csv", str(labels_csv),
        "--output-json", str(tmp_path / "identity.json"),
    ])
    gray_world_args = vf.parse_args([
        "--image-root", str(image_root), "--labels-csv", str(labels_csv),
        "--output-json", str(tmp_path / "gray_world.json"),
        "--color-constancy", "gray_world",
    ])

    identity_rows = vf.run(identity_args)
    gray_world_rows = vf.run(gray_world_args)

    ita_identity = identity_rows[0]["whole_image"]["ita"]
    ita_gray_world = gray_world_rows[0]["whole_image"]["ita"]
    assert ita_identity != pytest.approx(ita_gray_world)


def test_report_records_which_color_constancy_op_was_applied(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_disk_image(image_root / "aaa.jpg")
    labels_csv = tmp_path / "Fitzpatrick17k-C.csv"
    _write_labels_csv(labels_csv, [{"md5hash": "aaa", "fst": "1"}])
    out_json = tmp_path / "out.json"
    args = vf.parse_args([
        "--image-root", str(image_root), "--labels-csv", str(labels_csv),
        "--output-json", str(out_json), "--color-constancy", "shades_of_gray",
    ])

    vf.run(args)

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["color_constancy"] == "shades_of_gray"


def test_missing_image_is_recorded_without_crashing(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    labels_csv = tmp_path / "Fitzpatrick17k-C.csv"
    _write_labels_csv(labels_csv, [{"md5hash": "ghost", "fst": "4"}])
    args = vf.parse_args([
        "--image-root", str(image_root),
        "--labels-csv", str(labels_csv),
        "--output-json", str(tmp_path / "out.json"),
    ])

    rows = vf.run(args)

    assert rows[0]["status"] == "image_missing"
    assert rows[0]["whole_image"] is None
    assert rows[0]["masked"] is None
