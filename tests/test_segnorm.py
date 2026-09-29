"""Contract tests for the isolated SegRNorm/SegXNorm baseline."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dermssl.segnorm import (
    SegNormConfig,
    _as_rgb_float,
    _gaussian_blur,
    normalize_red,
    normalize_x,
    segment,
)
from env_sidecar import sidecar_env  # noqa: E402


def red_dominant_disk(size: int = 33) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= 7**2
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[...] = (0.2, 0.45, 0.2)
    image[disk] = (0.9, 0.2, 0.2)
    return image


def x_dominant_disk(size: int = 33) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[...] = (0.2, 0.2, 0.9)
    image[12:21, 12:21] = (0.9, 0.1, 0.1)
    return image


def circular_field_image(size: int = 41, radius: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    field = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= radius**2
    lesion = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= 5**2
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[field] = (0.2, 0.45, 0.2)
    image[lesion] = (0.9, 0.2, 0.2)
    return image


def test_red_normalization_is_the_paper_formula_and_is_deterministic():
    image = red_dominant_disk()

    first = normalize_red(image)
    second = normalize_red(image)

    expected = image[..., 0] / np.sqrt(np.sum(image**2, axis=-1))
    np.testing.assert_allclose(first, expected)
    np.testing.assert_array_equal(first, second)


def test_x_normalization_uses_the_paper_xyz_matrix():
    image = np.array([[[0.4, 0.5, 0.2]]], dtype=np.float32)
    xyz = np.array([0.4, 0.5, 0.2]) @ np.array(
        [[0.49, 0.31, 0.2], [0.17697, 0.8124, 0.01063], [0.0, 0.01, 0.99]]
    ).T / 0.17697

    normalized = normalize_x(image)

    assert normalized.shape == (1, 1)
    assert normalized[0, 0] == pytest.approx(xyz[0] / np.linalg.norm(xyz))


def test_integer_inputs_are_scaled_by_their_dtype_range():
    image = np.array([[[65535, 32768, 1]]], dtype=np.uint16)

    converted = _as_rgb_float(image)

    np.testing.assert_allclose(converted, [[[1.0, 32768 / 65535, 1 / 65535]]])


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (np.uint8, [[[255, 128, 1]]], [[[1.0, 128 / 255, 1 / 255]]]),
        (np.uint16, [[[65535, 32768, 1]]], [[[1.0, 32768 / 65535, 1 / 65535]]]),
        (np.float32, [[[0.25, 0.5, 0.75]]], [[[0.25, 0.5, 0.75]]]),
    ],
)
def test_conversion_helper_preserves_mixed_channel_values(dtype, values, expected):
    converted = _as_rgb_float(np.array(values, dtype=dtype))

    np.testing.assert_allclose(converted, expected, rtol=0, atol=1e-12)


def test_float_inputs_outside_unit_range_are_rejected():
    with pytest.raises(ValueError, match="float image values must be in the range"):
        normalize_red(np.array([[[255.0, 0.0, 0.0]]], dtype=np.float32))


def test_segment_returns_true_lesion_uint8_contract_and_bright_polarity():
    image = red_dominant_disk()
    result = segment(image, method="rnorm", config=SegNormConfig(sigma=0, min_component_pixels=1))

    assert result.mask.shape == image.shape[:2]
    assert result.mask.dtype == np.bool_
    assert result.mask[16, 16]
    assert not result.mask[0, 0]
    assert result.threshold is not None


def test_circular_roi_excludes_black_border_pixels_from_normalization_and_histogram():
    image = circular_field_image()

    result = segment(
        image,
        method="rnorm",
        config=SegNormConfig(roi="circular", sigma=0, min_component_pixels=1),
    )

    assert result.diagnostics["roi"]["method"] == "circular"
    assert result.diagnostics["histogram_pixel_count"] == result.diagnostics["roi"]["pixel_count"]
    assert result.diagnostics["histogram_pixel_count"] < image.shape[0] * image.shape[1]
    assert result.normalized_channel[0, 0] == 0
    assert not result.mask[0, 0]
    assert not result.mask[4, 4]


def test_circular_roi_is_deterministic_and_forces_outside_pixels_false():
    image = circular_field_image()
    config = SegNormConfig(roi="circular", sigma=0, min_component_pixels=1)

    first = segment(image, method="xnorm", config=config)
    second = segment(image, method="xnorm", config=config)

    np.testing.assert_array_equal(first.mask, second.mask)
    np.testing.assert_array_equal(first.normalized_channel, second.normalized_channel)
    assert first.diagnostics == second.diagnostics
    assert not first.mask[0, 20]
    assert not first.mask[4, 20]


def test_circular_roi_masked_blur_does_not_use_outside_pixels():
    size = 25
    yy, xx = np.mgrid[:size, :size]
    roi = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= 8**2
    channel = np.full((size, size), 5.0, dtype=np.float64)
    channel[roi] = 1.0

    smoothed = _gaussian_blur(channel, sigma=2.0, size=5, valid_mask=roi)

    assert smoothed[12, 12] == pytest.approx(1.0)
    assert smoothed[12, 20] == pytest.approx(1.0)
    assert smoothed[12, 3] == 0.0


def test_circular_roi_configuration_rejects_invalid_numeric_values():
    with pytest.raises(ValueError, match="roi_black_threshold"):
        segment(
            circular_field_image(),
            method="rnorm",
            config=SegNormConfig(roi="circular", roi_black_threshold=1.1),
        )

    with pytest.raises(ValueError, match="roi_margin"):
        segment(
            circular_field_image(),
            method="rnorm",
            config=SegNormConfig(roi="circular", roi_margin=-1),
        )


def test_segment_supports_explicit_dark_polarity():
    image = np.zeros((33, 33, 3), dtype=np.float32)
    image[...] = (0.9, 0.2, 0.2)
    yy, xx = np.mgrid[:33, :33]
    disk = (yy - 16) ** 2 + (xx - 16) ** 2 <= 7**2
    image[disk] = (0.2, 0.45, 0.2)
    result = segment(
        image,
        method="rnorm",
        config=SegNormConfig(sigma=0, min_component_pixels=1, polarity="dark"),
    )

    assert result.diagnostics["polarity"] == "dark"
    assert result.mask[16, 16]
    assert not result.mask[0, 0]


def test_segment_xnorm_separates_a_synthetic_foreground():
    result = segment(
        x_dominant_disk(), method="xnorm", config=SegNormConfig(sigma=0, min_component_pixels=1)
    )

    assert result.mask[16, 16]
    assert not result.mask[0, 0]


@pytest.mark.parametrize("method", ["rnorm", "xnorm"])
def test_constant_images_are_safe_and_empty(method: str):
    result = segment(
        np.full((12, 10, 3), 0.5, dtype=np.float32),
        method=method,
        config=SegNormConfig(sigma=0),
    )

    assert result.mask.dtype == np.bool_
    assert not result.mask.any()
    assert result.diagnostics["degenerate_channel"] is True


def test_roi_excludes_border_and_preserves_output_shape():
    image = red_dominant_disk()
    config = SegNormConfig(roi=(4, 4, 29, 29), sigma=0, min_component_pixels=1)

    result = segment(image, method="rnorm", config=config)

    assert result.mask.shape == image.shape[:2]
    assert not result.mask[:4].any()
    assert not result.mask[:, :4].any()


def test_small_components_are_removed_and_holes_are_filled():
    image = red_dominant_disk(size=41)
    image[2, 2] = (0.95, 0.05, 0.05)
    image[20, 20] = (0.2, 0.45, 0.2)
    config = SegNormConfig(sigma=0, min_component_pixels=10)

    result = segment(image, method="rnorm", config=config)

    assert not result.mask[2, 2]
    assert result.mask[20, 20]


def test_cli_writes_new_artifacts_without_overwriting_existing_output(tmp_path, monkeypatch):
    from scripts.run_segnorm_baseline import run
    from PIL import Image

    image_path = tmp_path / "case.png"
    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "image_id", "image_path"])
        writer.writeheader()
        writer.writerow({"dataset_id": "synthetic", "image_id": "case", "image_path": str(image_path)})
    output_dir = tmp_path / "new-results"

    run(["--manifest", str(manifest_path), "--output-dir", str(output_dir), "--method", "both"])

    assert (output_dir / "masks" / "case_rnorm.png").exists()
    assert (output_dir / "masks" / "case_xnorm.png").exists()
    records = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert {record["method"] for record in records} == {"rnorm", "xnorm"}
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileExistsError):
        run(["--manifest", str(manifest_path), "--output-dir", str(output_dir), "--method", "rnorm"])


def _write_manifest(tmp_path: Path, records: list[dict[str, str]]) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "image_id", "image_path"])
        writer.writeheader()
        writer.writerows(records)
    return manifest_path


def test_cli_rejects_unsafe_image_ids(tmp_path):
    from scripts.run_segnorm_baseline import run

    image_path = tmp_path / "case.png"
    from PIL import Image

    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "../case", "image_path": str(image_path)}],
    )

    with pytest.raises(ValueError, match="safe simple component"):
        run(["--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results")])


def test_cli_reports_all_missing_requested_ids(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": "case.png"}],
    )

    with pytest.raises(ValueError, match="missing image IDs: absent-one, absent-two"):
        run(
            [
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(tmp_path / "results"),
                "--image-id",
                "absent-one",
                "--image-id",
                "absent-two",
            ]
        )


@pytest.mark.parametrize("seed", [None, 20260920])
def test_explicit_image_ids_ignore_seed_without_sampling_and_are_deterministic(seed):
    from scripts.run_segnorm_baseline import _select_records

    records = [
        {"dataset_id": "synthetic", "image_id": image_id, "image_path": f"{image_id}.png"}
        for image_id in ("case-a", "case-b", "case-c")
    ]
    requested_ids = ["case-c", "case-a"]

    first = _select_records(records, "synthetic", requested_ids, None, seed, None)
    second = _select_records(records, "synthetic", requested_ids, None, seed, None)

    assert [record["image_id"] for record in first] == ["case-a", "case-c"]
    assert first == second


def test_explicit_image_ids_use_dataset_field_from_real_manifest_schema():
    from scripts.run_segnorm_baseline import _select_records

    records = [
        {"index": "0", "dataset": "isic2019", "image_id": "ISIC_0000000", "path": "one.jpg", "split": "train"},
        {"index": "1", "dataset": "isic2019", "image_id": "ISIC_0000001", "path": "two.jpg", "split": "val"},
        {"index": "2", "dataset": "ham10000", "image_id": "ISIC_0000002", "path": "three.jpg", "split": "train"},
    ]

    selected = _select_records(records, "isic2019", ["ISIC_0000001"], None, None, None)

    assert selected == [records[1]]


def test_real_manifest_dataset_column_remains_canonical_for_stratification():
    from scripts.run_segnorm_baseline import _normalise_stratification_column

    assert _normalise_stratification_column("dataset") == "dataset"


def test_cli_runs_end_to_end_with_real_manifest_schema(tmp_path):
    from PIL import Image

    from scripts.run_segnorm_baseline import run

    image_path = tmp_path / "images" / "case.png"
    image_path.parent.mkdir()
    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "dataset", "image_id", "path", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "index": "0",
                "dataset": "synthetic-real-schema",
                "image_id": "case",
                "path": "images/case.png",
                "split": "diagnostic",
            }
        )
    output_dir = tmp_path / "results"

    run(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--method",
            "rnorm",
            "--sigma",
            "0",
            "--min-component-pixels",
            "1",
        ]
    )

    assert (output_dir / "masks" / "case_rnorm.png").exists()
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics[0]["dataset"] == "synthetic-real-schema"
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["sources"][0]["dataset"] == "synthetic-real-schema"
    assert provenance["sources"][0]["path"] == str(image_path.resolve())


def test_cli_path_base_resolves_relative_paths_and_records_policy(tmp_path):
    from PIL import Image

    image_path = tmp_path / "images" / "case.png"
    image_path.parent.mkdir()
    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    manifest_path = tmp_path / "manifests" / "manifest.csv"
    manifest_path.parent.mkdir()
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "image_id", "path"])
        writer.writeheader()
        writer.writerow({"dataset": "synthetic", "image_id": "case", "path": "images/case.png"})

    output_dir = tmp_path / "results"
    from scripts.run_segnorm_baseline import run

    run(
        [
            "--manifest",
            str(manifest_path),
            "--path-base",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--method",
            "rnorm",
            "--sigma",
            "0",
            "--min-component-pixels",
            "1",
        ]
    )

    assert (output_dir / "masks" / "case_rnorm.png").exists()
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["path_resolution"] == {
        "policy": "explicit_path_base",
        "base": str(tmp_path.resolve()),
    }
    assert provenance["sources"][0]["path"] == str(image_path.resolve())


def test_cli_reports_missing_relative_image_path_with_image_id_and_resolved_path(tmp_path):
    (tmp_path / "manifests").mkdir()
    manifest_path = _write_manifest(
        tmp_path / "manifests",
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": "missing.png"}],
    )

    from scripts.run_segnorm_baseline import run

    with pytest.raises(FileNotFoundError, match=r"image path for image_id 'case' does not exist"):
        run(
            [
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(tmp_path / "results"),
                "--method",
                "rnorm",
            ]
        )


def test_cli_rejects_manifest_without_real_or_legacy_path_field(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "image_id", "split"])
        writer.writeheader()
        writer.writerow({"dataset": "synthetic", "image_id": "case", "split": "diagnostic"})

    with pytest.raises(ValueError, match="required image path field: expected 'path' or 'image_path'"):
        run(["--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results")])


def test_cli_rejects_duplicate_manifest_ids(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [
            {"dataset_id": "synthetic", "image_id": "case", "image_path": "case.png"},
            {"dataset_id": "synthetic", "image_id": "case", "image_path": "case.png"},
        ],
    )

    with pytest.raises(ValueError, match="ambiguous image IDs: case"):
        run(["--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results")])


def test_cli_rejects_case_insensitive_output_collisions(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [
            {"dataset_id": "synthetic", "image_id": "case", "image_path": "case.png"},
            {"dataset_id": "synthetic", "image_id": "CASE", "image_path": "CASE.png"},
        ],
    )

    with pytest.raises(ValueError, match="output filename collision"):
        run(["--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results")])


def test_cli_rejects_duplicate_requested_ids(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(tmp_path, [])

    with pytest.raises(ValueError, match="duplicate requested image IDs: case"):
        run(
            [
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(tmp_path / "results"),
                "--image-id",
                "case",
                "--image-id",
                "case",
            ]
        )


def test_cli_diagnostics_record_effective_configuration_and_provenance(tmp_path):
    from scripts.run_segnorm_baseline import run
    from PIL import Image

    image_path = tmp_path / "case.png"
    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": str(image_path)}],
    )

    run(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "results"),
            "--method",
            "rnorm",
            "--polarity",
            "dark",
            "--sigma",
            "0",
            "--min-component-pixels",
            "1",
        ]
    )

    record = json.loads((tmp_path / "results" / "diagnostics.json").read_text(encoding="utf-8"))[0]
    assert record["method"] == "rnorm"
    assert record["polarity"] == "dark"
    assert record["input_dimensions"] == [33, 33, 3]
    assert len(record["input_hash"]) == 64
    assert record["configuration"]["sigma"] == 0.0
    assert record["provenance"]


def test_cli_circular_roi_records_method_threshold_margin_and_assumption(tmp_path):
    from scripts.run_segnorm_baseline import run
    from PIL import Image

    image_path = tmp_path / "case.png"
    Image.fromarray((circular_field_image() * 255).astype(np.uint8)).save(image_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": str(image_path)}],
    )

    run(
        [
            "--manifest", str(manifest_path),
            "--output-dir", str(tmp_path / "results"),
            "--method", "rnorm",
            "--roi", "circular",
            "--roi-black-threshold", "0.1",
            "--roi-margin", "2",
            "--sigma", "0",
            "--min-component-pixels", "1",
        ]
    )

    record = json.loads((tmp_path / "results" / "diagnostics.json").read_text(encoding="utf-8"))[0]
    provenance = json.loads((tmp_path / "results" / "provenance.json").read_text(encoding="utf-8"))
    assert record["configuration"]["roi"] == "circular"
    assert record["configuration"]["roi_black_threshold"] == pytest.approx(0.1)
    assert record["configuration"]["roi_margin"] == 2
    assert provenance["configuration"]["roi"] == "circular"
    assert provenance["configuration"]["roi_black_threshold"] == pytest.approx(0.1)
    assert provenance["configuration"]["roi_margin"] == 2
    assert provenance["roi"]["assumption"]


def test_cli_rejects_invalid_roi_mode(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": "missing.png"}],
    )

    with pytest.raises(SystemExit):
        run([
            "--manifest", str(manifest_path),
            "--output-dir", str(tmp_path / "results"),
            "--roi", "invalid",
        ])


def test_cli_writes_run_level_provenance_with_hashes_and_polarity_assumption(tmp_path):
    from scripts.run_segnorm_baseline import run
    from PIL import Image

    image_path = tmp_path / "case.png"
    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": str(image_path)}],
    )
    output_dir = tmp_path / "results"

    run(["--manifest", str(manifest_path), "--output-dir", str(output_dir), "--method", "rnorm"])

    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["manifest"]["path"] == str(manifest_path.resolve())
    assert len(provenance["manifest"]["sha256"]) == 64
    assert provenance["selected_image_ids"] == ["case"]
    assert len(provenance["sources"][0]["file_sha256"]) == 64
    assert provenance["polarity"]["value"] == "bright"
    assert provenance["polarity"]["status"] == "implementation_assumption"
    assert provenance["paper_variant"] == "diagnostic_baseline_not_exact_reproduction"
    assert provenance["decoded_image_hashes"]["case"]
    assert len(provenance["source_tree_hash"]) == 64
    assert {source["path"] for source in provenance["source_files"]} == {
        "scripts/run_segnorm_baseline.py",
        "src/dermssl/segnorm.py",
    }
    assert provenance["environment"]["python_version"]
    assert len(provenance["environment"]["packages_sha256"]) == 64
    assert sidecar_env(output_dir / "provenance.json")["schema"] == 1
    assert "secret" not in json.dumps(provenance).lower()


def test_source_tree_provenance_changes_when_an_implementation_file_changes(tmp_path):
    from scripts.run_segnorm_baseline import _source_tree_provenance

    source_root = tmp_path / "repo"
    source_file = source_root / "src" / "dermssl" / "segnorm.py"
    cli_file = source_root / "scripts" / "run_segnorm_baseline.py"
    source_file.parent.mkdir(parents=True)
    cli_file.parent.mkdir(parents=True)
    source_file.write_text("source-v1\n", encoding="utf-8")
    cli_file.write_text("cli\n", encoding="utf-8")

    first = _source_tree_provenance(source_root, [source_file, cli_file])
    source_file.write_text("source-v2\n", encoding="utf-8")
    second = _source_tree_provenance(source_root, [source_file, cli_file])

    assert first["source_tree_hash"] != second["source_tree_hash"]
    assert first["files"][0]["path"] == "scripts/run_segnorm_baseline.py"
    assert first["files"][0]["size"] == cli_file.stat().st_size
    assert len(first["files"][0]["sha256"]) == 64


def test_environment_fingerprint_is_deterministic_and_excludes_environment_values(monkeypatch):
    from scripts.run_segnorm_baseline import _environment_fingerprint

    monkeypatch.setenv("SEGNORM_TEST_SECRET", "do-not-record-this")

    first = _environment_fingerprint()
    second = _environment_fingerprint()

    assert first == second
    assert first["python_version"]
    assert first["platform"]
    assert len(first["packages_sha256"]) == 64
    assert "do-not-record-this" not in json.dumps(first)
    assert "SEGNORM_TEST_SECRET" not in json.dumps(first)


def test_cli_sampling_is_deterministic_and_stratified(tmp_path):
    from scripts.run_segnorm_baseline import run
    from PIL import Image

    image_path = tmp_path / "case.png"
    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    records = [
        {"dataset_id": dataset, "image_id": f"{dataset}-{index}", "image_path": str(image_path)}
        for dataset in ("a", "b")
        for index in range(3)
    ]
    manifest_path = _write_manifest(tmp_path, records)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    args = [
        "--manifest", str(manifest_path), "--output-dir", str(first_output),
        "--method", "rnorm", "--sample-size", "4", "--seed", "19",
        "--stratify-by", "dataset_id", "--sigma", "0", "--min-component-pixels", "1",
    ]

    run(args)
    run([*args[0:3], str(second_output), *args[4:]])

    first = json.loads((first_output / "provenance.json").read_text(encoding="utf-8"))
    second = json.loads((second_output / "provenance.json").read_text(encoding="utf-8"))
    assert first["selected_image_ids"] == second["selected_image_ids"]
    assert first["sampling"] == {"sample_size": 4, "seed": 19, "stratify_by": "dataset_id"}
    assert {source["stratum"] for source in first["sources"]} == {"a", "b"}


def test_cli_rejects_impossible_stratified_sample(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [
            {"dataset_id": "a", "image_id": "a-1", "image_path": "missing.png"},
            {"dataset_id": "b", "image_id": "b-1", "image_path": "missing.png"},
        ],
    )

    with pytest.raises(ValueError, match="at least one image per stratum"):
        run([
            "--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results"),
            "--sample-size", "1", "--seed", "1", "--stratify-by", "dataset",
        ])


def test_cli_rejects_sample_larger_than_selection(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "a", "image_id": "a-1", "image_path": "missing.png"}],
    )

    with pytest.raises(ValueError, match="exceeds available records"):
        run([
            "--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results"),
            "--sample-size", "2", "--seed", "1",
        ])


def test_cli_rejects_non_positive_sample_size(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "a", "image_id": "a-1", "image_path": "missing.png"}],
    )

    with pytest.raises(ValueError, match="sample_size must be positive"):
        run([
            "--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results"),
            "--sample-size", "0", "--seed", "1",
        ])


def test_cli_cleans_staging_after_processing_failure(tmp_path):
    from scripts.run_segnorm_baseline import run
    from PIL import Image

    image_path = tmp_path / "case.png"
    Image.fromarray((red_dominant_disk() * 255).astype(np.uint8)).save(image_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": str(image_path)}],
    )
    output_dir = tmp_path / "results"

    with pytest.raises(ValueError, match="kernel_size"):
        run(
            [
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--kernel-size",
                "4",
            ]
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".results.staging-*"))


def test_cli_removes_only_empty_parent_directories_created_by_invocation(tmp_path):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": "missing.png"}],
    )
    pre_existing_parent = tmp_path / "pre-existing"
    pre_existing_parent.mkdir()
    output_dir = pre_existing_parent / "nested" / "results"

    with pytest.raises(FileNotFoundError):
        run([
            "--manifest", str(manifest_path), "--output-dir", str(output_dir),
            "--method", "rnorm",
        ])

    assert pre_existing_parent.exists()
    assert not (pre_existing_parent / "nested").exists()


def test_cli_preserves_original_error_if_staging_cleanup_fails(tmp_path, monkeypatch):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": "missing.png"}],
    )

    def fail_cleanup(_path):
        raise OSError("cleanup failed")

    monkeypatch.setattr("scripts.run_segnorm_baseline.shutil.rmtree", fail_cleanup)
    with pytest.raises(FileNotFoundError):
        run([
            "--manifest", str(manifest_path), "--output-dir", str(tmp_path / "results"),
            "--method", "rnorm",
        ])


def test_cli_cleans_created_parents_when_staging_creation_fails(tmp_path, monkeypatch):
    from scripts.run_segnorm_baseline import run

    manifest_path = _write_manifest(
        tmp_path,
        [{"dataset_id": "synthetic", "image_id": "case", "image_path": "missing.png"}],
    )
    output_dir = tmp_path / "created" / "results"

    def fail_staging(*_args, **_kwargs):
        raise OSError("staging failed")

    monkeypatch.setattr("scripts.run_segnorm_baseline.tempfile.mkdtemp", fail_staging)
    with pytest.raises(OSError, match="staging failed"):
        run([
            "--manifest", str(manifest_path), "--output-dir", str(output_dir),
            "--method", "rnorm",
        ])

    assert not output_dir.parent.exists()


def test_cli_max_side_downscales_the_image_before_segmenting(tmp_path):
    from scripts.run_segnorm_baseline import run
    from PIL import Image

    big = np.kron(red_dominant_disk(), np.ones((8, 8, 1)))  # 8x the synthetic size
    image_path = tmp_path / "big.png"
    Image.fromarray((big * 255).astype(np.uint8)).save(image_path)
    manifest_path = _write_manifest(
        tmp_path, [{"dataset_id": "synthetic", "image_id": "big", "image_path": str(image_path)}]
    )
    output_dir = tmp_path / "small"

    run(["--manifest", str(manifest_path), "--output-dir", str(output_dir), "--method", "rnorm",
         "--max-side", "64"])

    with Image.open(output_dir / "masks" / "big_rnorm.png") as mask:
        assert max(mask.size) == 64
    records = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert records[0]["max_side"] == 64
