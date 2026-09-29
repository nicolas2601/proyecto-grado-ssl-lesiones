"""TDD contract for the Camino B pseudo-mask -> ITA/FST adapter."""

import csv
import json

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import fst_pseudo_manifest as fpm
from dermssl import fitzpatrick


def _write_fixture(tmp_path, *, include_stats=True):
    image_path = tmp_path / "images" / "ham" / "image-a.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (180, 130, 100)).save(image_path)

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "index,dataset,image_id,path,split\n"
        f"0,ham10000,img-a,{image_path},train\n",
        encoding="utf-8",
    )

    masks = np.zeros((1, 8, 8), dtype=np.uint8)
    masks[0, 2:5, 2:5] = 1
    masks[0, 0, 0:2] = 255
    masks_path = tmp_path / "pseudo_masks.npy"
    np.save(masks_path, masks)

    if include_stats:
        (tmp_path / "pseudo_masks_stats.json").write_text(
            json.dumps({"source": "ccam", "round": 2, "high": 0.65}),
            encoding="utf-8",
        )
    return manifest_path, masks_path


def test_run_reads_cache_inputs_and_writes_traceable_json(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)

    output_path = tmp_path / "fst_manifest.json"
    result = fpm.run(
        fpm.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--pseudo-masks",
                str(masks_path),
                "--output-json",
                str(output_path),
            ]
        )
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    row = saved["records"][0]
    assert result == saved
    assert saved["environment"]["schema"] == 1
    assert row["index"] == 0
    assert row["dataset"] == "ham10000"
    assert row["image_id"] == "img-a"
    assert row["split"] == "train"
    assert row["image_path"] == str((tmp_path / "images" / "ham" / "image-a.png").resolve())
    assert row["mask_source"] == "ccam"
    assert row["mask_round"] == 2
    assert row["mask_coverage"] == 9 / 64
    assert row["ignore_fraction"] == 2 / 64
    assert row["status"] == "ok"
    assert isinstance(row["ITA"], float)
    assert row["FST_bucket"] in range(1, 7)
    assert row["confidence"] > 0
    assert saved["summary"]["by_dataset"]["ham10000"]["n"] == 1
    assert saved["summary"]["by_dataset"]["ham10000"]["states"] == {"ok": 1}


def test_run_quarantines_insufficient_valid_lesion_without_inventing_fst(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path, include_stats=False)
    masks = np.load(masks_path)
    masks[:] = 0
    np.save(masks_path, masks)

    output = fpm.run(
        fpm.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--pseudo-masks",
                str(masks_path),
                "--output-json",
                str(tmp_path / "fst_manifest.json"),
            ]
        )
    )

    row = output["records"][0]
    assert row["status"] == "quarantine"
    assert row["quality"] == "insufficient_mask_coverage"
    assert row["ITA"] is None
    assert row["FST_bucket"] == "unavailable"
    assert row["confidence"] == 0.0


def test_ignore_pixels_are_excluded_from_ita_calculation(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    image_path = tmp_path / "images" / "ham" / "image-a.png"
    image = np.full((8, 8, 3), (180, 130, 100), dtype=np.uint8)
    image[0, 0:2] = 0
    Image.fromarray(image).save(image_path)

    output = fpm.run(
        fpm.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--pseudo-masks",
                str(masks_path),
                "--output-json",
                str(tmp_path / "fst_manifest.json"),
            ]
        )
    )
    pseudo = np.load(masks_path)[0]
    expected = fitzpatrick.estimate_fst(
        image.astype(np.float32) / 255.0,
        mask=pseudo == 1,
        border_mask=pseudo == 255,
    )

    assert output["records"][0]["ITA"] == expected.ita


def test_invalid_estimator_status_is_quarantined(tmp_path, monkeypatch):
    manifest_path, masks_path = _write_fixture(tmp_path)

    def invalid_estimate(*args, **kwargs):
        return fitzpatrick.FSTEstimate(
            bucket=None,
            ita=float("nan"),
            skin_fraction=0.0,
            status="no_skin",
            fallback_used=False,
            confidence=0.0,
            artifact_detected=False,
        )

    monkeypatch.setattr(fpm.fitzpatrick, "estimate_fst_from_mask_record", invalid_estimate)
    output = fpm.run(
        fpm.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--pseudo-masks",
                str(masks_path),
                "--output-json",
                str(tmp_path / "fst_manifest.json"),
            ]
        )
    )

    row = output["records"][0]
    assert row["status"] == "quarantine"
    assert row["quality"] == "invalid_estimate:no_skin"
    assert row["FST_bucket"] == "unavailable"


def test_summary_is_grouped_by_dataset_and_ignores_quarantined_ita(tmp_path):
    first_manifest, masks_path = _write_fixture(tmp_path, include_stats=False)
    second_image = tmp_path / "images" / "ph2" / "image-b.png"
    second_image.parent.mkdir()
    Image.new("RGB", (8, 8), (120, 90, 70)).save(second_image)
    first_manifest.write_text(
        "index,dataset,image_id,path,split\n"
        f"0,ham10000,img-a,{tmp_path / 'images' / 'ham' / 'image-a.png'},train\n"
        f"1,ph2,img-b,{second_image},val\n",
        encoding="utf-8",
    )
    masks = np.load(masks_path)
    np.save(masks_path, np.concatenate([masks, masks], axis=0))

    output = fpm.run(
        fpm.parse_args(
            [
                "--manifest",
                str(first_manifest),
                "--pseudo-masks",
                str(masks_path),
                "--output-json",
                str(tmp_path / "fst_manifest.json"),
            ]
        )
    )

    assert set(output["summary"]["by_dataset"]) == {"ham10000", "ph2"}
    assert output["summary"]["by_dataset"]["ham10000"]["ITA"]["n"] == 1
    assert output["summary"]["by_dataset"]["ph2"]["FST_distribution"]


def test_unknown_pseudo_mask_label_is_rejected(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    masks = np.load(masks_path)
    masks[0, 0, 0] = 7
    np.save(masks_path, masks)

    with pytest.raises(ValueError, match="only 0, 1 and 255"):
        fpm.run(
            fpm.parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--pseudo-masks",
                    str(masks_path),
                    "--output-json",
                    str(tmp_path / "fst_manifest.json"),
                ]
            )
        )


def test_parse_args_exposes_standalone_cli_contract():
    args = fpm.parse_args(
        ["--manifest", "manifest.csv", "--pseudo-masks", "masks.npy", "--output-json", "out.json"]
    )

    assert args.manifest.name == "manifest.csv"
    assert args.pseudo_masks.name == "masks.npy"
    assert args.output_json.name == "out.json"


def _write_field(tmp_path, valid_rows=6):
    field = np.zeros((1, 8, 8), dtype=np.uint8)
    field[0, :valid_rows] = 1
    path = tmp_path / "field.npy"
    np.save(path, field)
    return path


def _run_with_field(tmp_path, manifest_path, masks_path, field_path=None):
    argv = [
        "--manifest",
        str(manifest_path),
        "--pseudo-masks",
        str(masks_path),
        "--output-json",
        str(tmp_path / "fst_manifest.json"),
    ]
    if field_path is not None:
        argv += ["--field", str(field_path)]
    return fpm.run(fpm.parse_args(argv))


def test_the_field_is_excluded_from_the_ita_and_reported(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    image_path = tmp_path / "images" / "ham" / "image-a.png"
    image = np.full((8, 8, 3), (180, 130, 100), dtype=np.uint8)
    image[6:] = (40, 40, 40)  # grey vignette rows, well above the black threshold
    Image.fromarray(image).save(image_path)
    field_path = _write_field(tmp_path)

    output = _run_with_field(tmp_path, manifest_path, masks_path, field_path)
    pseudo = np.load(masks_path)[0]
    field = np.load(field_path)[0].astype(bool)
    expected = fitzpatrick.estimate_fst(
        image.astype(np.float32) / 255.0,
        mask=pseudo == 1,
        border_mask=(pseudo == 255) | ~field,
    )

    assert output["field_used"] is True
    assert output["records"][0]["ITA"] == expected.ita


def test_without_the_field_flag_the_manifest_says_so(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    output = _run_with_field(tmp_path, manifest_path, masks_path)
    assert output["field_used"] is False


def test_a_field_shorter_than_the_masks_is_rejected(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    field_path = tmp_path / "field.npy"
    np.save(field_path, np.ones((1, 4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="field"):
        _run_with_field(tmp_path, manifest_path, masks_path, field_path)


def _skin_gradient(height=40, width=24):
    ramp = np.linspace(150.0, 210.0, height).reshape(height, 1)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = ramp
    image[..., 1] = ramp * 0.72
    image[..., 2] = ramp * 0.55
    return image


def test_an_image_larger_than_the_mask_is_resized_before_estimating(tmp_path):
    """Real frames are 600x450 while the masks live at the cache resolution."""
    manifest_path, masks_path = _write_fixture(tmp_path)
    image_path = tmp_path / "images" / "ham" / "image-a.png"
    large = _skin_gradient()
    Image.fromarray(large).save(image_path)

    output = _run_with_field(tmp_path, manifest_path, masks_path)

    pseudo = np.load(masks_path)[0]
    resized = Image.fromarray(large).resize((8, 8), Image.BILINEAR)
    expected = fitzpatrick.estimate_fst(
        np.asarray(resized, dtype=np.float32) / 255.0,
        mask=pseudo == 1,
        border_mask=pseudo == 255,
    )
    assert expected.status == "ok"
    assert output["records"][0]["ITA"] == expected.ita


def test_the_field_still_applies_when_the_image_is_resized(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    Image.fromarray(_skin_gradient()).save(tmp_path / "images" / "ham" / "image-a.png")
    field_path = _write_field(tmp_path)

    output = _run_with_field(tmp_path, manifest_path, masks_path, field_path)

    assert output["field_used"] is True
    assert output["records"][0]["ITA"] is not None


def _csv_rows(path):
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def test_output_csv_writes_one_bucket_per_image(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    csv_path = tmp_path / "fst.csv"
    output = fpm.run(
        fpm.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--pseudo-masks",
                str(masks_path),
                "--output-json",
                str(tmp_path / "fst_manifest.json"),
                "--output-csv",
                str(csv_path),
            ]
        )
    )

    rows = _csv_rows(csv_path)
    assert list(rows[0])[:2] == ["image_id", "fst_bucket"]
    assert [row["image_id"] for row in rows] == ["img-a"]
    assert rows[0]["fst_bucket"] == str(output["records"][0]["FST_bucket"])
    assert rows[0]["fst_bucket"] in {"1", "2", "3", "4", "5", "6"}


def test_output_csv_marks_quarantined_images_as_unavailable(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    masks = np.load(masks_path)
    masks[:] = 0
    np.save(masks_path, masks)
    csv_path = tmp_path / "fst.csv"
    fpm.run(
        fpm.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--pseudo-masks",
                str(masks_path),
                "--output-json",
                str(tmp_path / "fst_manifest.json"),
                "--output-csv",
                str(csv_path),
            ]
        )
    )

    assert _csv_rows(csv_path)[0]["fst_bucket"] == "unavailable"


def test_without_output_csv_no_bucket_table_is_written(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    args = fpm.parse_args(
        [
            "--manifest",
            str(manifest_path),
            "--pseudo-masks",
            str(masks_path),
            "--output-json",
            str(tmp_path / "fst_manifest.json"),
        ]
    )
    assert args.output_csv is None
    fpm.run(args)
    assert not (tmp_path / "fst.csv").exists()


# ------------------------------------------------ skin medians and tone terciles
def _write_three_tone_fixture(tmp_path):
    """Three skin tones (dark, medium, light) sharing one lesion geometry."""
    tones = {"img-dark": (120, 80, 60), "img-mid": (180, 130, 100), "img-light": (235, 200, 180)}
    lines = ["index,dataset,image_id,path,split"]
    for index, (image_id, rgb) in enumerate(tones.items()):
        image_path = tmp_path / "images" / f"{image_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), rgb).save(image_path)
        lines.append(f"{index},ham10000,{image_id},{image_path},train")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    masks = np.zeros((3, 8, 8), dtype=np.uint8)
    masks[:, 2:5, 2:5] = 1
    masks_path = tmp_path / "pseudo_masks.npy"
    np.save(masks_path, masks)
    return manifest_path, masks_path


def _run(tmp_path, manifest_path, masks_path, *extra):
    return fpm.run(
        fpm.parse_args(
            ["--manifest", str(manifest_path), "--pseudo-masks", str(masks_path),
             "--output-json", str(tmp_path / "fst_manifest.json"), *extra]
        )
    )


def test_records_carry_the_skin_medians_and_the_pixel_count(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    row = _run(tmp_path, manifest_path, masks_path)["records"][0]
    assert isinstance(row["L_median"], float) and 0 < row["L_median"] < 100
    assert isinstance(row["b_median"], float)
    assert row["n_skin_px"] == 64 - 9 - 2


def test_an_indeterminate_ita_keeps_its_status_and_medians_without_a_bucket(tmp_path):
    manifest_path, masks_path = _write_fixture(tmp_path)
    Image.new("RGB", (8, 8), (150, 155, 190)).save(tmp_path / "images" / "ham" / "image-a.png")
    row = _run(tmp_path, manifest_path, masks_path)["records"][0]
    assert row["status"] == "ita_indeterminate"
    assert row["FST_bucket"] == "unavailable"
    assert row["ITA"] is None
    assert row["b_median"] < 0
    assert row["quality"] == "invalid_estimate:ita_indeterminate"


def test_tone_terciles_split_the_pool_by_skin_lightness(tmp_path):
    manifest_path, masks_path = _write_three_tone_fixture(tmp_path)
    output = _run(tmp_path, manifest_path, masks_path)
    terciles = {row["image_id"]: row["tone_tercile"] for row in output["records"]}
    assert terciles == {"img-dark": "dark", "img-mid": "medium", "img-light": "light"}
    cuts = output["summary"]["tone_tercile_cuts"]
    assert cuts["low"] < cuts["high"]
    assert output["summary"]["by_dataset"]["ham10000"]["tone_distribution"] == {
        "dark": 1, "medium": 1, "light": 1
    }


def test_tone_terciles_still_cover_images_whose_ita_is_indeterminate(tmp_path):
    manifest_path, masks_path = _write_three_tone_fixture(tmp_path)
    Image.new("RGB", (8, 8), (150, 155, 190)).save(tmp_path / "images" / "img-mid.png")
    output = _run(tmp_path, manifest_path, masks_path)
    rows = {row["image_id"]: row for row in output["records"]}
    assert rows["img-mid"]["FST_bucket"] == "unavailable"
    assert rows["img-mid"]["tone_tercile"] in {"dark", "medium", "light"}


def test_images_without_skin_get_no_tone_tercile(tmp_path):
    manifest_path, masks_path = _write_three_tone_fixture(tmp_path)
    masks = np.load(masks_path)
    masks[1] = 0
    np.save(masks_path, masks)
    output = _run(tmp_path, manifest_path, masks_path)
    rows = {row["image_id"]: row for row in output["records"]}
    assert rows["img-mid"]["tone_tercile"] == "unavailable"
    assert rows["img-mid"]["n_skin_px"] == 0
    assert output["summary"]["tone_tercile_cuts"]["n"] == 2


def test_output_csv_carries_the_tone_and_the_medians(tmp_path):
    manifest_path, masks_path = _write_three_tone_fixture(tmp_path)
    csv_path = tmp_path / "fst.csv"
    _run(tmp_path, manifest_path, masks_path, "--output-csv", str(csv_path))
    rows = _csv_rows(csv_path)
    assert list(rows[0]) == [
        "image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita", "n_skin_px"
    ]
    assert [row["tone_tercile"] for row in rows] == ["dark", "medium", "light"]
    assert all(row["n_skin_px"] == "55" for row in rows)
