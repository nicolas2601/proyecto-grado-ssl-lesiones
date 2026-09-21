"""TDD contract for the Camino B pseudo-mask -> ITA/FST adapter."""

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
        valid_pixel_mask=pseudo != 255,
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
