"""The camino B cache and pseudo-masks become the inputs of the CIRCLe tone transform."""

import json

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import cache_io
import export_tone_inputs as exporter
from dermssl.pretrain import tone_transforms

RESOLUTION = 16


def _build_cache(tmp_path):
    """Three images in two datasets; the third one gets a degenerate mask."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rng = np.random.default_rng(0)
    stack = rng.integers(60, 200, size=(3, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    records = [
        cache_io.ImageRecord("ham10000", "ISIC_0001", "a/ISIC_0001.jpg"),
        cache_io.ImageRecord("ham10000", "ISIC_0002", "a/ISIC_0002.jpg"),
        cache_io.ImageRecord("isic2020", "ISIC_9000", "b/ISIC_9000.jpg"),
    ]
    cache = cache_io.create_cache(cache_dir / "images.npy", stack.shape, np.uint8)
    cache[:] = stack
    cache.flush()
    cache_io.write_manifest(cache_dir / "manifest.csv", records, ["train"] * 3)

    masks = np.zeros((3, RESOLUTION, RESOLUTION), dtype=np.uint8)
    masks[0, 4:10, 4:10] = 1
    masks[1, 2:12, 2:12] = 1
    masks[1, 0, 0:3] = 255  # ignore band: not lesion
    masks_path = tmp_path / "pseudo_masks.npy"
    np.save(masks_path, masks)
    return cache_dir, masks_path, stack


def _run(tmp_path, *extra):
    cache_dir, masks_path, stack = _build_cache(tmp_path)
    output = tmp_path / "tone_inputs"
    argv = ["--cache-dir", str(cache_dir), "--pseudo-masks", str(masks_path),
            "--output-root", str(output), "--source-revision", "pool-final", *extra]
    summary = exporter.run(exporter.parse_args(argv))
    return output, summary, stack


def test_eligible_images_are_written_losslessly_under_their_dataset(tmp_path):
    output, _, stack = _run(tmp_path)
    written = np.asarray(Image.open(output / "originals" / "ham10000" / "ISIC_0001.png"))
    assert np.array_equal(written, stack[0])
    assert (output / "originals" / "ham10000" / "ISIC_0002.png").is_file()


def test_quarantined_images_are_not_exported_as_originals(tmp_path):
    output, summary, _ = _run(tmp_path)
    assert not (output / "originals" / "isic2020").exists()
    assert not (output / "masks" / "isic2020").exists()
    assert summary["by_dataset"]["isic2020"] == {"total": 1, "eligible": 0, "quarantined": 1}


def test_masks_mirror_the_originals_as_binary_png(tmp_path):
    output, _, _ = _run(tmp_path)
    mask = np.asarray(Image.open(output / "masks" / "ham10000" / "ISIC_0002.png"))
    assert set(np.unique(mask)) == {0, 255}
    assert mask[5, 5] == 255
    assert mask[0, 0] == 0, "the ignore band is not lesion"


def test_the_mask_manifest_lists_every_image_with_its_eligibility(tmp_path):
    output, _, _ = _run(tmp_path)
    payload = json.loads((output / "mask_manifest.json").read_text(encoding="utf-8"))
    by_id = {row["image_id"]: row for row in payload["masks"]}
    assert set(by_id) == {"ISIC_0001", "ISIC_0002", "ISIC_9000"}
    assert by_id["ISIC_0001"]["eligible"] is True
    assert by_id["ISIC_9000"]["eligible"] is False
    assert by_id["ISIC_0001"]["source_revision"] == "pool-final"
    assert by_id["ISIC_0001"]["mask_semantics"] == "true=lesion"


def test_the_manifest_hash_matches_the_png_that_was_written(tmp_path):
    output, _, _ = _run(tmp_path)
    payload = json.loads((output / "mask_manifest.json").read_text(encoding="utf-8"))
    record = next(row for row in payload["masks"] if row["image_id"] == "ISIC_0002")
    png = np.asarray(Image.open(output / "masks" / "ham10000" / "ISIC_0002.png")) > 0
    from dermssl.mask_manifest import mask_sha256

    assert record["mask_sha256"] == mask_sha256(png)


def test_the_summary_reports_the_export_and_is_saved(tmp_path):
    output, summary, _ = _run(tmp_path)
    saved = json.loads((output / "export_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["resolution"] == RESOLUTION
    assert summary["source_revision"] == "pool-final"
    assert summary["eligible"] == 2 and summary["total"] == 3


def test_a_duplicate_image_id_is_rejected(tmp_path):
    cache_dir, masks_path, _ = _build_cache(tmp_path)
    rows = (cache_dir / "manifest.csv").read_text(encoding="utf-8").replace("ISIC_0002", "ISIC_0001")
    (cache_dir / "manifest.csv").write_text(rows, encoding="utf-8")
    args = exporter.parse_args(["--cache-dir", str(cache_dir), "--pseudo-masks", str(masks_path),
                                "--output-root", str(tmp_path / "o"), "--source-revision", "r"])
    with pytest.raises(ValueError, match="ISIC_0001"):
        exporter.run(args)


def test_a_mask_stack_of_the_wrong_length_is_rejected(tmp_path):
    cache_dir, masks_path, _ = _build_cache(tmp_path)
    np.save(masks_path, np.load(masks_path)[:2])
    args = exporter.parse_args(["--cache-dir", str(cache_dir), "--pseudo-masks", str(masks_path),
                                "--output-root", str(tmp_path / "o"), "--source-revision", "r"])
    with pytest.raises(ValueError, match="pseudo_masks"):
        exporter.run(args)


def test_limit_keeps_the_first_rows_only(tmp_path):
    output, summary, _ = _run(tmp_path, "--limit", "1")
    assert summary["total"] == 1
    assert not (output / "originals" / "ham10000" / "ISIC_0002.png").exists()


def test_the_export_feeds_the_tone_transform_end_to_end(tmp_path):
    output, _, _ = _run(tmp_path)
    result = tone_transforms.generate_tone_transformations(
        image_root=output / "originals",
        mask_root=output / "masks",
        mask_manifest_path=output / "mask_manifest.json",
        output_root=tmp_path / "shift2",
        shift_buckets=2,
        source_revision="pool-final",
    )
    assert result["summary"]["transformed"] == 2
    assert (tmp_path / "shift2" / "ham10000" / "ISIC_0001.png").is_file()
