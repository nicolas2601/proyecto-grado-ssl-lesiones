"""JSON-first FST manifest contracts without loading torch or a GPU."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import kaggle_fst_manifest as kfm  # noqa: E402


def test_find_target_images_is_recursive_like_ham10000s_part_1_part_2_layout(tmp_path):
    """HAM10000 reparte las imagenes en subcarpetas (part_1/, part_2/) - un
    glob no recursivo sobre --target-image-root encuentra cero imagenes,
    que es justo el bug que este test evita reintroducir."""
    (tmp_path / "part_1").mkdir()
    (tmp_path / "part_2").mkdir()
    (tmp_path / "part_1" / "img_a.jpg").write_bytes(b"")
    (tmp_path / "part_2" / "img_b.jpg").write_bytes(b"")

    found = kfm.find_target_images(tmp_path)

    assert [p.name for p in found] == ["img_a.jpg", "img_b.jpg"]


def test_build_fst_manifest_record_preserves_mask_qa_and_unavailable_bucket():
    """A quarantined mask must stay traceable and unavailable in JSON."""
    from dermssl.fitzpatrick import FSTEstimate
    from dermssl.mask_manifest import build_mask_record

    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    mask_record = build_mask_record(
        dataset="ham10000",
        image_id="ISIC_1",
        source_revision="v1",
        mask=mask,
        producer="unet",
        version="checkpoint-a",
    )
    whole = FSTEstimate(2, 48.0, 1.0, "ok", False, 1.0, False)
    masked = FSTEstimate(
        None,
        float("nan"),
        0.0,
        "mask_ineligible",
        False,
        0.0,
        False,
        mask_producer="unet",
        mask_version="checkpoint-a",
        mask_sha256=mask_record["mask_sha256"],
    )

    record = kfm.build_fst_manifest_record(
        dataset="ham10000",
        path=Path("ISIC_1.jpg"),
        source_revision="v1",
        whole=whole,
        masked=masked,
        mask_record=mask_record,
    )

    assert record["whole_image"]["bucket"] == 2
    assert record["masked"]["bucket"] == "unavailable"
    assert record["masked"]["mask"]["quality"] == "degenerate"
    assert record["masked"]["mask"]["eligible"] is False
    assert record["masked"]["mask"]["mask_sha256"] == mask_record["mask_sha256"]


def test_build_fst_manifest_record_keeps_accepted_mask_and_measured_bucket():
    from dermssl.fitzpatrick import FSTEstimate
    from dermssl.mask_manifest import build_mask_record

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    mask_record = build_mask_record(
        dataset="isic2020",
        image_id="ISIC_2",
        source_revision="resized-v1",
        mask=mask,
        producer="unet",
        version="checkpoint-b",
    )
    measured = FSTEstimate(4, 22.0, 0.8, "ok", False, 0.9, False)

    record = kfm.build_fst_manifest_record(
        dataset="isic2020",
        path=Path("ISIC_2.jpg"),
        source_revision="resized-v1",
        whole=FSTEstimate(3, 35.0, 1.0, "ok", False, 1.0, False),
        masked=measured,
        mask_record=mask_record,
    )

    assert record["masked"]["bucket"] == 4
    assert record["masked"]["status"] == "ok"
    assert record["masked"]["mask"]["quality"] == "accepted"
    assert record["masked"]["mask"]["mask_pixel_count"] == 9
