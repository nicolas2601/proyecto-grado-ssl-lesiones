"""JSON mask contracts: a boolean True pixel always denotes lesion."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl import mask_manifest as mm  # noqa: E402


def test_build_mask_record_quarantines_missing_and_degenerate_masks():
    missing = mm.build_mask_record(
        dataset="ham10000",
        image_id="missing",
        source_revision="v1",
        mask=None,
        producer="otsu",
        version="1",
    )
    degenerate = mm.build_mask_record(
        dataset="ham10000",
        image_id="all-lesion",
        source_revision="v1",
        mask=np.ones((10, 10), dtype=bool),
        producer="otsu",
        version="1",
    )

    assert missing["eligible"] is False
    assert missing["quality"] == "missing"
    assert missing["quarantine_reason"] == "missing_mask"
    assert degenerate["eligible"] is False
    assert degenerate["quality"] == "degenerate"
    assert degenerate["area_fraction"] == 1.0


def test_build_mask_record_records_true_as_lesion_and_provenance():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    record = mm.build_mask_record(
        dataset="isic2020",
        image_id="lesion",
        source_revision="resized-v1",
        mask=mask,
        producer="unet",
        version="checkpoint-a",
        fallback_used=False,
    )

    assert record["eligible"] is True
    assert record["quality"] == "accepted"
    assert record["mask_semantics"] == "true=lesion"
    assert record["area_fraction"] == 0.09
    assert len(record["mask_sha256"]) == 64


def test_build_mask_record_quarantines_low_confidence_mask_despite_valid_area():
    """Una mascara de area valida pero con baja confianza del U-Net no debe
    ser elegible - el area sola no alcanza para confiar en la segmentacion.
    Antes de este fix, `confidence` de infer_mask_with_confidence solo
    llegaba al CSV de auditoria y nunca afectaba `eligible`."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    record = mm.build_mask_record(
        dataset="ham10000",
        image_id="dudosa",
        source_revision="v1",
        mask=mask,
        producer="unet",
        version="checkpoint-a",
        confidence=0.2,
    )

    assert record["eligible"] is False
    assert record["quality"] == "degenerate"
    assert record["quarantine_reason"] == "low_confidence"


def test_build_mask_record_keeps_high_confidence_mask_eligible():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    record = mm.build_mask_record(
        dataset="ham10000",
        image_id="segura",
        source_revision="v1",
        mask=mask,
        producer="unet",
        version="checkpoint-a",
        confidence=0.9,
    )

    assert record["eligible"] is True
    assert record["quality"] == "accepted"


def test_build_mask_record_without_confidence_keeps_prior_area_only_behavior():
    """Compatibilidad hacia atras: el fallback de Otsu no reporta
    confianza calibrada, y no debe quedar quarantineado por eso."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    record = mm.build_mask_record(
        dataset="ham10000",
        image_id="otsu",
        source_revision="v1",
        mask=mask,
        producer="otsu",
        version="1",
    )

    assert record["eligible"] is True


def test_build_mask_record_records_pixel_count_and_hash_changes_for_one_pixel():
    """QA provenance must make a one-pixel mask mutation observable."""
    baseline = np.zeros((10, 10), dtype=bool)
    baseline[2:5, 2:5] = True
    changed = baseline.copy()
    changed[5, 5] = True

    common = {
        "dataset": "isic2020",
        "image_id": "lesion",
        "source_revision": "resized-v1",
        "producer": "unet",
        "version": "checkpoint-a",
    }
    baseline_record = mm.build_mask_record(mask=baseline, **common)
    changed_record = mm.build_mask_record(mask=changed, **common)

    assert baseline_record["mask_pixel_count"] == 9
    assert changed_record["mask_pixel_count"] == 10
    assert changed_record["mask_sha256"] != baseline_record["mask_sha256"]
