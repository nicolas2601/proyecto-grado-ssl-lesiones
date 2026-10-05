"""E5 result document: caveats, definitions, terciles per dataset and the per-image columns."""

import json

import pytest

from dermssl import e5_analyze as an
from dermssl import e5_counts as ec
from dermssl import e5_document as doc
from dermssl import e5_ita as ita
from dermssl import e5_roi as roi
from dermssl import fitzpatrick as fz


def _record(i, split, label, bucket, L, roi_bucket=None, roi_status="no_mask"):
    est = {"status": "ok" if bucket else "no_skin", "bucket": bucket, "ita": 10.0 if bucket else None,
           "L_median": L, "b_median": 15.0, "n_skin_px": 9}
    return {"image_id": f"i{i}", "split": split, "label": label, "otsu": est, "whole_image": dict(est),
            "roi": {"status": roi_status, "bucket": roi_bucket, "ita": None, "L_median": None, "b_median": None,
                    "n_px": 0, "ring_px": 0, "valid_fraction": None}}


def _spec(masks=False):
    tasks = [an.Task(f"i{i}", f"/x/i{i}.jpg", "train" if i < 6 else "val", "mel" if i % 2 else "nv",
                     i if masks else None) for i in range(9)]
    return doc.DatasetSpec(name="demo", classes=("mel", "nv"), splits=("train", "val"), tasks=tasks,
                           provenance={"n_train": 6, "n_val": 3}, masks=None if not masks else {"masks_sha256": "ab"})


def _records(masks=False):
    return [_record(i, "train" if i < 6 else "val", "mel" if i % 2 else "nv", "II", 40.0 + i,
                    "II" if masks else None, "ok" if masks else "no_mask") for i in range(9)]


def _build(masks=False, smoke=None):
    return doc.build_document(_spec(masks), _records(masks), code_revision="abc", smoke=smoke,
                              hair_detector={"KERNEL": 9} if masks else None)


def test_the_document_is_json_safe_and_says_it_is_descriptive_and_not_validated():
    d = _build()
    json.dumps(d, allow_nan=False)
    assert d["descriptive"] is True and d["validated"] is False
    for needle in ("NO está validado", "22,9 %", "PMC13459198", "no es tono de piel", "relativos a cada dataset"):
        assert needle in d["caveat"], needle


def test_terciles_are_cut_on_this_datasets_own_L_and_written_per_image():
    d = _build()
    cols = d["per_image"]["columns"]
    rows = d["per_image"]["rows"]
    tercile = [r[cols.index("tercile")] for r in rows]
    assert tercile == ["dark"] * 3 + ["medium"] * 3 + ["light"] * 3
    assert d["report"]["terciles"]["cuts"]["n"] == 9


def test_the_per_image_table_is_columnar_with_one_row_per_image_in_task_order():
    d = _build(masks=True)
    cols, rows = d["per_image"]["columns"], d["per_image"]["rows"]
    assert cols[:3] == ["image_id", "split", "label"] and len(rows) == 9 and all(len(r) == len(cols) for r in rows)
    assert [r[0] for r in rows] == [f"i{i}" for i in range(9)]
    assert rows[0][cols.index("otsu_bucket")] == "II" and rows[0][cols.index("roi_bucket")] == "II"


def test_floats_are_rounded_so_the_file_does_not_depend_on_the_last_digits():
    rec = _record(0, "train", "mel", "II", 41.123456789)
    rec["otsu"]["ita"], rec["tercile"] = 12.3456789, "dark"
    row = doc.per_image_rows([rec])[1][0]
    assert 41.1235 in row and 12.3457 in row


def test_without_masks_the_roi_estimator_is_declared_not_applicable_in_the_definitions_and_report():
    d = _build()
    assert d["estimators"]["roi"]["applicable"] is False
    assert "máscara" in d["estimators"]["roi"]["reason"]
    assert d["report"]["estimators"]["roi"]["applicable"] is False
    assert d["masks"] is None


def test_with_masks_the_ring_definition_carries_its_constants_and_the_mask_provenance():
    d = _build(masks=True)
    r = d["estimators"]["roi"]
    assert r["applicable"] is True and r["hair_detector"] == {"KERNEL": 9}
    assert r["saturation_level"] == roi.SATURATION_LEVEL and r["min_valid_px"] == roi.MIN_RING_VALID_PX
    assert r["min_valid_fraction"] == roi.MIN_RING_VALID_FRACTION and r["abstains_on"] == list(roi.STATUSES[1:])
    assert d["masks"] == {"masks_sha256": "ab"}


def test_with_masks_but_no_hair_detector_the_document_cannot_be_built():
    with pytest.raises(doc.MissingHairDetector):
        doc.build_document(_spec(True), _records(True), code_revision="abc", smoke=None, hair_detector=None)


def test_definitions_come_from_the_estimator_modules_not_from_typed_numbers():
    d = _build()
    assert d["fst_thresholds"] == list(fz.FST_THRESHOLDS) and d["min_b_for_ita"] == fz.MIN_B_FOR_ITA
    assert d["canonical_input"]["short_side"] == ita.CANONICAL_SHORT_SIDE
    assert "draft" in d["canonical_input"]["decode"] and d["abstention"]["min_skin_fraction"] == fz.MIN_SKIN_FRACTION


def test_a_smoke_run_is_marked_and_a_real_one_is_not():
    assert _build(smoke=50)["smoke_limit"] == 50 and _build()["smoke_limit"] is None


def test_the_document_is_deterministic():
    assert json.dumps(_build(), sort_keys=True) == json.dumps(_build(), sort_keys=True)


def test_unreadable_images_are_listed_in_the_document():
    recs = _records()
    recs[2] = ec.unreadable_record("i2", "train", "nv", "cannot decode /x/i2.jpg")
    d = doc.build_document(_spec(), recs, code_revision="abc", smoke=None, hair_detector=None)
    assert d["report"]["n_unreadable"] == 1 and d["unreadable"] == [{"image_id": "i2", "error": "cannot decode /x/i2.jpg"}]
