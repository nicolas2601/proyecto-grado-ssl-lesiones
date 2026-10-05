"""E5 aggregation: class x bucket, bucket x split, L* terciles and the ROI-vs-Otsu agreement."""

import json

import pytest

from dermssl import e5_counts as ec
from dermssl import e5_ita as ita

CLASSES = ("mel", "nv")


def _est(bucket, status=None, L=60.0):
    return {"status": status or ("ok" if bucket else "no_skin"), "bucket": bucket, "ita": None if bucket is None else 20.0, "L_median": L,
            "b_median": 15.0, "n_skin_px": 100}


def _roi(bucket, status="ok"):
    return {"status": status, "bucket": bucket, "ita": None, "L_median": None, "b_median": None, "n_px": 0,
            "ring_px": 0, "valid_fraction": None}


def _rec(i, split, label, otsu, whole=None, roi=None, tercile="medium"):
    return {"image_id": f"i{i}", "split": split, "label": label, "otsu": _est(otsu),
            "whole_image": _est(otsu if whole is None else whole),
            "roi": roi if roi is not None else _roi(None, "no_mask"), "tercile": tercile}


RECORDS = [
    _rec(0, "train", "mel", "I", tercile="light"), _rec(1, "train", "mel", "II", tercile="medium"),
    _rec(2, "train", "nv", "II", tercile="medium"), _rec(3, "val", "nv", None, tercile="unavailable"),
    _rec(4, "val", "nv", "VI", tercile="dark"), _rec(5, "val", "mel", "I", tercile="light"),
]


def _report(records=RECORDS, masks=False):
    return ec.build_report(records, classes=CLASSES, splits=("train", "val"),
                           tercile_cuts={"low": 1.0, "high": 2.0, "n": 5}, roi_applicable=masks,
                           roi_reason="" if masks else "this dataset has no lesion masks")


def test_class_by_bucket_has_every_column_with_zeros_and_an_abstain_column():
    table = _report()["estimators"]["otsu"]["class_x_bucket"]
    assert list(table["mel"]) == [*ita.BUCKET_LABELS, "abstain"]
    assert table["mel"] == {"I": 2, "II": 1, "III": 0, "IV": 0, "V": 0, "VI": 0, "abstain": 0}
    assert table["nv"] == {"I": 0, "II": 1, "III": 0, "IV": 0, "V": 0, "VI": 1, "abstain": 1}


def test_bucket_by_split_and_bucket_totals_add_up_to_the_images():
    est = _report()["estimators"]["otsu"]
    assert est["bucket_x_split"]["train"]["II"] == 2 and est["bucket_x_split"]["val"]["abstain"] == 1
    assert sum(est["bucket_counts"].values()) == len(RECORDS)
    assert est["status_counts"] == {"no_skin": 1, "ok": 5}


def test_terciles_report_cuts_counts_class_and_split_tables():
    t = _report()["terciles"]
    assert t["cuts"] == {"low": 1.0, "high": 2.0, "n": 5}
    assert t["counts"] == {"dark": 1, "medium": 2, "light": 2, "unavailable": 1}
    assert t["class_x_tercile"]["mel"] == {"dark": 0, "medium": 1, "light": 2, "unavailable": 0}
    assert t["tercile_x_split"]["val"] == {"dark": 1, "medium": 0, "light": 1, "unavailable": 1}


def test_a_dataset_without_masks_declares_the_roi_estimator_not_applicable_and_invents_no_table():
    roi = _report()["estimators"]["roi"]
    assert roi["applicable"] is False and "no lesion masks" in roi["reason"]
    assert roi["status_counts"] == {"no_mask": 6}
    assert "class_x_bucket" not in roi and "agreement_with_otsu" not in roi


def test_with_masks_the_roi_tables_and_the_agreement_with_otsu_exist():
    recs = [_rec(0, "train", "mel", "I", roi=_roi("I")), _rec(1, "train", "mel", "II", roi=_roi("I")),
            _rec(2, "val", "nv", "II", roi=_roi(None, "ring_too_small")), _rec(3, "val", "nv", None, roi=_roi("II"))]
    roi = _report(recs, masks=True)["estimators"]["roi"]
    assert roi["applicable"] is True and roi["reason"] == ""
    assert roi["status_counts"] == {"ok": 3, "ring_too_small": 1}
    assert roi["class_x_bucket"]["mel"]["I"] == 2 and roi["class_x_bucket"]["nv"]["abstain"] == 1
    agree = roi["agreement_with_otsu"]
    assert agree["I"]["I"] == 1 and agree["II"]["I"] == 1 and agree["II"]["abstain"] == 1
    assert agree["abstain"]["II"] == 1
    assert roi["both_estimated"] == 2 and roi["same_bucket"] == 1


def test_an_unknown_class_or_split_is_a_typed_error_not_a_silent_new_row():
    bad = [_rec(0, "test", "mel", "I")]
    with pytest.raises(ec.UnknownKeyError, match="test"):
        _report(bad)
    with pytest.raises(ec.UnknownKeyError, match="akiec"):
        _report([_rec(0, "train", "akiec", "I")])


def test_the_report_is_json_safe():
    json.dumps(_report(), allow_nan=False)


def test_unreadable_records_count_as_abstentions_with_their_own_status():
    rec = ec.unreadable_record("x", "train", "mel", "cannot decode x")
    est = _report([*RECORDS, rec])["estimators"]["otsu"]
    assert est["status_counts"]["unreadable"] == 1 and est["bucket_counts"]["abstain"] == 2
    assert _report([*RECORDS, rec])["n_unreadable"] == 1


def _pool_rec(i, split, estimator, state, bucket, sha_ok=None, label="mel"):
    roi = _roi(bucket, "ok" if bucket else "ring_too_small")
    roi.update({"estimator": estimator, "mask_state": state, "roi_image": "pool_224", "mask_sha_ok": sha_ok})
    return _rec(i, split, label, "II", roi=roi)


POOL_RECORDS = [
    _pool_rec(0, "train", "perilesional_roi", "eligible", "I", True), _pool_rec(1, "train", "perilesional_roi", "eligible", "II", False),
    _pool_rec(2, "train", "no_mask_border", "quarantine", "II"), _pool_rec(3, "val", "no_mask_border", "absent", None),
    _pool_rec(4, "val", "perilesional_roi", "eligible", "I", True, label="nv"),
]


def _pool_report():
    return _report(POOL_RECORDS, masks=True)["estimators"]["roi"]


def test_mask_coverage_is_reported_per_split_and_in_total_with_every_state():
    cov = _pool_report()["mask_coverage"]
    assert list(cov["total"]) == ["c1_native", "eligible", "quarantine", "absent", "none"]
    assert cov["by_split"]["train"] == {"c1_native": 0, "eligible": 2, "quarantine": 1, "absent": 0, "none": 0}
    assert cov["by_split"]["val"] == {"c1_native": 0, "eligible": 1, "quarantine": 0, "absent": 1, "none": 0}
    assert cov["total"]["eligible"] == 3 and sum(cov["total"].values()) == 5


def test_the_two_roi_estimators_are_counted_apart_by_bucket_and_split():
    roi = _pool_report()
    assert roi["by_estimator"]["perilesional_roi"]["n"] == 3 and roi["by_estimator"]["no_mask_border"]["n"] == 2
    assert roi["by_estimator"]["no_mask_border"]["bucket_counts"]["abstain"] == 1
    assert roi["by_estimator"]["perilesional_roi"]["bucket_counts"]["I"] == 2
    assert roi["estimator_x_split"]["train"] == {"perilesional_roi": 2, "no_mask_border": 1, "none": 0}


def test_the_mask_file_hash_check_is_counted_and_never_hidden():
    assert _pool_report()["mask_coverage"]["sha256_file"] == {"match": 2, "mismatch": 1, "not_checked": 2}


def test_the_image_origin_of_each_mask_state_is_counted_so_the_source_fallback_is_visible():
    recs = [_pool_rec(0, "train", "perilesional_roi", "eligible", "I"), _pool_rec(1, "train", "no_mask_border", "quarantine", "II"),
            _pool_rec(2, "val", "no_mask_border", "quarantine", "II"), _pool_rec(3, "val", "no_mask_border", "absent", None)]
    recs[1]["roi"]["roi_image"] = "pool_224"
    recs[2]["roi"]["roi_image"] = recs[3]["roi"]["roi_image"] = "source_224"
    origin = _report(recs, masks=True)["estimators"]["roi"]["mask_coverage"]["image_origin_by_state"]
    assert list(origin["quarantine"]) == ["pool_224", "source_224", "canonical", "none"]
    assert origin["quarantine"] == {"pool_224": 1, "source_224": 1, "canonical": 0, "none": 0}
    assert origin["absent"]["source_224"] == 1 and origin["eligible"]["pool_224"] == 1
