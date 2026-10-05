"""FITZPATRICK.md is generated from the E5 JSON and checked for freshness; every number comes from the JSON."""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_fitzpatrick_table as bt  # noqa: E402

from dermssl import e5_analyze as an  # noqa: E402
from dermssl import e5_document as doc  # noqa: E402
from dermssl import e5_table as tab  # noqa: E402

POINT = re.compile(r"(?<![\w./])\d+\.\d{4}(?![\w/])")


def _est(bucket, L):
    return {"status": "ok" if bucket else "no_skin", "bucket": bucket, "ita": 10.0 if bucket else None,
            "L_median": L, "b_median": 15.0, "n_skin_px": 9}


def _record(i, roi_bucket=None):
    roi = {"status": "ok" if roi_bucket else "no_mask", "bucket": roi_bucket, "ita": None, "L_median": None,
           "b_median": None, "n_px": 0, "ring_px": 0, "valid_fraction": None}
    bucket = ("I", "II", "III", None)[i % 4]
    return {"image_id": f"i{i}", "split": "train" if i < 8 else "val", "label": "mel" if i % 3 == 0 else "nv",
            "otsu": _est(bucket, 40.0 + i), "whole_image": _est("II", 40.0 + i), "roi": roi}


def _document(name="ham10000", masks=True, n=12):
    tasks = [an.Task(f"i{i}", f"/x/{i}.jpg", "train" if i < 8 else "val", "mel" if i % 3 == 0 else "nv",
                     i if masks else None) for i in range(n)]
    spec = doc.DatasetSpec(name, ("mel", "nv"), ("train", "val"), tasks, {"split": {"n_train": 8}},
                           {"masks_sha256": "ab", "producer": "student_r3"} if masks else None)
    records = [_record(i, "II" if masks and i % 2 else None) for i in range(n)]
    return doc.build_document(spec, records, code_revision="abc1234", smoke=None,
                              hair_detector={"KERNEL": 9} if masks else None)


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "e5"
    root.mkdir()
    for name, masks in (("ham10000", True), ("isic2019", False)):
        (root / f"e5_fitzpatrick_{name}.json").write_text(doc.dump_document(_document(name, masks)), encoding="utf-8")
    return root


def test_with_no_json_every_dataset_is_pending_and_no_number_is_invented(tmp_path):
    text = tab.render({})
    assert text.count("pendiente") >= 3 and "HAM10000" in text and "ISIC 2020" in text
    assert not re.search(r"\| \d", text)


def test_the_glossary_and_the_caveats_come_before_any_table(inputs):
    text = tab.render(tab.load_documents(inputs))
    assert text.index("## Glosario") < text.index("## Cómo leer") < text.index("## HAM10000")
    assert text.index("## Glosario") < text.index("|")
    for needle in ("NO está validado", "22,9 %", "no es tono de piel", "relativos a cada dataset", "ITA"):
        assert needle in text.split("## HAM10000")[0], needle


def test_counts_in_the_tables_are_the_counts_of_the_json(inputs):
    d = json.loads((inputs / "e5_fitzpatrick_ham10000.json").read_text(encoding="utf-8"))
    text = tab.render(tab.load_documents(inputs))
    otsu = d["report"]["estimators"]["otsu"]
    row = "| " + " | ".join(["mel", *(str(otsu["class_x_bucket"]["mel"][b]) for b in
                                      ("I", "II", "III", "IV", "V", "VI", "abstain"))]) + " | "
    assert row in text
    assert f"{d['report']['n_images']}" in text and d["code_revision"] in text


def test_changing_a_count_in_the_json_changes_the_table(inputs):
    path = inputs / "e5_fitzpatrick_ham10000.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    d["report"]["estimators"]["otsu"]["class_x_bucket"]["mel"]["I"] += 7
    path.write_text(doc.dump_document(d), encoding="utf-8")
    before = tab.render({"ham10000": json.loads((inputs / "e5_fitzpatrick_isic2019.json").read_text())})
    assert tab.render(tab.load_documents(inputs)) != before


def test_a_dataset_without_masks_says_the_ring_does_not_apply_and_shows_no_ring_table(inputs):
    text = tab.render(tab.load_documents(inputs))
    isic = text.split("## ISIC 2019")[1].split("## ISIC 2020")[0]
    assert "no aplica" in isic and "Otsu × anillo" not in isic
    assert "Otsu × anillo" in text.split("## ISIC 2019")[0]


def test_terciles_are_shown_with_their_own_cuts_and_the_relative_warning(inputs):
    text = tab.render(tab.load_documents(inputs))
    assert "Cortes de L\\*" in text and "propios de este dataset" in text


def test_numbers_use_decimal_comma_and_dot_thousands(inputs):
    text = tab.render(tab.load_documents(inputs))
    assert not POINT.findall(text) and re.search(r"\d,\d %", text)
    assert tab.num(10015) == "10.015" and tab.pct(1, 3) == "33,3 %" and tab.pct(0, 0) == "n/d"


def test_a_document_whose_caveat_differs_from_the_code_is_refused(inputs):
    path = inputs / "e5_fitzpatrick_ham10000.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    d["caveat"] = "es un buen estimador"
    path.write_text(doc.dump_document(d), encoding="utf-8")
    with pytest.raises(tab.StaleDocumentError, match="caveat"):
        tab.load_documents(inputs)


def test_a_smoke_or_validated_document_is_refused(inputs):
    path = inputs / "e5_fitzpatrick_ham10000.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    d["smoke_limit"] = 50
    path.write_text(doc.dump_document(d), encoding="utf-8")
    with pytest.raises(tab.StaleDocumentError, match="smoke"):
        tab.load_documents(inputs)


def test_check_detects_missing_stale_and_fresh(inputs, tmp_path):
    out = tmp_path / "FITZPATRICK.md"
    flags = ["--input-dir", str(inputs), "--out", str(out)]
    assert bt.main(["--check", *flags]) == 1  # missing
    assert bt.main(flags) == 0 and bt.main(["--check", *flags]) == 0
    out.write_text(out.read_text(encoding="utf-8") + "\nedited by hand", encoding="utf-8")
    assert bt.main(["--check", *flags]) == 1
    assert bt.main(flags) == 0 and bt.main(["--check", *flags]) == 0


def test_the_committed_table_is_fresh():
    assert bt.main(["--check"]) == 0
