"""scripts/pad_ufes_manifest.py on a synthetic three-image PAD, offline.

The published counts are monkeypatched down to the synthetic ones; the check
itself is the real one (tests/test_pad_ufes_metadata.py pins the real numbers).
"""

from __future__ import annotations

import csv
import hashlib
import json

import pytest
from PIL import Image

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_files as pf
from scripts import pad_ufes_manifest as pm

ROWS = [
    ("PAT_1", "10", "BCC", "2"),
    ("PAT_1", "11", "NEV", ""),
    ("PAT_2", "12", "MEL", "4"),
]


@pytest.fixture()
def pad_root(tmp_path, monkeypatch):
    images = tmp_path / "images" / "imgs_part_1"
    images.mkdir(parents=True)
    with (tmp_path / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "lesion_id", "fitspatrick", "diagnostic", "img_id"])
        for i, (patient, lesion, dx, fst) in enumerate(ROWS):
            img_id = f"{patient}_{lesion}_{i}.png"
            writer.writerow([patient, lesion, fst, dx, img_id])
            Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(images / img_id)
    data = (tmp_path / "metadata.csv").read_bytes()
    official = pf.OfficialFile("metadata.csv", "id", len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(pf, "OFFICIAL_FILES", (official,))
    monkeypatch.setattr(pu, "EXPECTED_IMAGES", 3)
    monkeypatch.setattr(pu, "EXPECTED_PATIENTS", 2)
    monkeypatch.setattr(pu, "EXPECTED_LESIONS", 3)
    monkeypatch.setattr(pu, "EXPECTED_DIAGNOSTIC_COUNTS", {"BCC": 1, "NEV": 1, "MEL": 1})
    return tmp_path


def test_the_manifest_records_labels_fst_and_hashes(pad_root):
    out = pad_root / "manifest.json"
    assert pm.main(["--root", str(pad_root), "--out", str(out)]) == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["n_images"] == 3 and manifest["n_patients"] == 2
    assert manifest["dataset_doi"] == pf.DATASET_DOI
    assert manifest["fst"]["by_bin"] == {"I": 0, "II": 1, "III-VI": 1, pu.FST_MISSING: 1}
    first = manifest["images"][0]
    assert first["ham_label"] == "bcc" and first["fst_bin"] == "II"
    assert first["sha256"] == pf.sha256_of(pad_root / first["path"])


def test_it_does_not_overwrite(pad_root):
    out = pad_root / "manifest.json"
    out.write_text("{}", encoding="utf-8")
    assert pm.main(["--root", str(pad_root), "--out", str(out)]) == 1
    assert out.read_text(encoding="utf-8") == "{}"


def test_an_image_that_does_not_decode_aborts(pad_root):
    victim = next((pad_root / "images").rglob("*.png"))
    victim.write_bytes(b"not a png")
    assert pm.main(["--root", str(pad_root), "--out", str(pad_root / "m.json")]) == 1
    assert not (pad_root / "m.json").exists()


def test_counts_that_differ_from_the_article_abort(pad_root, monkeypatch):
    monkeypatch.setattr(pu, "EXPECTED_IMAGES", 2298)
    assert pm.main(["--root", str(pad_root), "--out", str(pad_root / "m.json")]) == 1


def test_a_metadata_that_is_not_the_official_one_aborts(pad_root):
    with (pad_root / "metadata.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert pm.main(["--root", str(pad_root), "--out", str(pad_root / "m.json")]) == 1
