"""scripts/pad_target_split.py: reads a manifest, writes the split, never overwrites it."""

from __future__ import annotations

import json

from dermssl import pad_target_split as ts
from scripts import pad_target_split as script


def _manifest(tmp_path):
    rows = [{"img_id": f"PAT_{p}_{i}_{p}{i}", "patient_id": f"PAT_{p}", "lesion_id": str(i),
             "diagnostic": "BCC" if p % 2 else "NEV", "path": f"p/{p}{i}.png"}
            for p in range(10) for i in range(2)]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"images": rows}), encoding="utf-8")
    return path


def _run(tmp_path, *extra):
    argv = ["--manifest", str(_manifest(tmp_path)), "--pad-root", str(tmp_path),
            "--out", str(tmp_path / "split.json"), *extra]
    return script.main(argv, hash_of=lambda row: row["img_id"])


def test_it_writes_a_split_pinned_to_the_manifest(tmp_path):
    assert _run(tmp_path) == 0
    doc = json.loads((tmp_path / "split.json").read_text())
    assert doc["target_adapt"]["n_images"] + doc["target_test"]["n_images"] == 20
    assert doc["sha256"]["target_test"] == ts.ids_sha256(doc["target_test"]["img_ids"])
    assert len(doc["manifest_sha256"]) == 64


def test_it_does_not_overwrite_an_existing_split(tmp_path, capsys):
    (tmp_path / "split.json").write_text("{}", encoding="utf-8")
    assert _run(tmp_path) == 1
    assert "ya existe" in capsys.readouterr().err
    assert (tmp_path / "split.json").read_text() == "{}"


def test_a_bad_fraction_aborts_without_writing(tmp_path, capsys):
    assert _run(tmp_path, "--adapt-fraction", "1.0") == 1
    assert not (tmp_path / "split.json").exists()
    assert "ABORT" in capsys.readouterr().err


def test_the_declared_fraction_assumption_is_in_the_script_docstring():
    assert "ASSUMPTION" in script.__doc__ and "50 %" in script.__doc__
