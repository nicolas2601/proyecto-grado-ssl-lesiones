"""scripts/pad_ufes_eval.py: refuses before reading data, and builds the registered document.

No backbone, no real image: the model-facing step is not exercised here, only
what surrounds it (guards, manifest re-hashing, readings on synthetic probabilities).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_files as pf
from dermssl import pad_ufes_guard as pg
from dermssl import pad_ufes_readings as pr
from scripts import pad_ufes_eval as ev

REPO = Path(__file__).resolve().parents[1]
FINALS = ["--plain-s42", "d:a", "--proposed-s42", "d:b", "--plain-s43", "d:c", "--proposed-s43", "d:e"]


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["--manifest", str(tmp_path / "no-manifest.json"), "--pad-root", str(tmp_path / "no-pad"),
            "--ham-root", str(tmp_path / "no-ham"), "--out", str(tmp_path / "out.json"), *FINALS, *extra]


def test_the_real_registration_precedes_the_real_eval_script():
    """The pinned commit registered docs/30 in this repository, before this script existed."""
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow != "false":
        pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
    assert pg.check_registration(REPO) == pg.REGISTRATION_COMMIT


class TestItRefusesBeforeReadingData:
    def test_without_single_evaluation(self, tmp_path):
        with pytest.raises(SystemExit, match="single-evaluation"):
            ev.main(_argv(tmp_path))

    def test_when_the_output_exists(self, tmp_path):
        (tmp_path / "out.json").write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="ya existe"):
            ev.main(_argv(tmp_path, "--single-evaluation"))

    def test_when_the_prediction_is_not_registered_first(self, tmp_path, monkeypatch):
        def refuse(repo):
            raise pg.RegistrationError("el bloque registrado cambio")
        monkeypatch.setattr(pg, "check_registration", refuse)
        # The manifest path does not exist: reaching it would raise FileNotFoundError instead.
        with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
            ev.main(_argv(tmp_path, "--single-evaluation"))

    def test_every_final_is_required(self, tmp_path):
        with pytest.raises(SystemExit):
            ev.parse_args(_argv(tmp_path)[:-2])


def _manifest(tmp_path: Path, n: int = 4) -> Path:
    rows = []
    for i in range(n):
        path = tmp_path / "images" / f"PAT_{i}_{i}_{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([i]) * 8)
        rows.append({"img_id": path.stem, "patient_id": f"PAT_{i // 2}", "ham_label": "bcc",
                     "fst_bin": "II", "path": str(path.relative_to(tmp_path)),
                     "sha256": pf.sha256_of(path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "images": rows}), encoding="utf-8")
    return manifest


class TestLoadPad:
    def test_loads_labels_patients_and_bins(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pu, "EXPECTED_IMAGES", 4)
        pad = ev.load_pad(_manifest(tmp_path), tmp_path)
        assert pad["groups"].tolist() == ["PAT_0", "PAT_0", "PAT_1", "PAT_1"]
        assert set(pad["paths"]) == set(pad["ids"])

    def test_an_image_changed_since_the_manifest_aborts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pu, "EXPECTED_IMAGES", 4)
        manifest = _manifest(tmp_path)
        (tmp_path / "images" / "PAT_2_2_2.png").write_bytes(b"changed")
        with pytest.raises(pf.IntegrityError, match="cambio desde el manifiesto"):
            ev.load_pad(manifest, tmp_path)

    def test_a_manifest_with_other_counts_aborts(self, tmp_path):
        with pytest.raises(pf.IntegrityError, match="2298"):
            ev.load_pad(_manifest(tmp_path), tmp_path)


def test_the_document_has_every_registered_reading_and_the_strata(monkeypatch):
    monkeypatch.setattr(pr, "BOOTSTRAP", {"n_resamples": 20, "rng_seed": 20260923})
    n = 60
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    pad = {"ham_labels": np.array([cycle[i % 5] for i in range(n)]),
           "groups": np.array([f"P{i // 3}" for i in range(n)]),
           "fst_bins": np.array([pu.FST_BINS[i % 4] for i in range(n)])}
    rng = np.random.default_rng(0)
    probas7 = {role: (lambda raw: raw / raw.sum(axis=1, keepdims=True))(rng.random((n, 7)))
               for role in pr.roles()}
    document = ev.build_document(probas7, pad, {"code_revision": "abc"})
    assert set(document["readings"]) == set(pu.READINGS)
    assert document["bootstrap"]["resampling_unit"] == "patient_id"
    assert set(document["fst_strata"]["bins"]) == set(pu.FST_BINS)
    assert all(b["power"] == pr.NO_POWER for b in document["fst_strata"]["bins"].values())
    json.dumps(document)  # everything in it must be serializable as is
