"""The T2 script: guard, tercile extract, document and generated table (docs/27, T2)."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

import pytest

from dermssl import pad_ufes_guard as pg
from scripts import t2_luminance_dice as t2

REPO = Path(__file__).resolve().parents[1]
MANIFEST_COLUMNS = ["image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita",
                    "n_skin_px"]


def _write_per_image(root: Path, run: str, dataset: str, dice: list[float]) -> None:
    path = root / run / f"evaluacion_{dataset}" / "per_image.csv"
    path.parent.mkdir(parents=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "dice", "iou", "hd95"])
        writer.writerows([f"{dataset}_{i}", d, d, 10.0 + i] for i, d in enumerate(dice))


def _fixture_root(tmp_path: Path, n: int = 12) -> tuple[Path, Path]:
    root = tmp_path / "resultados"
    for run in t2.RUNS.values():
        for dataset in t2.DATASETS:
            _write_per_image(root, run, dataset, [0.5 + 0.01 * i for i in range(n)])
    labels = ["dark", "medium", "light", "unavailable"]
    terciles = {f"{d}_{i}": labels[i % 4] for d in t2.DATASETS for i in range(n - 1)}
    extract = tmp_path / "terciles.csv"
    t2.write_terciles(extract, terciles)
    return root, extract


def _write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows([i, "", t, "ok", "70", "10", "40", "100"] for i, t in rows)


def test_the_real_registration_precedes_the_real_script():
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow != "false":
        pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
    assert t2.check_registration(REPO) == t2.REGISTRATION_COMMIT


def test_it_refuses_before_reading_data_when_the_registration_fails(tmp_path, monkeypatch):
    def refuse(repo, **kwargs):
        raise pg.RegistrationError("el bloque registrado cambio")
    monkeypatch.setattr(pg, "check_registration", refuse)
    with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
        t2.main(["--manifest", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "out")])


def test_the_extract_refuses_a_manifest_that_is_not_the_registered_one(tmp_path):
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [("a", "dark")])
    with pytest.raises(ValueError, match="sha256"):
        t2.extract_terciles(manifest, {"a"})


def test_the_extract_keeps_only_the_evaluated_ids(tmp_path, monkeypatch):
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [("a", "dark"), ("b", "light"), ("c", "unavailable")])
    monkeypatch.setattr(t2, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert t2.extract_terciles(manifest, {"a", "c", "z"}) == {"a": "dark", "c": "unavailable"}


def test_the_extract_round_trips(tmp_path):
    path = tmp_path / "t.csv"
    t2.write_terciles(path, {"b": "light", "a": "dark"})
    assert t2.read_terciles(path) == {"a": "dark", "b": "light"}
    assert path.read_text().splitlines()[1].startswith("a,")


def test_the_document_covers_both_segmenters_on_both_datasets(tmp_path):
    root, extract = _fixture_root(tmp_path)
    doc = t2.build_document(root, extract, n_resamples=50, seed=1)
    assert set(doc["datasets"]) == set(t2.DATASETS)
    cell = doc["datasets"]["ham10000"]
    assert cell["coverage"]["not_in_manifest"] == 1 and cell["coverage"]["total"] == 12
    assert set(cell["segmenters"]) == set(t2.RUNS)
    assert cell["segmenters"]["A2"]["dark_minus_rest"]["dice"]["verdict"] == "sin potencia"
    assert doc["params"] == {"n_resamples": 50, "seed": 1, "clear_margin": -0.02, "min_n": 100}
    assert doc["inputs"]["terciles_sha256"] == hashlib.sha256(extract.read_bytes()).hexdigest()


def test_the_segmenters_must_score_the_same_images(tmp_path):
    root, extract = _fixture_root(tmp_path)
    path = root / t2.RUNS["A2"] / "evaluacion_isic2018" / "per_image.csv"
    path.write_text(path.read_text().rsplit("\n", 2)[0] + "\n")
    with pytest.raises(ValueError, match="mismas imagenes"):
        t2.build_document(root, extract, n_resamples=10, seed=1)


def test_the_table_is_rendered_from_the_document_with_decimal_commas(tmp_path):
    root, extract = _fixture_root(tmp_path)
    text = t2.render_markdown(t2.build_document(root, extract, n_resamples=50, seed=1))
    assert "sin potencia" in text and "Experimento de segmentación 3" in text and "Experimento de segmentación 5" in text
    assert "0,5" in text and "0.5" not in text
    assert "no es tono de piel" in text
