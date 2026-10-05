"""The T6 runner: guards, Dice curves with the 0.5 sanity check, document and table (docs/27, T6)."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
from dermssl import c1_split_identity as si
from dermssl import luminance_threshold as lt
from dermssl import pad_ufes_guard as pg
from scripts import t6_luminance_threshold as t6

REPO = Path(__file__).resolve().parents[1]
SIZE = 16
MANIFEST_COLUMNS = ["image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita",
                    "n_skin_px"]


def test_the_real_registration_precedes_the_real_script():
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow != "false":
        pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
    assert t6.check_registration(REPO) == t6.REGISTRATION_COMMIT


def test_it_refuses_before_reading_data_when_the_registration_fails(tmp_path, monkeypatch):
    def refuse(repo, **kwargs):
        raise pg.RegistrationError("el bloque registrado cambio")
    monkeypatch.setattr(pg, "check_registration", refuse)
    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
        t6.main(["--manifest", missing, "--ham-root", missing, "--probs-ham", missing,
                 "--probs-isic", missing, "--ham-mask-dir", missing, "--isic-mask-dir", missing,
                 "--out", str(tmp_path / "out")])


def _write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows([i, "", t, "ok", "70", "10", "40", "100"] for i, t in rows)


def test_the_terciles_come_only_from_the_registered_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [("a", "dark"), ("b", "unavailable")])
    with pytest.raises(ValueError, match="sha256"):
        t6.extract_terciles(manifest, {"a"})
    monkeypatch.setattr(t6, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert t6.extract_terciles(manifest, {"a", "b", "z"}) == {"a": "dark", "b": "unavailable"}


def test_the_split_must_be_the_c1_v7_one(monkeypatch):
    split = {"train_ids": ["t1", "t2"], "val_ids": ["v2", "v1"]}
    monkeypatch.setattr(t6, "_rebuild_split", lambda root: split)
    with pytest.raises(ValueError, match="6ff4c17f"):
        t6.split_ids(Path("ham"))
    monkeypatch.setattr(t6, "VAL_IDS_SHA256", si.ids_sha256(["v2", "v1"]))
    assert t6.split_ids(Path("ham")) == (["t1", "t2"], ["v2", "v1"])


def _probabilities(tmp_path: Path, name: str, maps: dict[str, np.ndarray]) -> tuple[Path, Path]:
    probs, masks = tmp_path / f"{name}_probs", tmp_path / f"{name}_gt"
    probs.mkdir()
    masks.mkdir()
    np.save(probs / "probabilities.npy", np.stack(list(maps.values())).astype(np.float32))
    (probs / "ids.txt").write_text("\n".join(maps) + "\n", encoding="utf-8")
    truth = np.zeros((SIZE, SIZE), np.uint8)
    truth[4:12, 4:12] = 255
    for image_id in maps:
        Image.fromarray(truth).save(masks / f"{image_id}_segmentation.png")
    return probs, masks


def _lesion_map(inside: float) -> np.ndarray:
    probability = np.full((SIZE, SIZE), 0.1, np.float32)
    probability[4:12, 4:12] = inside
    return probability


def test_curves_are_computed_only_after_the_dice_at_half_matches_the_published_one(tmp_path):
    probs, masks = _probabilities(tmp_path, "ham", {"a": _lesion_map(0.42), "b": _lesion_map(0.9)})
    reference = {"a": 0.0, "b": 1.0}
    curves = t6.compute_curves(probs, masks, {"a", "b"}, reference, "_segmentation", ".png")
    assert curves["a"][lt.GRID.index(0.4)] == 1.0 and curves["b"][lt.GRID.index(0.5)] == 1.0
    with pytest.raises(ValueError, match="0,5"):
        t6.compute_curves(probs, masks, {"a"}, {"a": 0.5, "b": 1.0}, "_segmentation", ".png")


def test_curves_skip_ids_without_a_manifest_row(tmp_path):
    probs, masks = _probabilities(tmp_path, "isic", {"a": _lesion_map(0.9), "z": _lesion_map(0.9)})
    curves = t6.compute_curves(probs, masks, {"a"}, {"a": 1.0, "z": 1.0}, "_segmentation", ".png")
    assert set(curves) == {"a"}


def _rows(n_fit: int = 120, n_eval: int = 110) -> list[dict]:
    rng = np.random.RandomState(0)
    rows = []
    for part, n in (("fit", n_fit), ("eval", n_eval)):
        for tercile, inside in (("dark", 0.42), ("medium", 0.7), ("light", 0.9)):
            for k in range(n):
                curve = lt.dice_curve(_lesion_map(inside + rng.uniform(-0.03, 0.03)), _truth())
                rows.append({"image_id": f"ham_{part}_{tercile}_{k}", "dataset": "ham10000", "part": part,
                             "tone_tercile": tercile, "curve": curve})
    for tercile in ("dark", "medium", "light", "unavailable"):
        rows.append({"image_id": f"isic_{tercile}", "dataset": "isic2018", "part": "eval",
                     "tone_tercile": tercile, "curve": lt.dice_curve(_lesion_map(0.6), _truth())})
    return rows


def _truth() -> np.ndarray:
    truth = np.zeros((SIZE, SIZE), bool)
    truth[4:12, 4:12] = True
    return truth


def test_the_curves_csv_round_trips(tmp_path):
    path = tmp_path / "curves.csv"
    rows = _rows(n_fit=2, n_eval=1)
    t6.write_curves(path, rows)
    back = t6.read_curves(path)
    assert [r["image_id"] for r in back] == sorted(r["image_id"] for r in rows)
    first = next(r for r in rows if r["image_id"] == back[0]["image_id"])
    assert np.array_equal(back[0]["curve"], first["curve"])


def test_the_document_fits_on_ham_train_and_reads_isic_only_descriptively(tmp_path):
    path = tmp_path / "curves.csv"
    t6.write_curves(path, _rows())
    doc = t6.build_document(path, n_resamples=50, seed=1)
    ham, isic = doc["datasets"]["ham10000"], doc["datasets"]["isic2018"]
    assert ham["thresholds"]["dark"]["threshold"] < 0.5 and ham["thresholds"]["dark"]["fit_n"] == 120
    assert ham["evaluation"]["verdict"]["prediction"] == lt.FULFILLED
    assert "verdict" not in isic["evaluation"] and isic["evaluation"]["n_without_tercile"] == 1
    assert ham["coverage"] == {"fit": {"dark": 120, "medium": 120, "light": 120, "unavailable": 0},
                               "eval": {"dark": 110, "medium": 110, "light": 110, "unavailable": 0}}
    assert doc["params"]["grid"] == list(lt.GRID) and doc["params"]["seed"] == 1
    assert doc["inputs"]["curves_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_table_is_rendered_from_the_document_with_decimal_commas(tmp_path):
    path = tmp_path / "curves.csv"
    t6.write_curves(path, _rows())
    doc = t6.build_document(path, n_resamples=50, seed=1)
    text = t6.render_markdown(doc)
    assert "no son tono de piel" in text and "2510.06655" in text
    dark = doc["datasets"]["ham10000"]["thresholds"]["dark"]["threshold"]
    assert f"| oscuro | {dark:.2f} |".replace(".", ",") in text and "0.3" not in text
    assert lt.FULFILLED in text
