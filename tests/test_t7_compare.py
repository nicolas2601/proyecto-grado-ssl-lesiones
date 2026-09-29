"""The T7 comparison (docs/27): arm records against A2's, paired gaps, verdicts and the generated table."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import seg_paths  # noqa: F401
from dermssl import luminance_retrain as lr
from dermssl import luminance_threshold as lt
from dermssl import pad_ufes_guard as pg
from scripts import t7_compare as tc

REPO = Path(__file__).resolve().parents[1]
A2_RECORD = json.loads((tc.RESULTS / tc.A2_RUN / "run.json").read_text(encoding="utf-8"))
FST_COLUMNS = ["image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita", "n_skin_px"]


def _arm_record(arm: str) -> dict:
    record = copy.deepcopy(A2_RECORD)
    if arm == "A2-s43":
        record["seed"] = 43
        return record
    record["variant"] = "A2-T7"
    record["t7"] = {"registration_commit": tc.ls.REGISTRATION_COMMIT, "tercile_manifest_sha256": tc.ls.MANIFEST_SHA256,
                    "oversample_k": 2 if arm == "T7" else 1,
                    "darken": {"probability": 0.5, "gamma_range": [1.0, 2.2], "brightness_range": [0.85, 1.0],
                               "contrast_range": [0.85, 1.0], "terciles": ["medium", "light"]},
                    "n_dark_train_confident": 16000, "n_darken_eligible_train_confident": 33000}
    return record


def test_the_real_registration_precedes_the_real_script():
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow != "false":
        pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
    assert tc.check_registration(REPO) == tc.REGISTRATION_COMMIT


@pytest.mark.parametrize("arm", tc.ARMS)
def test_a_registered_arm_record_passes(arm):
    tc.check_arm(arm, _arm_record(arm), A2_RECORD)


@pytest.mark.parametrize(("arm", "change", "match"), [
    ("T7", lambda r: r["schedule"].update(lr=2e-4), "schedule"),
    ("T7", lambda r: r.update(seed=43), "semilla"),
    ("T7", lambda r: r["t7"].update(oversample_k=3), "k"),
    ("T7-aug", lambda r: r["t7"].update(oversample_k=2), "k"),
    ("T7", lambda r: r["t7"]["darken"].update(probability=0.4), "darken"),
    ("A2-s43", lambda r: r.update(seed=42), "semilla"),
    ("A2-s43", lambda r: r.update(t7={}), "t7"),
    ("T7", lambda r: r["pseudo_masks"].update(sha256="0" * 64), "pseudo_masks"),
    ("T7", lambda r: r["training"].update(n_train_confident=1), "training"),
    ("T7", lambda r: r.update(complete=False), "completa"),
])
def test_an_arm_that_moved_anything_else_is_refused(arm, change, match):
    record = _arm_record(arm)
    change(record)
    with pytest.raises(ValueError, match=match):
        tc.check_arm(arm, record, A2_RECORD)


def _dice(rng, n: int, mean: float) -> np.ndarray:
    return np.clip(rng.normal(mean, 0.05, n), 0, 1)


def _models(rng, ids_dark, ids_rest, dark_means: dict) -> tuple[dict, dict]:
    dice, hd95 = {}, {}
    for name, mean in dark_means.items():
        values = np.concatenate([_dice(rng, len(ids_dark), mean), _dice(rng, len(ids_rest), 0.82)])
        dice[name] = dict(zip([*ids_dark, *ids_rest], values.tolist()))
        hd95[name] = {i: 20.0 for i in dice[name]}
    return dice, hd95


def test_the_analysis_gives_the_registered_verdicts_and_the_seed_contrast():
    rng = np.random.default_rng(0)
    dark, rest = [f"d{k}" for k in range(400)], [f"r{k}" for k in range(600)]
    terciles = {**dict.fromkeys(dark, "dark"), **dict.fromkeys(rest, "medium"), "u": "unavailable"}
    dice, hd95 = _models(rng, dark, rest, {"A2": 0.76, "T7": 0.80, "A2-s43": 0.761, "T7-aug": 0.77})
    out = tc.analyze_dataset(dice, hd95, terciles, [*dark, *rest, "u"], 300, 20260923, with_verdict=True)
    assert out["n_dark"] == 400 and out["n_rest"] == 600 and out["n_without_tercile"] == 1
    assert out["reduction"]["T7"]["delta"] == pytest.approx(0.04, abs=0.01)
    assert out["verdict"]["gap"] == lt.GAP_FULFILLED and out["verdict"]["global"] == lt.GLOBAL_HOLDS
    assert out["verdict"]["prediction"] == lt.FULFILLED
    assert out["verdict"]["seed"] == lr.SEED_SEPARATED
    assert out["verdict"]["conclusion"] == lr.SHRINKS
    assert out["seed_contrast"]["delta"] == pytest.approx(
        out["reduction"]["T7"]["delta"] - out["reduction"]["A2-s43"]["delta"])
    assert set(out["reduction"]) == {"T7", "A2-s43", "T7-aug"}
    descriptive = tc.analyze_dataset(dice, hd95, terciles, [*dark, *rest], 50, 1, with_verdict=False)
    assert "verdict" not in descriptive


def _write_per_image(path: Path, dice: dict, hd95: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "dice", "hd95"])
        writer.writerows([i, d, hd95[i]] for i, d in dice.items())


def test_the_runner_writes_the_json_and_a_table_generated_from_it(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    ham_dark, ham_rest = [f"ISIC_{k:07d}" for k in range(24306, 24506)], [f"ISIC_{k:07d}" for k in range(24506, 24806)]
    isic = [f"ISIC_{k:07d}" for k in range(0, 300)]
    ham_dice, ham_hd = _models(rng, ham_dark, ham_rest, {"A2": 0.76, "T7": 0.78, "A2-s43": 0.76, "T7-aug": 0.77})
    isic_dice, isic_hd = _models(rng, isic[:120], isic[120:], {"A2": 0.78, "T7": 0.79, "A2-s43": 0.78,
                                                                 "T7-aug": 0.78})
    manifest = tmp_path / "fst.csv"
    with open(manifest, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FST_COLUMNS)
        rows = [(i, "dark") for i in ham_dark + isic[:120]] + [(i, "light") for i in ham_rest + isic[120:]]
        writer.writerows([i, "", t, "ok", "70", "10", "40", "100"] for i, t in rows)
    dirs = {}
    for arm in tc.ARMS:
        dirs[arm] = tmp_path / arm
        dirs[arm].mkdir()
        (dirs[arm] / "run.json").write_text(json.dumps(_arm_record(arm)), encoding="utf-8")
        _write_per_image(dirs[arm] / "evaluacion_ham10000" / "per_image.csv", ham_dice[arm], ham_hd[arm])
        _write_per_image(dirs[arm] / "evaluacion_isic2018" / "per_image.csv", isic_dice[arm], isic_hd[arm])
    reference = {"ham10000": (ham_dice["A2"], ham_hd["A2"]), "isic2018": (isic_dice["A2"], isic_hd["A2"])}
    monkeypatch.setattr(tc, "check_registration", lambda repo: "r" * 40)
    monkeypatch.setattr(tc.rp, "clean_revision", lambda repo: "c" * 40)
    monkeypatch.setattr(tc, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(tc, "read_a2", lambda dataset: reference[dataset])
    monkeypatch.setattr(tc, "fold0_val_ids", lambda ham_root: ham_dark[:150] + ham_rest[:150])
    out = tmp_path / "out"
    assert tc.main(["--t7", str(dirs["T7"]), "--s43", str(dirs["A2-s43"]), "--aug", str(dirs["T7-aug"]),
                    "--manifest", str(manifest), "--ham-root", str(tmp_path), "--out", str(out),
                    "--n-resamples", "100"]) == 0
    doc = json.loads((out / tc.JSON_FILE).read_text(encoding="utf-8"))
    assert set(doc["datasets"]) == {"ham10000", "ham10000_fold0_val", "isic2018"}
    assert "verdict" in doc["datasets"]["ham10000"]
    assert "verdict" not in doc["datasets"]["isic2018"] and "verdict" not in doc["datasets"]["ham10000_fold0_val"]
    assert doc["provenance"]["registration_commit"] == "r" * 40 and "environment" in doc
    assert (out / tc.TABLE_FILE).read_text(encoding="utf-8") == tc.render_markdown(doc)


def test_it_refuses_before_reading_data_when_the_registration_fails(tmp_path, monkeypatch):
    def refuse(repo, **kwargs):
        raise pg.RegistrationError("el bloque registrado cambio")
    monkeypatch.setattr(pg, "check_registration", refuse)
    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
        tc.main(["--t7", missing, "--s43", missing, "--aug", missing, "--manifest", missing,
                 "--ham-root", missing, "--out", str(tmp_path / "out")])
