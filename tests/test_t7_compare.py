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
from env_sidecar import sidecar_env  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
A2_RECORD = json.loads((tc.RESULTS / tc.A2_RUN / "run.json").read_text(encoding="utf-8"))
A2_META = {d: json.loads((tc.RESULTS / tc.A2_RUN / f"evaluacion_{d}" / "summary.json").read_text())["meta"]
           for d in ("ham10000", "isic2018")}
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
    ("T7", lambda r: r.update(evaluation_regime="inductive"), "evaluation_regime"),
    ("T7-aug", lambda r: r["prediction"].update(threshold=0.5), "prediction"),
    ("A2-s43", lambda r: r.update(reads_masks_or_labels=True), "reads_masks_or_labels"),
    ("T7", lambda r: r.update(extra_flag=1), "extra_flag"),
    ("T7", lambda r: r["pool"].update(n_images=1), "pool"),
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
        for dataset, n in (("ham10000", 500), ("isic2018", 300)):
            meta = {**A2_META[dataset], "checkpoint": f"/pod/t7/{dirs[arm].name}/best.pt", "n_images": n}
            (dirs[arm] / f"evaluacion_{dataset}" / "summary.json").write_text(json.dumps({"meta": meta}))
    reference = {"ham10000": (ham_dice["A2"], ham_hd["A2"]), "isic2018": (isic_dice["A2"], isic_hd["A2"])}
    monkeypatch.setattr(tc, "check_registration", lambda repo: "r" * 40)
    monkeypatch.setattr(tc.rp, "clean_revision", lambda repo: "c" * 40)
    monkeypatch.setattr(tc, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(tc, "read_a2", lambda dataset: reference[dataset])
    monkeypatch.setattr(tc, "read_a2_meta", lambda dataset: {**A2_META[dataset],
                                                             "n_images": {"ham10000": 500, "isic2018": 300}[dataset]})
    monkeypatch.setattr(tc, "fold0_val_ids", lambda ham_root: ham_dark[:150] + ham_rest[:150])
    monkeypatch.setattr(tc, "REGISTERED_BOOTSTRAP", {"n_resamples": 100, "seed": 20260923})
    out = tmp_path / "out"
    assert tc.main(["--t7", str(dirs["T7"]), "--s43", str(dirs["A2-s43"]), "--aug", str(dirs["T7-aug"]),
                    "--manifest", str(manifest), "--ham-root", str(tmp_path), "--out", str(out),
                    "--n-resamples", "100"]) == 0
    doc = json.loads((out / tc.JSON_FILE).read_text(encoding="utf-8"))
    assert set(doc["datasets"]) == {"ham10000", "ham10000_fold0_val", "isic2018"}
    assert "verdict" in doc["datasets"]["ham10000"]
    assert "verdict" not in doc["datasets"]["isic2018"] and "verdict" not in doc["datasets"]["ham10000_fold0_val"]
    assert doc["provenance"]["registration_commit"] == "r" * 40
    assert sidecar_env(out / tc.JSON_FILE)["schema"] == 1
    assert (out / tc.TABLE_FILE).read_text(encoding="utf-8") == tc.render_markdown(doc)
    table = (out / tc.TABLE_FILE).read_text(encoding="utf-8")
    assert "Bootstrap estratificado y pareado" not in table
    assert "brecha y sus contrastes: estratificado" in table and "ΔD global: pareado por imagen, sin estratificar" in table


def test_it_refuses_before_reading_data_when_the_registration_fails(tmp_path, monkeypatch):
    def refuse(repo, **kwargs):
        raise pg.RegistrationError("el bloque registrado cambio")
    monkeypatch.setattr(pg, "check_registration", refuse)
    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
        tc.main(["--t7", missing, "--s43", missing, "--aug", missing, "--manifest", missing,
                 "--ham-root", missing, "--out", str(tmp_path / "out")])


def test_arm_paths_and_bookkeeping_may_differ_from_a2():
    record = _arm_record("T7")
    record["pool"]["cache_dir"] = "/elsewhere/cache"
    record["pseudo_masks"]["path"] = "/elsewhere/pseudo_masks.npy"
    record.update(code_revision="f" * 40, result={"epochs": 20}, environment={"python": "3.12"})
    tc.check_arm("T7", record, A2_RECORD)


@pytest.mark.parametrize("extra", [("--n-resamples", "999"), ("--seed", "1")])
def test_an_unregistered_bootstrap_is_refused_before_anything_runs(tmp_path, extra):
    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="bootstrap"):
        tc.main(["--t7", missing, "--s43", missing, "--aug", missing, "--manifest", missing,
                 "--ham-root", missing, "--out", str(tmp_path / "out"), *extra])


def _csv(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "per_image.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "dice", "hd95"])
        writer.writerows(rows)
    return path


def test_per_image_rows_must_be_unique_with_a_finite_dice(tmp_path):
    dice, hd95 = tc._read_per_image(_csv(tmp_path, [("a", "0.8", "nan"), ("b", "0.7", "12")]))
    assert dice == {"a": 0.8, "b": 0.7} and np.isnan(hd95["a"])  # an empty mask has no HD95; Dice must exist
    with pytest.raises(ValueError, match="duplicad"):
        tc._read_per_image(_csv(tmp_path, [("a", "0.8", "1"), ("a", "0.7", "1")]))
    with pytest.raises(ValueError, match="Dice no finito"):
        tc._read_per_image(_csv(tmp_path, [("a", "nan", "1")]))


def test_the_published_a2_evaluation_meta_is_readable():
    assert tc.read_a2_meta("ham10000")["threshold"] == 0.5


def _meta(arm_dir: str = "t7", **change) -> dict:
    return {**A2_META["ham10000"], "checkpoint": f"/somewhere/{arm_dir}/best.pt", **change}


def test_an_arm_evaluated_like_a2_passes(tmp_path):
    tc.check_evaluation("T7", "ham10000", tmp_path / "t7", _meta(), A2_META["ham10000"])


@pytest.mark.parametrize(("meta", "match"), [
    (_meta(threshold=0.4), "threshold"), (_meta(tta=True), "tta"), (_meta(n_images=10), "n_images"),
    (_meta(resolution=448), "resolution"), (_meta(predictions="external"), "predictions"),
    (_meta(arm_dir="a2"), "checkpoint"), (_meta(checkpoint="/x/t7/last.pt"), "checkpoint"),
])
def test_an_arm_evaluated_otherwise_is_refused(tmp_path, meta, match):
    with pytest.raises(ValueError, match=match):
        tc.check_evaluation("T7", "ham10000", tmp_path / "t7", meta, A2_META["ham10000"])


@pytest.mark.parametrize("value", ["1.2", "-0.1"])
def test_a_dice_outside_zero_one_is_refused(tmp_path, value):
    with pytest.raises(ValueError, match="fuera de"):
        tc._read_per_image(_csv(tmp_path, [("a", value, "1")]))
