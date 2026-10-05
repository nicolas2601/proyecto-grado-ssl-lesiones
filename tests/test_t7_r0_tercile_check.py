"""The T7 teacher check: r0 pseudo-masks against HAM10000 reference masks by luminance tercile (docs/27, T7)."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import evaluate_segmentation as es
from dermssl import pad_ufes_guard as pg
from scripts import t7_r0_tercile_check as t7r
from env_sidecar import sidecar_env  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SIZE = 8
FST_COLUMNS = ["image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita", "n_skin_px"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_real_registration_precedes_the_real_script():
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow != "false":
        pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
    assert t7r.check_registration(REPO) == t7r.REGISTRATION_COMMIT


def test_it_refuses_before_reading_data_when_the_registration_fails(tmp_path, monkeypatch):
    def refuse(repo, **kwargs):
        raise pg.RegistrationError("el bloque registrado cambio")
    monkeypatch.setattr(pg, "check_registration", refuse)
    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
        t7r.main(["--cache-dir", missing, "--pseudo-masks", missing, "--manifest", missing,
                  "--ham-mask-dir", missing, "--out", str(tmp_path / "out")])


def test_the_two_forms_of_the_teacher_dice_and_the_exclusion():
    truth = np.zeros((4, 4), bool)
    truth[:2, :2] = True
    mask = np.zeros((4, 4), np.uint8)
    mask[:2, :2] = 1
    mask[2:, 2:] = 255
    mask[0, 3] = 255
    row = t7r.teacher_dice(mask, truth)
    assert row["excluded"] is False
    assert row["dice_binary"] == es.compute_binary_metrics(mask == 1, truth)["dice"] == 1.0
    truth[3, 3] = True  # a reference pixel the teacher ignored: missed by (a), invisible to (b)
    row = t7r.teacher_dice(mask, truth)
    assert row["dice_binary"] == pytest.approx(8 / 9)
    assert row["dice_confident"] == 1.0
    excluded = t7r.teacher_dice(np.full((4, 4), 255, np.uint8), truth)
    assert excluded == {"excluded": True, "dice_binary": None, "dice_confident": None}


def test_a_file_with_another_sha256_is_refused(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="sha256"):
        t7r.require_sha256(path, "0" * 64, "manifiesto")
    t7r.require_sha256(path, _sha(path), "manifiesto")


def test_the_a2_reference_must_match_its_published_sha256sums():
    path = t7r.RESULTS / t7r.A2_RUN / "evaluacion_ham10000" / "per_image.csv"
    dice = t7r.read_a2_reference(path)
    assert len(dice) == 10015 and all(0.0 <= v <= 1.0 for v in dice.values())


def _rows(dark: list[float], rest: list[float], excluded_dark: int = 0) -> list[dict]:
    rows = [{"image_id": f"d{k}", "tone_tercile": "dark", "excluded": False, "dice_binary": v,
             "dice_confident": v} for k, v in enumerate(dark)]
    rows += [{"image_id": f"r{k}", "tone_tercile": "medium", "excluded": False, "dice_binary": v,
              "dice_confident": v} for k, v in enumerate(rest)]
    rows += [{"image_id": f"x{k}", "tone_tercile": "dark", "excluded": True, "dice_binary": None,
              "dice_confident": None} for k in range(excluded_dark)]
    return rows


def test_the_addendum_is_required_when_the_teacher_gap_is_at_least_a2s():
    rows = _rows([0.5] * 5 + [0.6] * 5, [0.8] * 10, excluded_dark=3)
    a2 = {**{f"d{k}": 0.7 for k in range(10)}, **{f"r{k}": 0.8 for k in range(10)},
          **{f"x{k}": 0.0 for k in range(3)}}
    doc = t7r.analyze(rows, a2, n_resamples=50, seed=1)
    assert doc["terciles"]["dark"] == {"n": 13, "n_excluded": 3, "excluded_fraction": pytest.approx(3 / 13),
                                       "dice_binary": pytest.approx(0.55), "dice_confident": pytest.approx(0.55),
                                       "dice_a2_same_images": pytest.approx(0.7)}
    assert doc["teacher_gap_binary"]["delta"] == pytest.approx(-0.25)
    assert doc["a2_gap_same_images"]["delta"] == pytest.approx(-0.1)
    assert doc["reduction_a2_minus_teacher"]["delta"] == pytest.approx(0.1 - 0.25)
    assert doc["addendum_required"] is True


def test_no_addendum_when_the_teacher_gap_is_smaller():
    rows = _rows([0.78] * 10, [0.8] * 10)
    a2 = {**{f"d{k}": 0.7 for k in range(10)}, **{f"r{k}": 0.8 for k in range(10)}}
    assert t7r.analyze(rows, a2, n_resamples=50, seed=1)["addendum_required"] is False


def _write_fixture(tmp: Path) -> dict:
    ids = [f"ISIC_{k:07d}" for k in range(6)]
    tercile = ["dark", "dark", "medium", "light", "medium", "unavailable"]
    cache = tmp / "cache"
    cache.mkdir()
    with open(cache / "manifest.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "dataset", "image_id", "path", "split"])
        writer.writerows([k, "ham10000", i, f"{i}.jpg", "train"] for k, i in enumerate(ids))
    masks = np.zeros((len(ids), SIZE, SIZE), np.uint8)
    masks[:, 2:6, 2:6] = 1
    masks[1] = 255
    np.save(tmp / "pseudo.npy", masks)
    gt = tmp / "gt"
    gt.mkdir()
    for i in ids:
        array = np.zeros((SIZE, SIZE), np.uint8)
        array[2:6, 2:7] = 255
        Image.fromarray(array).save(gt / f"{i}_segmentation.png")
    with open(tmp / "fst.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FST_COLUMNS)
        writer.writerows([i, "", t, "ok", "70", "10", "40", "100"] for i, t in zip(ids, tercile))
    return {"ids": ids, "cache": cache, "pseudo": tmp / "pseudo.npy", "gt": gt, "fst": tmp / "fst.csv"}


def test_the_runner_writes_the_json_and_a_table_generated_from_it(tmp_path, monkeypatch):
    fx = _write_fixture(tmp_path)
    monkeypatch.setattr(t7r, "check_registration", lambda repo: "r" * 40)
    monkeypatch.setattr(t7r.rp, "clean_revision", lambda repo: "c" * 40)
    monkeypatch.setattr(t7r, "MANIFEST_SHA256", _sha(fx["fst"]))
    monkeypatch.setattr(t7r, "PSEUDO_R0_SHA256", _sha(fx["pseudo"]))
    monkeypatch.setattr(t7r, "CACHE_MANIFEST_SHA256", _sha(fx["cache"] / "manifest.csv"))
    monkeypatch.setattr(t7r, "read_a2_reference", lambda path: {i: 0.8 for i in fx["ids"]})
    monkeypatch.setattr(t7r, "REGISTERED_BOOTSTRAP", {"n_resamples": 20, "seed": 20260923})
    out = tmp_path / "out"
    assert t7r.main(["--cache-dir", str(fx["cache"]), "--pseudo-masks", str(fx["pseudo"]),
                     "--manifest", str(fx["fst"]), "--ham-mask-dir", str(fx["gt"]), "--out", str(out),
                     "--n-resamples", "20"]) == 0
    doc = json.loads((out / t7r.JSON_FILE).read_text(encoding="utf-8"))
    assert doc["coverage"] == {"evaluated": 5, "without_tercile": 1, "not_in_pool": 0}
    assert doc["analysis"]["terciles"]["dark"]["n_excluded"] == 1
    assert doc["analysis"]["terciles"]["dark"]["dice_binary"] == pytest.approx(2 * 16 / (16 + 20))
    assert doc["provenance"]["registration_commit"] == "r" * 40
    listing = "".join(f"{i}  {_sha(fx['gt'] / f'{i}_segmentation.png')}\n" for i in fx["ids"][:5])
    assert doc["provenance"]["reference_masks_sha256"] == hashlib.sha256(listing.encode()).hexdigest()
    assert sidecar_env(out / t7r.JSON_FILE)["schema"] == 1
    assert (out / t7r.TABLE_FILE).read_text(encoding="utf-8") == t7r.render_markdown(doc)
    with open(out / t7r.ROWS_FILE, newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 5


@pytest.mark.parametrize("extra", [("--n-resamples", "999"), ("--seed", "1")])
def test_an_unregistered_bootstrap_is_refused_before_anything_runs(tmp_path, extra):
    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="bootstrap"):
        t7r.main(["--cache-dir", missing, "--pseudo-masks", missing, "--manifest", missing,
                  "--ham-mask-dir", missing, "--out", str(tmp_path / "out"), *extra])


@pytest.mark.parametrize("extra", [("--mask-suffix", "_other"), ("--mask-extension", ".jpg")])
def test_the_reference_mask_naming_is_fixed(tmp_path, extra):
    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit):
        t7r.parse_args(["--cache-dir", missing, "--pseudo-masks", missing, "--manifest", missing,
                        "--ham-mask-dir", missing, "--out", str(tmp_path / "out"), *extra])
