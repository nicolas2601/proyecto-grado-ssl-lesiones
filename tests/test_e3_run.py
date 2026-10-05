"""scripts/e3_run.py end to end with a fake extractor and synthetic items (no GPU, no real images)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e3_io  # noqa: E402
import e3_run  # noqa: E402

from dermssl import c1_extraction as ex  # noqa: E402
from dermssl import e3_store as st  # noqa: E402
from dermssl import run_environment  # noqa: E402

SIZES = {"fit": 30, "holdout": 15, "train": 36, "val": 18}
DIM = 6


def label_of(image_id):
    return int(image_id.split("_")[1]) % 3


def seed_of(image_id):
    return int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16)


def load_item(image_id):
    rng = np.random.default_rng(seed_of(image_id))
    image = rng.integers(20, 90, (90, 120, 3), dtype=np.uint8) + np.uint8(50 * label_of(image_id))
    mask = np.zeros((90, 120), dtype=bool)
    mask[25:65, 35:85] = True
    return image, mask


def resize_crop(image, resolution):
    return ex._center_crop(ex._resize_shorter_side(image, int(resolution * 1.14)), resolution)


class Extractor:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def __call__(self, images, n_views):
        self.calls += 1
        if self.fail:
            raise RuntimeError("cuda out of memory")
        flat = images.reshape(len(images), -1, 3).astype(np.float32)
        base = np.concatenate([flat.mean(axis=1), flat.std(axis=1)], axis=1)
        return np.stack([base + 0.01 * v for v in range(n_views)])


def runtime(fail=False, forbidden=()):
    ids = {s: [f"{s}_{i}" for i in range(n)] for s, n in SIZES.items()}
    seen = []

    def labels(split):
        seen.append(split)
        assert split not in forbidden, f"label read of {split}"
        return np.array([label_of(i) for i in ids[split]])

    rt = e3_io.Runtime(
        ids=ids, labels_of=labels, groups_of=lambda s: np.array([f"{s}{k // 2}" for k in range(SIZES[s])]),
        codes_of=lambda s: (np.arange(SIZES[s]) // 2) % 3, n_classes=3, mel=1, load_item=load_item,
        extract_fn=Extractor(fail), resize_crop=resize_crop, hair_fn=None,
        fingerprint={"protocol": e3_io.PROTOCOL, "masks_sha256": "m" * 64, "weights_sha256": "w" * 64,
                     "manifest_sha256": "x" * 64, "holdout_sha256": "h" * 64, "limit": None})
    rt.seen = seen
    return rt


def cli(tmp_path, stage, *extra, rt=None):
    argv = ["--stage", stage, "--out-dir", str(tmp_path), *extra]
    return e3_run.main(argv, runtime=rt or runtime(), revision="rev0")


def extract_selection_inputs(tmp_path, *extra):
    assert cli(tmp_path, "extract", "--splits", "fit", "holdout", "--batch-size", "7", *extra) == 0


def selected(tmp_path):
    extract_selection_inputs(tmp_path)
    assert cli(tmp_path, "select", "--ci-resamples", "10") == 0
    path = tmp_path / "e3_selection.json"
    return path, st.file_sha256(path)


def test_extract_writes_arrays_cells_and_environment_sidecars_and_resumes(tmp_path, capsys):
    extract_selection_inputs(tmp_path)
    assert np.load(tmp_path / "features" / "holdout__full.npy").shape == (8, 15, 2 * 3)
    assert np.load(tmp_path / "features" / "fit__crop_view.npy").shape == (1, 30, 6)
    cell = tmp_path / "cells" / "extract__holdout.json"
    assert run_environment.read_sidecar(cell)["knobs"]["batch_size"] == 7
    assert run_environment.sidecar_problems(run_environment.sidecar_path(cell)) == []
    capsys.readouterr()
    extract_selection_inputs(tmp_path)
    assert capsys.readouterr().out.count("skipped") == 2


def test_extract_of_train_and_val_needs_the_pinned_selection(tmp_path):
    extract_selection_inputs(tmp_path)
    assert cli(tmp_path, "extract", "--splits", "val") == 2
    assert not list((tmp_path / "features").glob("val__*"))


def test_select_writes_a_selection_and_never_asks_for_val_or_train_labels(tmp_path, capsys):
    extract_selection_inputs(tmp_path)
    rt = runtime(forbidden=("val", "train"))
    assert cli(tmp_path, "select", "--ci-resamples", "10", rt=rt) == 0
    doc = json.loads((tmp_path / "e3_selection.json").read_text())
    assert doc["complete"] and doc["split"] == "holdout" and doc["val_plan"][0] == "B71"
    assert "sha256" in capsys.readouterr().out
    assert run_environment.read_sidecar(tmp_path / "e3_selection.json") is not None
    assert set(rt.seen) <= {"fit", "holdout"}


def test_select_with_a_missing_variant_lists_the_failure_and_writes_no_selection(tmp_path, capsys):
    assert cli(tmp_path, "extract", "--splits", "fit", "holdout", "--conditions", "full", "crop_view") == 0
    assert cli(tmp_path, "select", "--variants", "B71", "F1", "F5", "--ci-resamples", "5") == 1
    assert "[FALLO] select__F5" in capsys.readouterr().out
    assert not (tmp_path / "e3_selection.json").exists()


def test_a_failing_extraction_is_retried_listed_and_the_exit_code_is_nonzero(tmp_path, capsys):
    rt = runtime(fail=True)
    assert cli(tmp_path, "extract", "--splits", "fit", rt=rt) == 1
    assert "[FALLO] extract__fit" in capsys.readouterr().out
    assert rt.extract_fn.calls == 3 * 1  # first batch of 16 images x 3 attempts (fails on the first condition)


def test_final_aborts_without_a_selection_before_loading_anything(tmp_path):
    rt = runtime()
    assert cli(tmp_path, "final", "--selection-sha256", "0" * 64, rt=rt) == 2
    assert rt.seen == [] and not (tmp_path / "val_read.lock").exists()


def test_final_aborts_when_the_selection_is_not_pinned(tmp_path):
    path, sha = selected(tmp_path)
    assert cli(tmp_path, "final") == 2  # no --selection-sha256
    assert cli(tmp_path, "final", "--selection-sha256", "f" * 64) == 2
    assert not (tmp_path / "val_read.lock").exists()


def test_full_flow_reads_val_once_and_a_second_final_is_refused(tmp_path, capsys):
    path, sha = selected(tmp_path)
    pin = ["--selection", str(path), "--selection-sha256", sha]
    assert cli(tmp_path, "extract", "--splits", "train", "val", *pin) == 0
    assert cli(tmp_path, "final", *pin, "--ci-resamples", "20") == 0
    doc = json.loads((tmp_path / "e3_final_eval.json").read_text())
    assert doc["selection_sha256"] == sha and doc["val_plan"] == json.loads(path.read_text())["val_plan"]
    assert doc["bootstrap"]["seed"] == 20260923 and "selection_committed" in doc
    assert run_environment.read_sidecar(tmp_path / "e3_final_eval.json") is not None
    rt = runtime()
    assert cli(tmp_path, "final", *pin, rt=rt) == 2
    assert "val ya se leyo" in capsys.readouterr().out and "val" not in rt.seen


def test_train_and_val_are_extracted_only_for_the_conditions_of_the_val_plan(tmp_path):
    path, sha = selected(tmp_path)
    plan = json.loads(path.read_text())["val_plan"]
    assert cli(tmp_path, "extract", "--splits", "val", "--selection", str(path), "--selection-sha256", sha) == 0
    from dermssl import e3_variants as var
    needed = {c for name in plan for c in var.required_conditions(name)}
    got = {p.name.split("__")[1][:-4] for p in (tmp_path / "features").glob("val__*.npy")}
    assert got == needed | {"descriptors", "fallbacks"}
