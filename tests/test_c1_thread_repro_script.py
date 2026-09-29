"""c1_thread_repro_check.py end to end on a fake published C1 grid (no GPU, no pod)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from env_sidecar import sidecar_env

from dermssl import c1_split_identity as si
from dermssl import c1_thread_repro as ctr
from dermssl import metrics as _metrics
from dermssl import probes
from dermssl.feature_cache import _digest

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 3
BACKBONE, CONDITION = "v7_astar_s42", "none"
MASKS_SHA = "m" * 64


def _load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("c1_thread_repro_check", ROOT / "scripts/c1_thread_repro_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data():
    rng = np.random.default_rng(3)
    y_tr, y_va = np.arange(90) % N_CLASSES, np.arange(30) % N_CLASSES
    x_tr = (rng.normal(size=(90, 8)) + y_tr[:, None] * 0.5).astype(np.float32)
    x_va = (rng.normal(size=(30, 8)) + y_va[:, None] * 0.5).astype(np.float32)
    return x_tr, y_tr, x_va, y_va


def _publish(tmp_path: Path):
    """A grid cell whose proba was fitted under 1 BLAS thread, plus its committed c1_cells.json."""
    x_tr, y_tr, x_va, y_va = _data()
    train_ids, val_ids = [f"t{i}" for i in range(90)], [f"v{i}" for i in range(30)]
    split = tmp_path / "c1_split_index.json"
    split.write_text(json.dumps({"train_ids": train_ids, "val_ids": val_ids}), encoding="utf-8")
    proba, _ = ctr.fit_with_threads(probes.run_probe, "logreg", x_tr, y_tr, x_va, N_CLASSES, threads=1)
    proba = np.asarray(proba, dtype=np.float64)
    macro_f1 = float(_metrics.summarize(y_va, proba, N_CLASSES)["macro_f1"])
    grid = tmp_path / "salidas/c1_v7/grid"
    grid.mkdir(parents=True)
    stem = grid / f"cell__{BACKBONE}__{CONDITION}"
    np.save(f"{stem}.features.npy", x_va)
    np.save(f"{stem}.proba.npy", proba)
    manifest = {"backbone": BACKBONE, "condition": CONDITION, "n_rows": 30, "features_sha256": _digest(x_va),
                "proba_sha256": _digest(proba), "code_revision": "old", "masks_sha256": MASKS_SHA,
                "c1_train_ids_sha256": si.ids_sha256(train_ids), "c1_val_ids_sha256": si.ids_sha256(val_ids),
                "probe": "logreg", "backbone_spec": "dinov2_vitb14:/w/a.pt",
                "labels_sha256": _digest(np.concatenate([y_tr, y_va])), "metrics": {"macro_f1": macro_f1},
                "n_classes": N_CLASSES}
    Path(f"{stem}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    published = tmp_path / "pub/student_r3/c1_cells.json"
    published.parent.mkdir(parents=True)
    published.write_text(json.dumps({f"{BACKBONE}__{CONDITION}": {"metrics": {"macro_f1": macro_f1}}}),
                         encoding="utf-8")
    return split, grid, published, (x_tr, y_tr, x_va, y_va)


@pytest.fixture
def run(tmp_path, monkeypatch):
    script = _load_script()
    split, grid, published, (x_tr, y_tr, x_va, y_va) = _publish(tmp_path)
    calls = []

    def fake_extract(ctx, cell, manifest):
        calls.append(cell)
        return x_tr.copy(), x_va.copy()

    monkeypatch.setattr(script, "extract_features", fake_extract)
    monkeypatch.setattr(script, "load_labels", lambda args, index: (y_tr, y_va))
    monkeypatch.setattr(script, "masks_sha256", lambda masks_dir: MASKS_SHA)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "diag"
    argv = [f"--cache-dir={tmp_path}/cache", f"--image-dir={tmp_path}/img", f"--split-index={split}",
            f"--grid=student_r3={grid}", f"--masks=student_r3={tmp_path}/masks",
            f"--published=student_r3={published}", f"--cell=student_r3:{BACKBONE}:{CONDITION}",
            "--threads", "default", "recorded", "1", "2", f"--scratch-dir={tmp_path}/scratch",
            f"--out-dir={out}", "--allow-dirty"]
    return script, argv, grid, out, calls


def _doc(out: Path) -> dict:
    return json.loads((out / ctr.OUT_JSON).read_text(encoding="utf-8"))


def test_the_published_thread_count_reproduces_bit_for_bit_and_json_readme_sidecar_are_written(run):
    script, argv, _, out, _ = run
    assert script.main(argv) == 0
    doc = _doc(out)
    cell = doc["cells"][0]
    assert cell["gate"]["passed"] and cell["repeat_bit_equal"] is True
    one = next(f for f in cell["fits"] if f["setting"] == "1")
    assert one["bit_equal"] and one["max_abs_delta"] == 0.0 and one["resolved_threads"]
    assert [s["label"] for s in doc["settings"]] == ["default", "1", "2"]  # nothing recorded next to the grid
    assert "1" in doc["summary"]["reproducing_settings"]
    assert sidecar_env(out / ctr.OUT_JSON)["knobs"]["threads"] == ["default", "recorded", "1", "2"]
    assert (out / "README.md").read_text(encoding="utf-8") == ctr.render_readme(doc)


def test_val_features_that_differ_from_the_published_cell_fail_the_gate_and_exit_1(run, monkeypatch):
    script, argv, _, out, _ = run
    x_tr, _, x_va, _ = _data()
    x_va[0, 0] = np.nextafter(x_va[0, 0], np.float32(9))
    monkeypatch.setattr(script, "extract_features", lambda ctx, cell, manifest: (x_tr, x_va))
    assert script.main(argv) == 1
    cell = _doc(out)["cells"][0]
    assert not cell["gate"]["passed"] and cell["fits"] == []
    assert any("features_sha256" in p for p in cell["gate"]["problems"])


def test_a_published_macro_f1_that_is_not_the_manifest_one_fails_the_gate(run, tmp_path):
    script, argv, _, out, calls = run
    path = tmp_path / "pub/student_r3/c1_cells.json"
    path.write_text(json.dumps({f"{BACKBONE}__{CONDITION}": {"metrics": {"macro_f1": 0.123}}}), encoding="utf-8")
    assert script.main(argv) == 1
    assert calls == []  # refused before any extraction


def test_the_scratch_features_are_reused_on_a_rerun(run, monkeypatch):
    script, argv, _, _, calls = run
    assert script.main(argv) == 0 and len(calls) == 1
    assert script.main(argv) == 0 and len(calls) == 1


def test_the_output_inside_a_published_grid_is_refused(run):
    script, argv, grid, _, _ = run
    argv = [a for a in argv if not a.startswith("--out-dir")] + [f"--out-dir={grid / 'diag'}"]
    with pytest.raises(SystemExit):
        script.main(argv)


def test_a_thread_env_var_is_refused_because_default_would_not_be_the_library_default(run, monkeypatch):
    script, argv, _, _, _ = run
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(SystemExit):
        script.main(argv)


def test_the_recorded_setting_is_read_from_the_grid_sidecar(run):
    script, argv, grid, out, _ = run
    (grid / "c1_cells.json").write_text("{}", encoding="utf-8")
    env = {"env": {"OMP_NUM_THREADS": "2"}, "threadpools": [{"internal_api": "openblas", "num_threads": 2}]}
    from dermssl import run_environment

    run_environment.write_sidecar(grid / "c1_cells.json", environment=env)
    assert script.main(argv) == 0
    doc = _doc(out)
    assert doc["recorded_threads"]["value"] == 2
    assert [s["label"] for s in doc["settings"]] == ["default", "recorded", "1", "2"]


def test_the_scratch_inside_a_published_grid_is_refused(run):
    script, argv, grid, _, _ = run
    argv = [a for a in argv if not a.startswith("--scratch-dir")] + [f"--scratch-dir={grid / 'f'}"]
    with pytest.raises(SystemExit):
        script.main(argv)
