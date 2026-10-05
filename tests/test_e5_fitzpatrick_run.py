"""E5 runner end to end on synthetic images: JSON + sidecar, resume, smoke naming, unreadable files."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import e5_data  # noqa: E402
import e5_fitzpatrick as run  # noqa: E402

from dermssl import run_environment  # noqa: E402
from dermssl.e5_analyze import Task  # noqa: E402
from dermssl.e5_document import DatasetSpec  # noqa: E402

SKINS = ((178, 133, 102), (238, 204, 178), (140, 96, 76))  # III, I, IV-ish


def _png(tmp_path, i):
    arr = np.empty((90, 120, 3), dtype=np.uint8)
    arr[:] = SKINS[i % 3]
    yy, xx = np.mgrid[:90, :120]
    arr[(yy - 45) ** 2 + (xx - 60) ** 2 <= 16**2] = (60, 35, 28)
    arr = np.clip(arr + np.random.RandomState(i).randint(-3, 4, arr.shape), 0, 255).astype(np.uint8)
    path = tmp_path / f"im{i}.png"
    Image.fromarray(arr).save(path)
    return path


def _spec(tmp_path, n=9, broken=False) -> DatasetSpec:
    tasks = [Task(f"im{i}", str(_png(tmp_path, i)), "train" if i < 6 else "val", "mel" if i % 2 else "nv")
             for i in range(n)]
    if broken:
        (tmp_path / "im0.png").write_bytes(b"broken")
    return DatasetSpec("isic2019", ("mel", "nv"), ("train", "val"), tasks, {"split": {"n_train": 6}}, None)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    state = {"spec": None}
    monkeypatch.setattr(e5_data, "load_spec", lambda args: state["spec"])
    monkeypatch.setattr(run, "revision", lambda args: "deadbeef")
    state["set"] = lambda spec: state.__setitem__("spec", spec)
    return state


def _args(tmp_path, *extra):
    return ["--dataset", "isic2019", "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out"),
            "--workers", "1", *extra]


def test_writes_the_json_and_its_environment_sidecar_and_records_the_workers(patched, tmp_path):
    patched["set"](_spec(tmp_path))
    assert run.main(_args(tmp_path)) == 0
    out = tmp_path / "out" / "e5_fitzpatrick_isic2019.json"
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["descriptive"] is True and doc["validated"] is False and doc["code_revision"] == "deadbeef"
    assert doc["report"]["n_images"] == 9 and doc["estimators"]["roi"]["applicable"] is False
    side = run_environment.read_sidecar(out)
    assert side["knobs"]["workers"] == 1 and run_environment.sidecar_problems(run_environment.sidecar_path(out)) == []


def test_the_result_does_not_depend_on_the_number_of_workers(patched, tmp_path):
    patched["set"](_spec(tmp_path))
    assert run.main(_args(tmp_path)) == 0
    one = (tmp_path / "out" / "e5_fitzpatrick_isic2019.json").read_bytes()
    assert run.main(["--dataset", "isic2019", "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out2"),
                     "--workers", "3", "--chunk-size", "2"]) == 0
    assert (tmp_path / "out2" / "e5_fitzpatrick_isic2019.json").read_bytes() == one


def test_an_existing_result_is_skipped_unless_overwrite_is_given(patched, tmp_path, capsys):
    patched["set"](_spec(tmp_path))
    run.main(_args(tmp_path))
    out = tmp_path / "out" / "e5_fitzpatrick_isic2019.json"
    out.write_text("{}", encoding="utf-8")
    assert run.main(_args(tmp_path)) == 0 and out.read_text(encoding="utf-8") == "{}"
    assert "skip" in capsys.readouterr().out
    assert run.main(_args(tmp_path, "--overwrite")) == 0 and json.loads(out.read_text())["dataset"] == "isic2019"


def test_a_smoke_run_writes_to_its_own_name(patched, tmp_path):
    patched["set"](_spec(tmp_path))
    assert run.main(_args(tmp_path, "--limit", "3")) == 0
    out = tmp_path / "out" / "e5_fitzpatrick_isic2019_smoke_3.json"
    assert json.loads(out.read_text())["smoke_limit"] == 3 and not out.with_name("e5_fitzpatrick_isic2019.json").exists()


def test_an_unreadable_file_is_recorded_and_the_exit_code_is_not_zero(patched, tmp_path):
    patched["set"](_spec(tmp_path, broken=True))
    assert run.main(_args(tmp_path)) == 1
    doc = json.loads((tmp_path / "out" / "e5_fitzpatrick_isic2019.json").read_text())
    assert doc["report"]["n_unreadable"] == 1 and doc["unreadable"][0]["image_id"] == "im0"


def test_dry_run_loads_the_spec_and_writes_nothing(patched, tmp_path, capsys):
    patched["set"](_spec(tmp_path))
    assert run.main(["--dataset", "isic2019", "--data-root", str(tmp_path), "--dry-run"]) == 0
    assert "9 images" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_out_dir_is_required_unless_dry_run(tmp_path):
    with pytest.raises(SystemExit):
        run.parse_args(["--dataset", "isic2019", "--data-root", str(tmp_path)])


def test_the_thread_settings_are_pinned_before_anything_else_is_imported():
    text = (REPO / "scripts" / "e5_fitzpatrick.py").read_text(encoding="utf-8")
    pin = text.index("os.environ.setdefault(_var")
    assert pin < text.index("import e5_data") and pin < text.index("from dermssl")
    assert all(v in text for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))


def test_dirty_tree_is_refused_except_for_a_smoke_run_with_allow_dirty(monkeypatch, tmp_path):
    monkeypatch.undo()
    from dermssl import run_provenance as rp

    def dirty(repo):
        raise rp.DirtyTreeError("dirty")

    monkeypatch.setattr(rp, "clean_revision", dirty)
    full = run.parse_args(["--dataset", "isic2019", "--data-root", str(tmp_path), "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="ABORT"):
        run.revision(full)
    smoke = run.parse_args(["--dataset", "isic2019", "--data-root", str(tmp_path), "--out-dir", str(tmp_path),
                            "--limit", "5", "--allow-dirty"])
    assert run.revision(smoke) == "dirty"
