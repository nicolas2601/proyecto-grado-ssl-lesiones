"""``scripts/isic_eval_run.py`` end to end on synthetic data: split guard, labels, sidecars, resume."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import isic_eval_run as run  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl.isic_eval import protocol as pr  # noqa: E402
from dermssl.isic_eval import splits as sp  # noqa: E402
from test_isic_eval_runner import FakeExtractor  # noqa: E402
from test_isic_eval_splits import CLASSES, _expected2019  # noqa: E402


@pytest.fixture
def world(tmp_path, monkeypatch):
    rows = [{"image": f"ISIC_{k:05d}", **{c: int(c == CLASSES[k % 8]) for c in (*CLASSES, "UNK")}} for k in range(160)]
    data = tmp_path / "data"
    (data / "imgs").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(data / "ISIC_2019_Training_GroundTruth.csv", index=False)
    for r in rows:
        (data / "imgs" / f"{r['image']}.jpg").write_bytes(b"")
    expected = _expected2019(rows)
    monkeypatch.setitem(sp.LOADERS, "isic2019", (lambda csv, img: sp.load_isic2019(csv, img, expected),
                                                 "ISIC_2019_Training_GroundTruth.csv"))
    roots = {}
    for variant, dirs in (("original", ("final_A", "final_A_s43", "final_B")),
                          ("clean", ("final_A_s42", "final_A_s43", "final_B_s42"))):
        roots[variant] = tmp_path / f"weights_{variant}"
        for name in dirs:
            (roots[variant] / name).mkdir(parents=True)
            (roots[variant] / name / "simsiam_dinov2_backbone_final.pt").write_bytes(name.encode())
            (roots[variant] / name / "run.json").write_text(
                json.dumps({"exclude_ids_sha256": pr.POOL_EXCLUDE_SHA256[variant]}))
    monkeypatch.setattr(run, "build_extractor", lambda args, ds, bank: FakeExtractor(ds))
    monkeypatch.setattr(run, "revision", lambda args: "testrev")
    return {"data": data, "roots": roots, "out": tmp_path / "out"}


def cli(world, *extra, config="B71", root=None):
    root = root or world["roots"]["clean" if "--pool-excludes-val" in extra else "original"]
    return ["--dataset", "isic2019", "--data-root", str(world["data"]), "--config", config,
            "--weights-root", str(root), "--out-dir", str(world["out"]), "--ci-resamples", "10", *extra]


def test_dry_run_verifies_the_split_and_needs_no_weights(world, capsys):
    assert run.main(["--dataset", "isic2019", "--data-root", str(world["data"]), "--dry-run"]) == 0
    assert "split verified" in capsys.readouterr().out


def test_the_committed_split_is_refused_on_data_that_does_not_reproduce_it(world, monkeypatch):
    monkeypatch.setitem(sp.LOADERS, "isic2019", (sp.load_isic2019, "ISIC_2019_Training_GroundTruth.csv"))
    with pytest.raises(SystemExit, match="ABORT.*not the committed one"):
        run.main(["--dataset", "isic2019", "--data-root", str(world["data"]), "--dry-run"])


def test_clean_run_writes_labelled_cells_with_sidecars_and_contrasts(world):
    assert run.main(cli(world, "--pool-excludes-val", config="PUBLIC")) == 0
    assert run.main(cli(world, "--pool-excludes-val", config="B71")) == 0
    assert run.main(cli(world, "--pool-excludes-val", config="A71")) == 0
    out = world["out"] / "isic2019"
    doc = json.loads((out / "B71__clean.json").read_text())
    assert doc["result"]["transductivo"] is False and doc["result"]["code_revision"] == "testrev"
    assert doc["result"]["pool_check"]["verified"] and "per_class" in doc["result"]["metrics"]
    names = {p.name for p in out.rglob("*.json") if not p.name.endswith(".environment.json")}
    assert {"B71__clean.json", "A71__clean.json", "PUBLIC_B71.json", "PUBLIC_A71.json",
            "B71__clean__vs__PUBLIC_B71.json", "B71__clean__vs__A71__clean.json"} <= names
    for path in out.rglob("*.json"):
        if not path.name.endswith(".environment.json"):
            assert run_environment.sidecar_path(path).is_file()
            assert not run_environment.sidecar_problems(run_environment.sidecar_path(path))


def test_original_run_is_marked_transductive_and_resumes(world):
    assert run.main(cli(world)) == 0
    cell = world["out"] / "isic2019" / "B71__original.json"
    assert json.loads(cell.read_text())["result"]["transductivo"] is True
    before = cell.stat().st_mtime_ns
    assert run.main(cli(world)) == 0 and cell.stat().st_mtime_ns == before


def test_mislabelled_weights_abort_before_any_work(world):
    with pytest.raises(SystemExit, match="ABORT.*exclude_ids_sha256"):  # clean finals declared as original
        run.main(cli(world, root=world["roots"]["clean"]))
    with pytest.raises(SystemExit, match="ABORT.*exclude_ids_sha256"):  # original finals declared as clean
        run.main(cli(world, "--pool-excludes-val", config="A71", root=world["roots"]["original"]))
    assert not (world["out"] / "isic2019").exists()


def test_limit_writes_to_a_separate_smoke_directory(world):
    assert run.main(cli(world, "--limit", "16", "--allow-dirty", "--pool-excludes-val")) == 0
    assert (world["out"] / "isic2019_smoke_16" / "B71__clean.json").is_file()
    assert not (world["out"] / "isic2019").exists()


def test_a_failing_cell_is_listed_and_returns_nonzero(world, monkeypatch, capsys):
    monkeypatch.setattr(run, "build_extractor", lambda args, ds, bank: FakeExtractor(ds, always_fail=True))
    assert run.main(cli(world, "--pool-excludes-val")) == 1
    assert "FAILED B71__clean" in capsys.readouterr().out
