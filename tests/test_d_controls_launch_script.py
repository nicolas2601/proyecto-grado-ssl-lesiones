"""scripts/d_controls_launch.py: refuses while pretrain.py ignores --initial-weights, then prints D2 and D3."""

from __future__ import annotations

import json

import pytest

from dermssl import pad_target_split as ts
from scripts import d_controls_launch as launch


@pytest.fixture
def env(tmp_path):
    pool = tmp_path / "originals"
    pool.mkdir()
    for i in range(3):
        (pool / f"ISIC_{i}.jpg").write_bytes(b"x" * (i + 1))
    pad = tmp_path / "pad" / "images"
    pad.mkdir(parents=True)
    ids = ["PAT_1_1_1", "PAT_2_2_2"]
    for i in ids:
        (pad / f"{i}.png").write_bytes(b"p")
    split = tmp_path / "split.json"
    test_ids = ["PAT_9_9_9"]
    split.write_text(json.dumps({
        "target_adapt": {"img_ids": ids}, "target_test": {"img_ids": test_ids},
        "sha256": {"target_adapt": ts.ids_sha256(ids), "target_test": ts.ids_sha256(test_ids)}}))
    weights, exclude = tmp_path / "w.pt", tmp_path / "exclude.txt"
    weights.write_bytes(b"w")
    exclude.write_text("ISIC_0\n")
    base = ["--backbone-weights", str(weights), "--pool-root", str(pool), "--pad-root", str(tmp_path / "pad"),
            "--split", str(split), "--exclude-ids-file", str(exclude), "--work-dir", str(tmp_path / "work"),
            "--runs-dir", str(tmp_path / "runs")]
    return base, tmp_path


def test_it_refuses_by_default_and_creates_nothing(env, capsys):
    argv, tmp = env
    assert launch.main(argv) == 2
    assert "initial-weights" in capsys.readouterr().err
    assert not (tmp / "work").exists()


def test_when_confirmed_it_builds_the_roots_and_prints_two_commands(env, capsys):
    argv, tmp = env
    assert launch.main([*argv, "--init-weights-honored"]) == 0
    out = capsys.readouterr().out
    assert out.count("scripts/pretrain.py") == 2
    assert f"--image-root {tmp / 'work' / 'd2_root'}" in out and f"--image-root {tmp / 'work' / 'd3_root'}" in out
    assert "D2_s42" in out and "D3_s42" in out
    assert "differ only in --image-root, --output-dir: OK" in out
    assert (tmp / "work" / "d3_root" / "pad_target_adapt" / "PAT_1_1_1.png").is_symlink()


def test_it_never_reads_target_test(env):
    argv, tmp = env
    launch.main([*argv, "--init-weights-honored"])
    assert not (tmp / ts.LOCK_NAME).exists()


def test_dry_run_prints_without_touching_the_disk(env, capsys):
    argv, tmp = env
    assert launch.main([*argv, "--init-weights-honored", "--dry-run"]) == 0
    assert "scripts/pretrain.py" in capsys.readouterr().out
    assert not (tmp / "work").exists()


def test_a_missing_weights_file_aborts(env, capsys):
    argv, tmp = env
    (tmp / "w.pt").unlink()
    assert launch.main([*argv, "--init-weights-honored"]) == 1
    assert "w.pt" in capsys.readouterr().err
