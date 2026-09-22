"""Every arm must record which data it trained on.

The orchestrator never passed --split-hash. The proposed arm refused to start
("split_hash es obligatorio para adaptacion transformada") and stopped the
chain; the control arm has no such guard, so it trained for six hours and
wrote sixty-four zeros into the field that identifies the data partition -
a run.json that looks complete and is not traceable.

These tests pin both halves: the flag reaches every arm, and the hash is
derived from the pool instead of being a constant that cannot notice change.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_stage2  # noqa: E402


def _settings(tmp_path: Path, archs=("dinov2_vits14",)) -> run_stage2.Settings:
    return run_stage2.Settings(
        repo=tmp_path / "repo", outputs=tmp_path / "out", python="python",
        ham_root=tmp_path / "ham", archs=list(archs),
    )


def _pool(settings: run_stage2.Settings, names=("ham/a.png", "ham/b.png")) -> Path:
    pool = settings.outputs / "tone" / "inputs" / "originals"
    for name in names:
        path = pool / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return pool


def test_every_arm_carries_the_split_hash_flag(tmp_path):
    settings = _settings(tmp_path, archs=("dinov2_vits14", "dinov2_vitb14"))
    arms = [s for s in run_stage2.all_stages(settings) if s.name.startswith("ssl_")]
    assert len(arms) == 4
    for arm in arms:
        assert "--split-hash" in arm.argv, f"{arm.name} entrenaria sin registrar la particion"
        assert run_stage2.SPLIT_HASH_TOKEN in arm.argv


def test_every_arm_carries_the_data_revision(tmp_path):
    """The control arm used to record data_revision: null while the proposed did not."""
    settings = _settings(tmp_path, archs=("dinov2_vits14", "dinov2_vitb14"))
    for arm in (s for s in run_stage2.all_stages(settings) if s.name.startswith("ssl_")):
        assert arm.argv.count("--data-revision") == 1, f"{arm.name}: falta o esta repetido"


def test_the_hash_is_derived_from_the_pool_contents(tmp_path):
    settings = _settings(tmp_path)
    pool = _pool(settings)
    first = run_stage2.compute_pool_split_hash(pool)
    assert len(first) == 64 and first != "0" * 64

    (pool / "ham" / "c.png").write_bytes(b"x")
    assert run_stage2.compute_pool_split_hash(pool) != first, (
        "un pool distinto debe dar un hash distinto, o el campo no identifica nada"
    )


def test_the_hash_is_stable_across_calls(tmp_path):
    settings = _settings(tmp_path)
    pool = _pool(settings)
    assert run_stage2.compute_pool_split_hash(pool) == run_stage2.compute_pool_split_hash(pool)


def test_an_empty_pool_is_refused(tmp_path):
    settings = _settings(tmp_path)
    empty = settings.outputs / "tone" / "inputs" / "originals"
    empty.mkdir(parents=True)
    with pytest.raises(ValueError, match="vacio"):
        run_stage2.compute_pool_split_hash(empty)


def test_the_receipt_is_written_once_and_reused(tmp_path):
    settings = _settings(tmp_path)
    _pool(settings)
    first = run_stage2.pool_split_hash(settings)
    receipt = settings.outputs / "tone" / "inputs" / run_stage2.POOL_HASH_FILE
    assert json.loads(receipt.read_text(encoding="utf-8"))["split_hash"] == first
    assert run_stage2.pool_split_hash(settings) == first


def test_a_pool_that_changed_mid_run_stops_the_chain(tmp_path):
    settings = _settings(tmp_path)
    pool = _pool(settings)
    run_stage2.pool_split_hash(settings)

    (pool / "ham" / "nueva.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="no son comparables"):
        run_stage2.pool_split_hash(settings)


def test_the_placeholder_is_replaced_at_launch(tmp_path):
    settings = _settings(tmp_path)
    _pool(settings)
    resolved = run_stage2.resolve_argv(
        ["python", "--split-hash", run_stage2.SPLIT_HASH_TOKEN], settings
    )
    assert run_stage2.SPLIT_HASH_TOKEN not in resolved
    assert len(resolved[-1]) == 64


def test_a_command_without_the_placeholder_is_untouched(tmp_path):
    settings = _settings(tmp_path)
    argv = ["python", "scripts/warm_hub_cache.py"]
    assert run_stage2.resolve_argv(argv, settings) is argv


def test_the_arms_of_one_run_share_the_same_hash(tmp_path):
    """A paired comparison across two pools is not a comparison."""
    settings = _settings(tmp_path, archs=("dinov2_vits14", "dinov2_vitb14"))
    _pool(settings)
    shift = settings.outputs / "tone" / "shift2"      # the paired arms read it too
    shift.mkdir(parents=True, exist_ok=True)
    (shift / "tone_transform_manifest.json").write_text(
        json.dumps({"revision": "r" * 64}), encoding="utf-8"
    )
    arms = [s for s in run_stage2.all_stages(settings) if s.name.startswith("ssl_")]
    hashes = set()
    for arm in arms:
        resolved = run_stage2.resolve_argv(arm.argv, settings)
        hashes.add(resolved[resolved.index("--split-hash") + 1])
    assert len(hashes) == 1
