"""Pure tests for sampling, checkpoint selection, logging, presets, synthetic data."""

import json

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import check_env
import presets
import run_logging
import sampling
import selection
import synthetic


# ---------------------------------------------------------------- sampling
def test_training_batches_are_disjoint_across_ranks_and_cover_the_same_steps():
    indices = np.arange(100)
    per_rank = [
        sampling.shard_batches(indices, batch_size=8, rank=r, world_size=2, epoch=0, seed=1)
        for r in range(2)
    ]
    assert len(per_rank[0]) == len(per_rank[1]) == 6  # 100 // 16
    seen = np.concatenate([b for rank_batches in per_rank for b in rank_batches])
    assert len(seen) == len(set(seen.tolist())) == 96
    assert all(len(b) == 8 for rank_batches in per_rank for b in rank_batches)


def test_training_order_is_deterministic_and_changes_with_epoch():
    indices = np.arange(64)
    first = sampling.shard_batches(indices, 8, 0, 1, epoch=0, seed=5)
    again = sampling.shard_batches(indices, 8, 0, 1, epoch=0, seed=5)
    other = sampling.shard_batches(indices, 8, 0, 1, epoch=1, seed=5)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))


def test_batches_are_sorted_for_memmap_locality():
    batches = sampling.shard_batches(np.arange(64), 16, 0, 1, epoch=0, seed=0)
    assert all(np.all(np.diff(b) > 0) for b in batches)


def test_too_few_images_for_one_global_batch_raises():
    with pytest.raises(ValueError):
        sampling.shard_batches(np.arange(10), 8, 0, 2, epoch=0, seed=0)


def test_evaluation_batches_keep_every_index_exactly_once():
    indices = np.arange(37)
    seen = np.concatenate(
        [
            b
            for r in range(3)
            for b in sampling.shard_batches(
                indices, 4, r, 3, epoch=0, seed=0, shuffle=False, drop_last=False
            )
        ]
    )
    assert sorted(seen.tolist()) == list(range(37))


def test_steps_per_epoch_matches_batches():
    assert sampling.steps_per_epoch(100, batch_size=8, world_size=2) == 6


# --------------------------------------------------------------- selection
def test_selector_tracks_lowest_validation_loss():
    selector = selection.CheckpointSelector("best-val")
    assert selector.update(0, 1.0) is True
    assert selector.update(1, 1.5) is False
    assert selector.update(2, 0.4) is True
    assert selector.best_epoch == 2 and selector.best_value == pytest.approx(0.4)


def test_selector_last_mode_always_selects_latest_epoch():
    selector = selection.CheckpointSelector("last")
    assert selector.update(0, 1.0) is True
    assert selector.update(1, 5.0) is True
    assert selector.best_epoch == 1


def test_selector_ignores_non_finite_values_in_best_mode():
    selector = selection.CheckpointSelector("best-val")
    selector.update(0, 1.0)
    assert selector.update(1, float("nan")) is False
    assert selector.best_epoch == 0


def test_selector_state_roundtrip_supports_resume():
    selector = selection.CheckpointSelector("best-val")
    selector.update(0, 2.0)
    selector.update(1, 1.0)
    restored = selection.CheckpointSelector.from_state(selector.state_dict())
    assert restored.update(2, 1.2) is False
    assert restored.best_epoch == 1


def test_selector_rejects_unknown_mode():
    with pytest.raises(ValueError):
        selection.CheckpointSelector("best-dice")


# ----------------------------------------------------------------- logging
def test_jsonl_logger_appends_valid_lines(tmp_path):
    path = tmp_path / "logs" / "metrics.jsonl"
    logger = run_logging.JsonlLogger(path, echo=False)
    logger.log({"epoch": 0, "loss": 1.5})
    logger.log({"epoch": 1, "loss": 1.0})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [0, 1]


def test_throughput_meter_uses_injected_clock():
    clock = iter([10.0, 14.0])
    meter = run_logging.ThroughputMeter(clock=lambda: next(clock))
    meter.start()
    meter.add(400)
    snapshot = meter.snapshot()
    assert snapshot["seconds"] == pytest.approx(4.0)
    assert snapshot["images"] == 400
    assert snapshot["images_per_second"] == pytest.approx(100.0)


def test_timing_summary_aggregates_epoch_records():
    records = [
        {"seconds": 10.0, "images": 1000},
        {"seconds": 20.0, "images": 1000},
    ]
    summary = run_logging.timing_summary(records)
    assert summary["epochs"] == 2
    assert summary["total_seconds"] == pytest.approx(30.0)
    assert summary["mean_seconds_per_epoch"] == pytest.approx(15.0)
    assert summary["images_processed"] == 2000
    assert summary["images_per_second"] == pytest.approx(2000 / 30.0)


def test_timing_summary_of_nothing_is_empty_but_valid():
    assert run_logging.timing_summary([])["epochs"] == 0


# ----------------------------------------------------------------- presets
def test_flags_from_mapping_translates_types():
    flags = presets.flags_from_mapping(
        {"epochs": 10, "lr": 0.001, "use_flag": True, "skip": False, "none": None,
         "name": "x", "roots": ["a=1", "b=2"]}
    )
    assert flags == [
        "--epochs", "10", "--lr", "0.001", "--use-flag", "--name", "x",
        "--roots", "a=1", "--roots", "b=2",
    ]


def test_load_preset_reads_stage_sections(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"student": {"epochs": 3}}), encoding="utf-8")
    assert presets.load_preset(path)["student"]["epochs"] == 3


def test_shipped_presets_are_valid_json_with_all_stages():
    for name in ("piloto.json", "final.json"):
        preset = presets.load_preset(seg_paths.COMUN / "configs" / name)
        assert {"pipeline", "cache", "contrastive", "ccam", "pseudo", "student"} <= set(preset)


# --------------------------------------------------------------- synthetic
def test_synthetic_lesion_is_deterministic_and_plausible():
    image, mask = synthetic.make_lesion_image(np.random.default_rng(0), 64)
    image2, mask2 = synthetic.make_lesion_image(np.random.default_rng(0), 64)
    assert image.shape == (64, 64, 3) and image.dtype == np.uint8
    assert mask.shape == (64, 64) and mask.dtype == bool
    assert np.array_equal(image, image2) and np.array_equal(mask, mask2)
    assert 0.05 < mask.mean() < 0.6
    assert image[mask].mean() < image[~mask].mean()


def test_write_eval_set_creates_image_mask_pairs(tmp_path):
    ids = synthetic.write_eval_set(tmp_path / "img", tmp_path / "msk", count=3, size=32, seed=1)
    assert len(ids) == 3
    for image_id in ids:
        assert (tmp_path / "img" / f"{image_id}.png").is_file()
        mask = np.asarray(Image.open(tmp_path / "msk" / f"{image_id}_segmentation.png"))
        assert set(np.unique(mask)) <= {0, 255}


def test_write_synthetic_pool_images(tmp_path):
    paths = synthetic.write_pool_images(tmp_path, count=4, size=32, seed=0)
    assert len(paths) == 4 and all(p.is_file() for p in paths)


# --------------------------------------------------------------- check_env
def test_collect_environment_reports_core_fields_without_torch():
    info = check_env.collect_environment()
    assert "python" in info and "numpy" in info and "torch" in info
    assert "shm_free_mb" in info
