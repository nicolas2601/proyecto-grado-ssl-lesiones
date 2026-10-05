"""E5 per-image analysis: one record from a file, with or without a lesion mask, plus the pool runner."""

import numpy as np
import pytest
from PIL import Image

from dermssl import e5_analyze as an

SKIN = (178, 133, 102)  # sRGB 8-bit, ITA ~21 (type III)


def _write(tmp_path, name, size=(160, 120), lesion=True):
    h, w = size[1], size[0]
    arr = np.empty((h, w, 3), dtype=np.uint8)
    arr[:] = SKIN
    yy, xx = np.mgrid[:h, :w]
    inside = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= 20**2
    if lesion:
        arr[inside] = (70, 40, 30)
    arr = np.clip(arr + np.random.RandomState(3).randint(-2, 3, arr.shape), 0, 255).astype(np.uint8)
    path = tmp_path / name
    Image.fromarray(arr).save(path, "PNG")
    return path, inside


def _task(path, i=0, mask_pos=None, split="train", label="mel"):
    return an.Task(image_id=f"i{i}", path=str(path), split=split, label=label, mask_pos=mask_pos)


def test_without_a_mask_the_record_has_global_estimates_and_a_roi_that_is_not_applicable(tmp_path):
    path, _ = _write(tmp_path, "a.png")
    rec = an.analyze(_task(path), masks=None, hair_fn=None)
    assert rec["image_id"] == "i0" and rec["split"] == "train" and rec["label"] == "mel"
    assert rec["otsu"]["bucket"] == "III" and rec["whole_image"]["status"] == "ok"
    assert rec["roi"]["status"] == "no_mask" and rec["roi"]["bucket"] is None
    assert "tercile" not in rec  # terciles need the whole dataset: assigned later


def test_with_a_mask_the_ring_estimator_runs_on_the_ring_outside_the_lesion(tmp_path):
    path, inside = _write(tmp_path, "b.png")
    masks = np.stack([np.zeros_like(inside, dtype=np.uint8), inside.astype(np.uint8)])
    rec = an.analyze(_task(path, mask_pos=1), masks=masks, hair_fn=None)
    assert rec["roi"]["status"] == "ok" and rec["roi"]["bucket"] == "III" and rec["roi"]["n_px"] > 300


def test_uncertain_mask_pixels_count_as_not_lesion(tmp_path):
    path, inside = _write(tmp_path, "c.png")
    mask = np.where(inside, 255, 0).astype(np.uint8)[None]  # only the uncertain label: no lesion at all
    rec = an.analyze(_task(path, mask_pos=0), masks=mask, hair_fn=None)
    assert rec["roi"]["status"] == "degenerate_mask"


def test_a_mask_on_another_grid_aborts_instead_of_misaligning(tmp_path):
    path, _ = _write(tmp_path, "d.png")
    with pytest.raises(an.MaskGridError, match="i0"):
        an.analyze(_task(path, mask_pos=0), masks=np.zeros((1, 50, 50), np.uint8), hair_fn=None)


def test_a_task_with_a_mask_position_but_no_masks_is_a_typed_error(tmp_path):
    path, _ = _write(tmp_path, "e.png")
    with pytest.raises(an.MaskGridError):
        an.analyze(_task(path, mask_pos=0), masks=None, hair_fn=None)


def test_an_unreadable_file_becomes_an_unreadable_record_not_a_crash(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    rec = an.analyze(_task(bad), masks=None, hair_fn=None)
    assert rec["otsu"]["status"] == "unreadable" and rec["roi"]["status"] == "unreadable" and "bad.png" in rec["error"]


def test_records_are_json_safe(tmp_path):
    import json

    path, _ = _write(tmp_path, "f.png")
    json.dumps(an.analyze(_task(path), masks=None, hair_fn=None), allow_nan=False)


def test_the_pool_returns_the_same_records_in_the_same_order_for_any_worker_count(tmp_path):
    tasks = [_task(_write(tmp_path, f"p{i}.png")[0], i) for i in range(7)]
    serial = an.run_tasks(tasks, workers=1)
    parallel = an.run_tasks(tasks, workers=3, chunk_size=2)
    assert serial == parallel and [r["image_id"] for r in serial] == [f"i{i}" for i in range(7)]


def test_the_pool_reads_masks_from_the_npy_in_each_worker(tmp_path):
    path, inside = _write(tmp_path, "m.png")
    np.save(tmp_path / "masks.npy", inside.astype(np.uint8)[None])
    recs = an.run_tasks([_task(path, mask_pos=0)], workers=2, masks_path=tmp_path / "masks.npy")
    assert recs[0]["roi"]["status"] == "ok"


def test_the_hair_detector_is_required_when_asked_and_missing(monkeypatch):
    monkeypatch.setattr(an, "_nucleo_dir", lambda: an.Path("/nonexistent"))
    with pytest.raises(an.HairDetectorUnavailable):
        an.load_hair_fn()


def test_the_pool_builds_the_real_hair_detector_once_per_worker_when_asked(tmp_path):
    pytest.importorskip("cv2")
    path, inside = _write(tmp_path, "h.png")
    np.save(tmp_path / "masks.npy", inside.astype(np.uint8)[None])
    recs = an.run_tasks([_task(path, mask_pos=0)], workers=2, masks_path=tmp_path / "masks.npy", use_hair=True)
    assert recs[0]["roi"]["status"] == "ok" and recs[0]["roi"]["bucket"] == "III"


def test_the_hair_detector_parameters_are_read_from_the_module_that_runs():
    pytest.importorskip("cv2")
    params = an.hair_parameters()
    assert params["module"].endswith("hair_removal.py") and params["KERNEL"] == 9 and "MIN_HAIR_LENGTH" in params


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_is_reported_every_interval_and_at_the_end(tmp_path, workers):
    tasks = [_task(_write(tmp_path, f"g{i}.png")[0], i) for i in range(5)]
    seen = []
    an.run_tasks(tasks, workers=workers, progress=lambda done, total: seen.append((done, total)), progress_every=2)
    assert seen == [(2, 5), (4, 5), (5, 5)]
