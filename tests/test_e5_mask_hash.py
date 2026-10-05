"""E5 verifies a pool mask with the SAME hash the pool builder wrote in mask_manifest.json (end to end)."""

import json

import numpy as np
import seg_paths  # noqa: F401
import export_tone_inputs as exporter
from PIL import Image

from dermssl import e5_analyze as an
from test_seg_export_tone_inputs import _build_cache


def _export(tmp_path):
    cache_dir, masks_path, _ = _build_cache(tmp_path)
    out = tmp_path / "tone_inputs"
    exporter.run(exporter.parse_args(["--cache-dir", str(cache_dir), "--pseudo-masks", str(masks_path),
                                      "--output-root", str(out), "--source-revision", "pool-final"]))
    entries = json.loads((out / "mask_manifest.json").read_text(encoding="utf-8"))
    return out, entries["masks"] if isinstance(entries, dict) else entries


def test_every_eligible_mask_written_by_the_real_exporter_matches_its_manifest_hash_in_e5(tmp_path):
    out, entries = _export(tmp_path)
    eligible = [e for e in entries if e["eligible"]]
    assert len(eligible) == 2
    for entry in eligible:
        ds, image_id = entry["dataset"], entry["image_id"]
        src = tmp_path / "src.png"
        Image.fromarray(np.zeros((40, 40, 3), np.uint8) + 120).save(src)
        task = an.Task(image_id, str(src), "train", "mel", pool_image=str(out / "originals" / ds / f"{image_id}.png"),
                       pool_mask=str(out / "masks" / ds / f"{image_id}.png"), mask_state="eligible",
                       mask_sha256=entry["mask_sha256"])
        assert an.analyze(task, masks=None, hair_fn=None)["roi"]["mask_sha_ok"] is True


def test_a_tampered_mask_png_no_longer_matches(tmp_path):
    out, entries = _export(tmp_path)
    entry = next(e for e in entries if e["eligible"])
    path = out / "masks" / entry["dataset"] / f"{entry['image_id']}.png"
    arr = np.asarray(Image.open(path)).copy()
    arr[0, 0] = 255 - arr[0, 0]
    Image.fromarray(arr).save(path)
    src = tmp_path / "src.png"
    Image.fromarray(np.zeros((40, 40, 3), np.uint8) + 120).save(src)
    task = an.Task(entry["image_id"], str(src), "train", "mel",
                   pool_image=str(out / "originals" / entry["dataset"] / f"{entry['image_id']}.png"),
                   pool_mask=str(path), mask_state="eligible", mask_sha256=entry["mask_sha256"])
    assert an.analyze(task, masks=None, hair_fn=None)["roi"]["mask_sha_ok"] is False
