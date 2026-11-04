"""Tests for the tone contact-sheet generator (Tarea 1B)."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from scripts.make_tone_contact_sheet import (
    _read_manifest,
    build_candidate_pool,
    render_contact_sheet,
    select_stratified_panels,
)


def _write_frame(path, value=(0.72, 0.58, 0.49), size=64, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, 0.03, size=(size, size, 1))
    img = np.clip(np.asarray(value)[None, None, :] * noise, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def _write_mask(path, size=64):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 255
    Image.fromarray(mask).save(path)


def test_manifest_resolves_relative_paths(tmp_path):
    (tmp_path / "img.png").touch()
    (tmp_path / "mask.png").touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image_path": "img.png", "mask_path": "mask.png"}) + "\n", encoding="utf-8"
    )
    rows = _read_manifest(manifest)
    assert len(rows) == 1
    assert rows[0]["image_path"] == str(tmp_path / "img.png")
    assert rows[0]["mask_path"] == str(tmp_path / "mask.png")


def test_build_candidate_pool_skips_missing_files(tmp_path):
    manifest_rows = [{"image_path": str(tmp_path / "missing.png"), "mask_path": str(tmp_path / "nope.png")}]
    pool = build_candidate_pool(manifest_rows, seed=42, policy_cap_degrees=None, min_pool_size=5)
    assert pool == []


def test_build_candidate_pool_finds_applied_cases(tmp_path):
    """Una imagen con tono tipico de piel produce al menos un caso aplicado en pocos pasos."""
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    _write_frame(img_path)
    _write_mask(mask_path)
    pool = build_candidate_pool(
        [{"image_path": str(img_path), "mask_path": str(mask_path)}],
        seed=42,
        policy_cap_degrees=None,
        min_pool_size=1,
    )
    assert len(pool) >= 1
    entry = pool[0]
    assert entry["ita_before"] is not None
    assert "ita_delta_realized" in entry
    assert entry["before_rgb"].shape == entry["after_rgb"].shape


def _fake_pool(deltas):
    return [
        {
            "ita_delta_realized": delta,
            "before_rgb": np.zeros((4, 4, 3)),
            "after_rgb": np.zeros((4, 4, 3)),
            "ita_before": 0.0,
            "ita_after": float(delta),
        }
        for delta in deltas
    ]


def test_select_stratified_panels_includes_both_extremes():
    deltas = list(range(-50, 51, 5))  # 21 candidates, spread wide
    pool = _fake_pool(deltas)
    panels = select_stratified_panels(pool, n_panels=6)
    realized = [p["ita_delta_realized"] for p in panels]
    assert min(realized) == min(deltas)
    assert max(realized) == max(deltas)
    assert realized == sorted(realized)


def test_select_stratified_panels_handles_empty_pool():
    assert select_stratified_panels([], n_panels=20) == []


def test_select_stratified_panels_does_not_exceed_pool_size():
    pool = _fake_pool([1.0, 2.0, 3.0])
    panels = select_stratified_panels(pool, n_panels=20)
    assert len(panels) <= 3


def test_render_contact_sheet_writes_a_file(tmp_path):
    pool = _fake_pool([-10.0, 0.0, 10.0])
    for row in pool:
        row["before_rgb"] = np.random.default_rng(0).random((8, 8, 3))
        row["after_rgb"] = np.random.default_rng(1).random((8, 8, 3))
        row["ita_objetivo"] = 0.0
        row["ita_after_status"] = "ok"
    output = tmp_path / "sheet.png"
    render_contact_sheet(pool, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_render_contact_sheet_labels_unmeasurable_output(tmp_path):
    pool = _fake_pool([5.0])
    pool[0]["ita_after"] = None
    pool[0]["before_rgb"] = np.zeros((8, 8, 3))
    pool[0]["after_rgb"] = np.zeros((8, 8, 3))
    output = tmp_path / "sheet.png"
    render_contact_sheet(pool, output)  # must not raise on a None ita_after
    assert output.exists()
