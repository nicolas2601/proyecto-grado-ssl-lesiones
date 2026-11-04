"""Tests for the pool eligibility triage (Fase 3)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from scripts.audit_pool_tone_eligibility import (
    CRITERION_7_FLOOR,
    _read_manifest,
    analyze_row,
    render_report,
    summarize,
)


def _write_frame(path, value, size=64, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, 0.03, size=(size, size, 1))
    img = np.clip(np.asarray(value)[None, None, :] * noise, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def _write_mask(path, size=64):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 255
    Image.fromarray(mask).save(path)


def test_manifest_resolves_relative_paths_and_keeps_label(tmp_path):
    (tmp_path / "img.png").touch()
    (tmp_path / "mask.png").touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image_path": "img.png", "mask_path": "mask.png", "label": 1}) + "\n",
        encoding="utf-8",
    )
    rows = _read_manifest(manifest)
    assert rows[0]["label"] == 1
    assert rows[0]["image_path"] == str(tmp_path / "img.png")


def test_analyze_row_returns_none_for_missing_files(tmp_path):
    row = {"image_path": str(tmp_path / "missing.png"), "mask_path": str(tmp_path / "nope.png")}
    assert analyze_row(row, seed=42, policy_cap_degrees=None) is None


def test_analyze_row_flags_measurable_skin_tone(tmp_path):
    img_path, mask_path = tmp_path / "img.png", tmp_path / "mask.png"
    _write_frame(img_path, value=(0.72, 0.58, 0.49))
    _write_mask(mask_path)
    result = analyze_row(
        {"image_path": str(img_path), "mask_path": str(mask_path)}, seed=42, policy_cap_degrees=None
    )
    assert result is not None
    assert result["measurable"] is True
    assert result["status"] == "ok"
    assert np.isfinite(result["b_median"])


def test_analyze_row_flags_unmeasurable_near_gray_skin(tmp_path):
    img_path, mask_path = tmp_path / "img.png", tmp_path / "mask.png"
    _write_frame(img_path, value=(0.5, 0.5, 0.5))
    _write_mask(mask_path)
    result = analyze_row(
        {"image_path": str(img_path), "mask_path": str(mask_path)}, seed=42, policy_cap_degrees=None
    )
    assert result is not None
    assert result["measurable"] is False
    assert result["reachable_without_mix"] is False
    assert result["applied_with_mix"] is False


def test_analyze_row_carries_label_and_its_absence(tmp_path):
    img_path, mask_path = tmp_path / "img.png", tmp_path / "mask.png"
    _write_frame(img_path, value=(0.72, 0.58, 0.49))
    _write_mask(mask_path)
    with_label = analyze_row(
        {"image_path": str(img_path), "mask_path": str(mask_path), "label": 1},
        seed=42,
        policy_cap_degrees=None,
    )
    without_label = analyze_row(
        {"image_path": str(img_path), "mask_path": str(mask_path)}, seed=42, policy_cap_degrees=None
    )
    assert with_label["has_label"] is True
    assert without_label["has_label"] is False


def _fake_rows(n_measurable, n_unmeasurable, *, labels_pos_measurable=0, labels_pos_unmeasurable=0):
    rows = []
    for i in range(n_measurable):
        rows.append(
            {
                "lesion_fraction": 0.1 + 0.01 * i,
                "b_median": 15.0,
                "measurable": True,
                "status": "ok",
                "reachable_without_mix": True,
                "population_shift_ok": True,
                "reachable_ita_span": 60.0,
                "applied_with_mix": i % 2 == 0,
                "label": 1 if i < labels_pos_measurable else 0,
                "has_label": True,
            }
        )
    for i in range(n_unmeasurable):
        rows.append(
            {
                "lesion_fraction": 0.2 + 0.01 * i,
                "b_median": 1.0,
                "measurable": False,
                "status": "ita_indeterminate",
                "reachable_without_mix": False,
                "population_shift_ok": None,
                "reachable_ita_span": None,
                "applied_with_mix": False,
                "label": 1 if i < labels_pos_unmeasurable else 0,
                "has_label": True,
            }
        )
    return rows


def test_summarize_reports_measurability_and_criterion_7():
    rows = _fake_rows(30, 10)  # 75% measurable
    summary = summarize(rows)
    assert summary["n_total"] == 40
    assert summary["measurable_rate"] == 0.75
    # applied_with_mix alternates True/False among the 30 measurable rows
    assert summary["effective_intervention_rate_with_mix"] == 15 / 40
    assert summary["sustains_criterion_7"] == (15 / 40 >= CRITERION_7_FLOOR)
    assert summary["sustains_criterion_7"] is False


def test_summarize_fisher_test_detects_class_imbalance():
    # measurable: 2/20 positive, unmeasurable: 8/20 positive -- matches the real finding's direction
    rows = _fake_rows(20, 20, labels_pos_measurable=2, labels_pos_unmeasurable=8)
    summary = summarize(rows)
    mv = summary["measurability_vs_label"]
    assert mv["positive_rate_measurable"] < mv["positive_rate_unmeasurable"]
    assert mv["odds_ratio"] < 1.0
    assert summary["label_coverage_fraction"] == 1.0


def test_summarize_declares_partial_label_coverage():
    rows = _fake_rows(10, 10)
    for row in rows[:15]:
        row["has_label"] = False
    summary = summarize(rows)
    assert summary["label_coverage_n"] == 5
    assert summary["label_coverage_fraction"] == pytest.approx(5 / 20)


def test_summarize_handles_no_labels_at_all():
    rows = _fake_rows(5, 5)
    for row in rows:
        row["has_label"] = False
    summary = summarize(rows)
    assert "measurability_vs_label" not in summary


def test_render_report_has_no_emoji_and_covers_all_sections():
    rows = _fake_rows(20, 10, labels_pos_measurable=3, labels_pos_unmeasurable=5)
    summary = summarize(rows)
    report = render_report(summary, manifest_path="fake.jsonl", seed=42, policy_cap_degrees=None)
    for header in ("1. Distribucion", "2. Tasa de medibilidad", "3. Contraste", "4. Medibilidad", "5. Recorrido"):
        assert header in report
    # No emoji: every character stays in the Basic Multilingual Plane's ASCII/Latin range used by the rest of the repo.
    assert all(ord(ch) < 0x2190 for ch in report)
