"""Paired image-by-image bootstrap between two segmenters' per_image.csv files."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from scripts import seg_paired_bootstrap as spb
from env_sidecar import sidecar_env  # noqa: E402


def _write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "dice", "hd95"])
        writer.writeheader()
        writer.writerows(rows)


def _rows(dice, hd95, prefix="ISIC_"):
    return [{"image_id": f"{prefix}{i:07d}", "dice": d, "hd95": h}
            for i, (d, h) in enumerate(zip(dice, hd95))]


def test_a_uniform_gain_excludes_zero_and_reads_better(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.4, 0.8, 300)
    _write(tmp_path / "a.csv", _rows(base + 0.05, 20 - 3 + rng.normal(0, 0.1, 300)))
    _write(tmp_path / "b.csv", _rows(base, np.full(300, 20.0)))
    out = spb.compare(tmp_path / "a.csv", tmp_path / "b.csv", n_resamples=500, seed=1)
    assert out["dice"]["delta"] == pytest.approx(0.05)
    assert out["dice"]["ci95_low"] > 0 and out["dice"]["direction"] == "better"
    assert out["hd95"]["delta"] < 0 and out["hd95"]["direction"] == "better"
    assert out["n_paired"] == 300


def test_pairs_by_image_id_not_by_row_order(tmp_path):
    a = _rows([0.9, 0.5], [5.0, 9.0])
    _write(tmp_path / "a.csv", list(reversed(a)))
    _write(tmp_path / "b.csv", _rows([0.9, 0.5], [5.0, 9.0]))
    out = spb.compare(tmp_path / "a.csv", tmp_path / "b.csv", n_resamples=50, seed=0)
    assert out["dice"]["delta"] == 0.0 and out["dice"]["direction"] == "indistinguishable"


def test_mismatched_image_sets_are_refused(tmp_path):
    _write(tmp_path / "a.csv", _rows([0.9], [5.0]))
    _write(tmp_path / "b.csv", _rows([0.9], [5.0], prefix="HAM_"))
    with pytest.raises(ValueError, match="same images"):
        spb.compare(tmp_path / "a.csv", tmp_path / "b.csv", n_resamples=10, seed=0)


def test_non_finite_hd95_pairs_are_dropped_and_counted(tmp_path):
    _write(tmp_path / "a.csv", _rows([0.9, 0.8, 0.7], [5.0, float("inf"), 7.0]))
    _write(tmp_path / "b.csv", _rows([0.9, 0.8, 0.7], [5.0, 6.0, float("nan")]))
    out = spb.compare(tmp_path / "a.csv", tmp_path / "b.csv", n_resamples=10, seed=0)
    assert out["hd95"]["n"] == 1 and out["hd95"]["n_dropped_non_finite"] == 2
    assert out["dice"]["n"] == 3


def test_cli_writes_json(tmp_path):
    _write(tmp_path / "a.csv", _rows([0.9, 0.6], [5.0, 8.0]))
    _write(tmp_path / "b.csv", _rows([0.8, 0.6], [6.0, 8.0]))
    out_path = tmp_path / "out.json"
    spb.main(["--a", str(tmp_path / "a.csv"), "--label-a", "A2", "--b", str(tmp_path / "b.csv"),
              "--label-b", "student_r3", "--n-resamples", "20", "--output-json", str(out_path)])
    doc = json.loads(out_path.read_text())
    assert doc["label_a"] == "A2" and doc["label_b"] == "student_r3"
    assert doc["rng_seed"] == spb.DEFAULT_SEED and set(doc["metrics"]) == {"dice", "hd95"}
    assert sidecar_env(out_path)["schema"] == 1
