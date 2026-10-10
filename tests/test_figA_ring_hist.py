"""Pod job for Figure A: the accumulator must hold exactly the pixels the E5 ring estimator reads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import figA_ring_hist as job  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state():
    job._reset()
    yield
    job._reset()


def test_accumulate_keeps_invariants_and_counts_out_of_range():
    lab = np.array([[50.0, 10.0, 20.0], [100.0, 0.0, 0.0], [101.0, 0.0, 0.0], [50.0, 200.0, 0.0]])
    job.accumulate(lab)
    assert job._STATE["n_pixels"] == 2 and job._STATE["out_of_range"] == 2 and job._STATE["n_images"] == 1
    assert job._STATE["L"].sum() == job._STATE["ab"].sum() == 2


def test_grids_match_the_loader_contract():
    sys.path.insert(0, str(REPO / "scripts" / "figures"))
    import fig_a_cielab_ring as figure  # noqa: PLC0415

    assert tuple(job.L_EDGES) == figure.EXPECTED_L_EDGES and tuple(job.AB_EDGES) == figure.EXPECTED_AB_EDGES


def test_ring_estimates_are_returned_unchanged_and_ring_pixels_are_the_valid_ones(tmp_path, monkeypatch):
    from dermssl import e5_roi  # noqa: PLC0415

    img = np.full((120, 160, 3), 0.7, dtype=np.float64)
    img[..., 1] = 0.55
    img[..., 2] = 0.45
    lesion = np.zeros((120, 160), dtype=bool)
    lesion[50:70, 70:90] = True
    baseline = e5_roi.ring_estimate(img, lesion)
    monkeypatch.setenv(job.OUT_ENV, str(tmp_path))
    originals = (e5_roi.ring_estimate, e5_roi.border_estimate, e5_roi._read_skin)
    try:
        job.install_patches()
        patched = e5_roi.ring_estimate(img, lesion)
        fallback = e5_roi.border_estimate(img)
        assert patched == baseline
        assert baseline.status == "ok" and job._STATE["n_images"] == 1
        assert job._STATE["n_pixels"] == baseline.n_px
        assert fallback.status in ("ok", "ring_too_small", "ita_indeterminate") and job._STATE["n_images"] == 1  # border is not the ring
    finally:
        e5_roi.ring_estimate, e5_roi.border_estimate, e5_roi._read_skin = originals
    parts = list((tmp_path / "parts").glob("part_*.npz"))
    assert len(parts) == 1
