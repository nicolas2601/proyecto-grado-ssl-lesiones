"""End-to-end smoke of every Camino B stage on 64 synthetic images (needs torch)."""

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

pytestmark = pytest.mark.torch

import seg_paths  # noqa: E402

DRIVER = seg_paths.CAMINO_B / "entrenamiento" / "run_pipeline.py"


def test_smoke_runs_every_stage_and_writes_the_small_artifacts(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "--smoke", "--workdir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    for relative in (
        "cache/images.npy",
        "moco/backbone.pt",
        "ccam/saliency.npy",
        "pseudo_r0/pseudo_masks.npy",
        "student_r1/best.pt",
        "pseudo_r1/pseudo_masks.npy",
        "student_r2/best.pt",
        "evaluacion_synthetic/per_image.csv",
    ):
        assert (tmp_path / relative).is_file(), relative
    summary = json.loads((tmp_path / "evaluacion_synthetic" / "summary.json").read_text("utf-8"))
    assert summary["meta"]["threshold"] == 0.5
    assert summary["meta"]["n_images"] == 16
