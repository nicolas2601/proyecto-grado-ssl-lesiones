"""S-diag pod script: token extraction -> masks the external evaluator scores.

A fake DINOv2 turns each 14x14 patch into a token of its mean colour, so a
bright, patch-aligned square must come back as a perfect mask, and the real
``evaluate_segmentation.py`` must read those masks as external predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import evaluate_segmentation as ev
from scripts import s_diag_parallel_check as pc
from scripts import s_diag_predict as sp

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SIZE = 224
SQUARE = slice(84, 140)  # patches 6..9 of the 16x16 grid


class FakeDino(torch.nn.Module):
    """forward_features like DINOv2: one token per 14x14 patch."""

    def forward_features(self, batch):
        pooled = torch.nn.functional.avg_pool2d(batch, 14).flatten(2).transpose(1, 2)
        ones = torch.ones(*pooled.shape[:2], 1)
        return {"x_norm_patchtokens": torch.cat([pooled, ones], dim=2)}


def _square_mask():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[SQUARE, SQUARE] = True
    return mask


def _write_set(tmp_path, count=3):
    images, references = tmp_path / "img", tmp_path / "gt"
    images.mkdir()
    references.mkdir()
    for index in range(count):
        image = np.zeros((SIZE, SIZE, 3), np.uint8)
        image[_square_mask()] = 230
        Image.fromarray(image).save(images / f"ISIC_{index:07d}.jpg", quality=100)
        mask = _square_mask().astype(np.uint8) * 255
        Image.fromarray(mask).save(references / f"ISIC_{index:07d}_segmentation.png")
    return images, references


@pytest.fixture
def fake_backbone(monkeypatch):
    calls = []

    def build(arch, source, resolution, device):
        calls.append((arch, source, resolution))
        return {"model": FakeDino().eval()}

    monkeypatch.setattr(sp.bank, "build_frozen_backbone", build)
    monkeypatch.setattr(sp.rp, "clean_revision", lambda repo: "c0ffee")
    return calls


def _run(tmp_path, images, *methods, extra=(), name="pred"):
    out = tmp_path / name
    argv = ["--backbone", "public=dinov2_vitb14:public", "--image-dir", str(images),
            "--output-dir", str(out), "--reference-set", "isic2018", "--device", "cpu"]
    for method in methods:
        argv += ["--method", method]
    sp.main([*argv, *extra])
    return out


def _write_noisy_set(tmp_path, count):
    """Textured images whose PCA and k-means masks are not trivial."""
    images = tmp_path / "noisy"
    images.mkdir()
    rng = np.random.default_rng(7)
    for index in range(count):
        image = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
        top, left = rng.integers(20, 120, 2)
        image[top : top + 80, left : left + 80] //= 3
        Image.fromarray(image).save(images / f"ISIC_{index:07d}.png")
    return images


def _evaluate(tmp_path, out, images, references, method):
    return ev.evaluate(ev.parse_args([
        "--pred-mask-dir", str(out / "masks"), "--pred-mask-suffix", f"_{method}",
        "--image-dir", str(images), "--mask-dir", str(references),
        "--output-dir", str(tmp_path / f"eval_{method}"), "--hausdorff",
    ]))


def test_input_resolution_is_fixed_per_method():
    assert sp.input_resolution(["primary", "kmeans"]) == 224
    assert sp.input_resolution(["pca448"]) == 448
    with pytest.raises(ValueError, match="one input resolution"):
        sp.input_resolution(["primary", "pca448"])


@pytest.mark.parametrize("method", ["primary", "kmeans", "pca448"])
def test_the_evaluator_scores_a_planted_square_as_perfect(tmp_path, fake_backbone, method):
    images, references = _write_set(tmp_path)
    out = _run(tmp_path, images, method)
    summary = _evaluate(tmp_path, out, images, references, method)
    assert summary["dice"]["mean"] == pytest.approx(1.0)
    assert summary["hd95"]["mean"] == pytest.approx(0.0)
    assert fake_backbone == [("dinov2_vitb14", "public", sp.input_resolution([method]))]


def test_masks_are_binary_pngs_at_the_evaluation_grid(tmp_path, fake_backbone):
    images, _ = _write_set(tmp_path, count=1)
    out = _run(tmp_path, images, "primary", "kmeans")
    for method in ("primary", "kmeans"):
        with Image.open(out / "masks" / f"ISIC_0000000_{method}.png") as mask:
            array = np.asarray(mask)
        assert mask.mode == "L" and array.shape == (SIZE, SIZE)
        assert set(np.unique(array)) == {0, 255}


def test_run_json_records_the_provenance(tmp_path, fake_backbone):
    images, _ = _write_set(tmp_path, count=2)
    record = json.loads((_run(tmp_path, images, "primary") / "run.json").read_text())
    assert record["methods"] == ["primary"]
    assert record["environment"]["knobs"]["workers"] == record["workers"]
    assert record["input_resolution"] == 224 and record["mask_resolution"] == 224
    assert record["backbone"] == {"name": "public", "spec": "dinov2_vitb14:public",
                                  "weights_sha256": "public"}
    assert record["code_revision"] == "c0ffee"
    assert record["reference_set"] == "isic2018" and record["n_images"] == 2
    assert "no training" in record["training_images"]
    assert record["complete"] is True


def test_limit_keeps_the_first_images_in_sorted_order(tmp_path, fake_backbone):
    images, _ = _write_set(tmp_path, count=3)
    out = _run(tmp_path, images, "primary", extra=["--limit", "2"])
    assert sorted(p.name for p in (out / "masks").iterdir()) == [
        "ISIC_0000000_primary.png", "ISIC_0000001_primary.png"]


def test_an_existing_output_is_never_overwritten(tmp_path, fake_backbone):
    images, _ = _write_set(tmp_path, count=1)
    (tmp_path / "pred").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        _run(tmp_path, images, "primary")


def test_a_dirty_tree_aborts_before_loading_weights(tmp_path, fake_backbone, monkeypatch):
    def dirty(repo):
        raise sp.rp.DirtyTreeError("dirty")

    monkeypatch.setattr(sp.rp, "clean_revision", dirty)
    images, _ = _write_set(tmp_path, count=1)
    with pytest.raises(SystemExit, match="dirty"):
        _run(tmp_path, images, "primary")
    assert fake_backbone == []


def test_weights_digest_hashes_a_file_and_names_the_public_weights(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    assert sp.weights_digest("public") == "public"
    assert sp.weights_digest(str(weights)) == hashlib.sha256(b"weights").hexdigest()


def test_importing_the_script_does_not_import_torch():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    code = "import sys; import scripts.s_diag_predict; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True, env=env, cwd=ROOT)
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("methods", [("primary", "kmeans"), ("pca448",)])
def test_parallel_workers_write_bit_identical_masks(tmp_path, fake_backbone, methods):
    images = _write_noisy_set(tmp_path, count=7)
    extra = ["--batch-size", "3"]
    serial = _run(tmp_path, images, *methods, extra=extra, name="serial")
    parallel = _run(tmp_path, images, *methods, extra=[*extra, "--workers", "4"],
                    name="parallel")
    assert pc.compare_runs(serial, parallel) == []
    assert len(list((parallel / "masks").iterdir())) == 7 * len(methods)
    workers = [json.loads((out / "run.json").read_text())["workers"] for out in (serial, parallel)]
    assert workers == [1, 4]


def test_the_check_reports_any_mask_or_record_difference(tmp_path, fake_backbone):
    images = _write_noisy_set(tmp_path, count=2)
    first = _run(tmp_path, images, "primary", name="first")
    second = _run(tmp_path, images, "primary", extra=["--workers", "2"], name="second")
    assert pc.compare_runs(first, second) == []
    (second / "masks" / "ISIC_0000001_primary.png").write_bytes(b"not a mask")
    (second / "masks" / "ISIC_0000009_primary.png").write_bytes(b"extra")
    record = json.loads((second / "run.json").read_text())
    (second / "run.json").write_text(json.dumps({**record, "n_images": 3}))
    assert pc.compare_runs(first, second) == [
        "mask differs: ISIC_0000001_primary.png",
        "mask only in one run: ISIC_0000009_primary.png",
        "run.json differs: n_images",
    ]


def test_workers_must_be_positive(capsys):
    with pytest.raises(SystemExit):
        sp.parse_args(["--backbone", "public=dinov2_vitb14:public", "--image-dir", "i",
                       "--output-dir", "o", "--reference-set", "isic2018",
                       "--method", "primary", "--workers", "0"])
    assert "--workers must be >= 1" in capsys.readouterr().err


def test_pool_tasks_live_in_an_importable_module(monkeypatch):
    """On the pod the script runs as ``__main__``; loky workers cannot resolve
    ``__main__`` globals, so what goes to the pool must come from dermssl."""
    submitted = []

    class RecordingExecutor:
        def submit(self, fn, *args):
            submitted.append(fn)

            class Done:
                def result(self):
                    return None

            return Done()

        def shutdown(self, wait=True):
            return None

    import joblib.externals.loky as loky

    monkeypatch.setattr(loky, "get_reusable_executor", lambda **kwargs: RecordingExecutor())
    monkeypatch.setattr(sp, "wait", lambda pending, **kwargs: type("W", (), {"done": set()})())
    with sp.mask_writer(2) as submit:
        submit(Path("m"), "ISIC_0000000", np.zeros((256, 4)), ["primary"], 16)
    assert [fn.__module__ for fn in submitted] == ["dermssl.s_diag"]


def test_serial_masks_use_one_cpu_thread_like_the_workers(tmp_path, fake_backbone, monkeypatch):
    """Multi-threaded BLAS moves the PCA by ~1e-16, so serial pins one thread too."""
    from threadpoolctl import threadpool_info

    threads, segment = [], sp.sd.segment

    def recording(*args, **kwargs):
        threads.append({info["num_threads"] for info in threadpool_info()})
        return segment(*args, **kwargs)

    monkeypatch.setattr(sp.sd, "segment", recording)
    out = _run(tmp_path, _write_noisy_set(tmp_path, count=2), "primary", "kmeans")
    assert threads == [{1}] * 4
    assert json.loads((out / "run.json").read_text())["cpu_threads_per_image"] == 1
