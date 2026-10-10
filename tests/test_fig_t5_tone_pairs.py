from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))
sys.path.insert(0, str(REPO / "src"))

import fig_t5_tone_pairs as figure  # noqa: E402
from dermssl.circle import FST_SHIFT_L_STEP  # noqa: E402
from qualitative_figure_inputs import sha256_file  # noqa: E402


def _write_fixture(root: Path, *, shift: int = 2, include_l_shift: bool = False) -> Path:
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True)
    image_id = "tone-image-1"
    original = np.full((16, 16, 3), 180, dtype=np.uint8)
    transformed = np.full((16, 16, 3), 150, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    transformed[mask > 0] = original[mask > 0]
    files = {
        "original.png": original,
        "transformed.png": transformed,
        "mask.png": mask,
    }
    routes = {}
    paths = {}
    for filename, pixels in files.items():
        path = cases_dir / f"{image_id}_{filename}"
        Image.fromarray(pixels).save(path)
        routes[filename] = path.relative_to(root).as_posix()
        paths[filename] = path
    entry = {
        "image_id": image_id,
        "tercil": "oscuro",
        "shift_buckets": shift,
        "mask_semantics": "true=lesion",
        "rutas": routes,
        "source_sha256": sha256_file(paths["original.png"]),
        "transformed_sha256": sha256_file(paths["transformed.png"]),
        "mask_sha256": sha256_file(paths["mask.png"]),
    }
    if include_l_shift:
        entry["L_shift"] = -FST_SHIFT_L_STEP * shift
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"cases": [entry]}), encoding="utf-8")
    return manifest


def test_t5_derives_signed_l_shift_from_source_step():
    assert figure.derive_l_shift(2) == -16.0
    assert figure.derive_l_shift(-2) == 16.0
    assert figure.derive_l_shift(0) == 0.0
    assert FST_SHIFT_L_STEP == 8.0


def test_t5_verifies_hashes_resolution_and_protected_lesion_pixels(tmp_path):
    manifest = _write_fixture(tmp_path / "T5", include_l_shift=True)
    pairs = figure.load_pairs(manifest)
    pair = pairs[0]

    assert pair["image_id"] == "tone-image-1"
    assert pair["shift_buckets"] == 2 and pair["L_shift"] == -16.0
    assert pair["mask_semantics"] == "true=lesion"
    assert pair["resolution"] == (16, 16)
    assert np.array_equal(pair["original"][pair["mask"] == 1], pair["transformed"][pair["mask"] == 1])


def test_t5_rejects_wrong_hashes_shift_metadata_and_mask_semantics(tmp_path):
    manifest = _write_fixture(tmp_path / "wrong-hash")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["cases"][0]["transformed_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        figure.load_pairs(manifest)

    manifest = _write_fixture(tmp_path / "wrong-shift", shift=2, include_l_shift=True)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["cases"][0]["L_shift"] = 16
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="L_shift mismatch"):
        figure.load_pairs(manifest)

    manifest = _write_fixture(tmp_path / "wrong-mask")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["cases"][0]["mask_semantics"] = "true=background"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="true=lesion"):
        figure.load_pairs(manifest)


def test_t5_rejects_degenerate_lesion_masks(tmp_path):
    manifest = _write_fixture(tmp_path / "degenerate-mask")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    mask_path = manifest.parent / data["cases"][0]["rutas"]["mask.png"]
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(mask_path)
    data["cases"][0]["mask_sha256"] = sha256_file(mask_path)
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="degenerate"):
        figure.load_pairs(manifest)


def test_t5_fails_closed_without_manifest_and_draws_only_in_memory(tmp_path):
    import matplotlib.pyplot as plt

    missing = tmp_path / "missing" / "manifest.json"
    output = tmp_path / "no-exports"
    with pytest.raises(FileNotFoundError, match="manifest"):
        figure.build(missing, outdir=output)
    assert not output.exists()

    manifest = _write_fixture(tmp_path / "T5-draw")
    fig = figure.draw(figure.load_pairs(manifest))
    assert len(fig.axes) == 3
    labels = [text.get_text() for text in fig.texts]
    labels.extend(text.get_text() for legend in fig.legends for text in legend.get_texts())
    assert any("L_shift = −16" in label for label in labels)
    assert any("Lesión protegida" in label for label in labels)
    plt.close(fig)
    assert not output.exists()
