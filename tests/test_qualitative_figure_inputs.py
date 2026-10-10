from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import qualitative_figure_inputs as inputs  # noqa: E402


def test_read_manifest_accepts_case_list_and_cases_object(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    cases = [{"image_id": "one"}]
    list_manifest = root / "list.json"
    list_manifest.write_text(json.dumps(cases), encoding="utf-8")
    object_manifest = root / "object.json"
    object_manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")

    assert inputs.read_manifest(list_manifest) == (root.resolve(), cases, list_manifest.resolve())
    assert inputs.read_manifest(object_manifest) == (root.resolve(), cases, object_manifest.resolve())


def test_manifest_loader_fails_when_manifest_missing_or_cases_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        inputs.read_manifest(tmp_path / "missing.json")
    empty = tmp_path / "empty.json"
    empty.write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        inputs.read_manifest(empty)


def test_resolve_input_path_requires_existing_relative_file_inside_root(tmp_path):
    root = tmp_path / "manifest-root"
    root.mkdir()
    target = root / "case.png"
    target.write_bytes(b"fixture")

    assert inputs.resolve_input_path(root, "case.png") == target.resolve()
    with pytest.raises(ValueError, match="relative"):
        inputs.resolve_input_path(root, str(target))
    with pytest.raises(ValueError, match="inside"):
        inputs.resolve_input_path(root, "../outside.png")
    with pytest.raises(FileNotFoundError, match="missing.png"):
        inputs.resolve_input_path(root, "missing.png")


def test_read_rgb_and_binary_mask_only_accept_binary_png_values(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    invalid_mask_path = tmp_path / "invalid.png"
    Image.fromarray(np.full((5, 7, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:5] = 255
    Image.fromarray(mask, mode="L").save(mask_path)
    invalid = mask.copy()
    invalid[0, 0] = 128
    Image.fromarray(invalid, mode="L").save(invalid_mask_path)

    rgb = inputs.read_rgb(image_path)
    binary = inputs.read_binary_mask(mask_path)
    assert rgb.shape == (5, 7, 3) and rgb.dtype == np.uint8
    assert binary.shape == (5, 7) and set(np.unique(binary)) == {0, 1}
    with pytest.raises(ValueError, match="binary"):
        inputs.read_binary_mask(invalid_mask_path)
