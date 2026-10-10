"""Figure A: the loader must reject any pod histogram that breaks the agreed contract; no figure is built from simulated data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_a_cielab_ring as figure  # noqa: E402


def _fixture(root: Path, **overrides) -> None:
    edges_ab = list(figure.EXPECTED_AB_EDGES)
    l_counts = np.zeros(50, dtype=int)
    l_counts[30] = 10
    ab = np.zeros((len(edges_ab) - 1, len(edges_ab) - 1), dtype=int)
    ab[40, 40] = 10
    doc = {"dataset": "ham10000", "n_images": 4539, "n_pixels": 10, "L_edges": list(figure.EXPECTED_L_EDGES),
           "L_counts": l_counts.tolist(), "a_edges": edges_ab, "b_edges": edges_ab, "ab_counts": ab.tolist(),
           "code_revision": "x", "masks_sha256": "m", "ids_sha256": "i", "canonical_input": "450", "ring_definition": "r"}
    doc.update(overrides)
    target = root / figure.INPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    (target / "cielab_ring_hist_ham10000.json").write_text(json.dumps(doc), encoding="utf-8")


def test_valid_contract_loads(tmp_path):
    _fixture(tmp_path)
    assert figure.load(tmp_path, "ham10000", 4539)["doc"]["n_pixels"] == 10


@pytest.mark.parametrize("override,message", [
    ({"n_images": 4538}, "n_images"),
    ({"n_pixels": 11}, "totals"),
    ({"L_edges": list(range(0, 101, 5))}, "edges"),
    ({"dataset": "isic2019"}, "dataset"),
    ({"L_counts": [0.5] * 50}, "integers"),
])
def test_broken_contract_aborts(tmp_path, override, message):
    _fixture(tmp_path, **override)
    with pytest.raises(ValueError, match=message):
        figure.load(tmp_path, "ham10000", 4539)


def test_missing_metadata_aborts(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / figure.INPUT_DIR / "cielab_ring_hist_ham10000.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["masks_sha256"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        figure.load(tmp_path, "ham10000", 4539)
