"""The v7.1 weight bank reads the original v7 layout exactly as before and, on request, the clean one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import v71_io  # noqa: E402

ROOT = Path("/w")
ORIGINAL = {("A", "s42"): "final_A", ("A", "s43"): "final_A_s43", ("B", "s42"): "final_B", ("B", "s43"): "final_B_s43"}
CLEAN = {("A", "s42"): "final_A_s42", ("A", "s43"): "final_A_s43", ("B", "s42"): "final_B_s42"}


def test_the_original_layout_resolves_exactly_as_before():
    assert v71_io.FINAL_DIRS == ORIGINAL
    bank = v71_io.WeightBank(ROOT)
    assert bank.clean is False
    for (arm, seed), folder in ORIGINAL.items():
        assert bank.final_path(arm, seed) == ROOT / folder / v71_io.WEIGHTS_FILE


@pytest.mark.parametrize(("arm", "seed"), sorted(CLEAN))
def test_the_clean_layout_names_every_final_with_its_seed(arm, seed):
    bank = v71_io.WeightBank(ROOT, clean=True)
    assert bank.final_path(arm, seed) == ROOT / CLEAN[(arm, seed)] / v71_io.WEIGHTS_FILE


def test_run_manifests_default_is_the_four_originals_and_can_be_narrowed(tmp_path):
    for folder in ORIGINAL.values():
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "run.json").write_text('{"seed": 1}', encoding="utf-8")
    assert set(v71_io.WeightBank(tmp_path).run_manifests()) == {"A_s42", "A_s43", "B_s42", "B_s43"}
    assert set(v71_io.WeightBank(tmp_path).run_manifests([("B", "s42")])) == {"B_s42"}
