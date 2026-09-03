"""The numbers printed in the READMEs must still be the ones in the JSONs.

Generating the tables does not stop them from rotting: someone re-runs a bank,
commits new JSONs, and forgets to paste. This test fails when that happens, so
the README can never quietly disagree with results/.
"""

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_results_tables as brt  # noqa: E402

BANKS = {"baselines": REPO / "notebooks" / "baselines" / "README.md",
         "baselines_isic2020": REPO / "README.md"}


def _rows(bank: str) -> list[brt.Row]:
    return brt.rank(brt.load_bank(REPO / "results" / bank))


def _table_rows(markdown: str) -> set[str]:
    """Every data row of every markdown table, whitespace-normalised."""
    return {re.sub(r"\s+", " ", line).strip()
            for line in markdown.splitlines()
            if line.startswith("|") and not set(line) <= set("|- ")}


@pytest.mark.parametrize("bank", sorted(BANKS))
def test_every_generated_row_appears_in_a_readme(bank):
    generated = _table_rows(brt.render(REPO / "results" / bank))
    published = _table_rows((REPO / "README.md").read_text())
    published |= _table_rows(BANKS[bank].read_text())
    missing = {r for r in generated if r not in published and "Modelo" not in r}
    assert not missing, (
        f"{bank}: these generated rows are not in any README - re-run "
        f"scripts/build_results_tables.py and paste: {sorted(missing)[:3]}")


@pytest.mark.parametrize("bank", sorted(BANKS))
def test_readmes_carry_no_number_from_a_superseded_run(bank):
    """Guards the specific rot that happened on 2026-09-03: vit_b16 changed
    weights and vgg19's extraction point moved, but the tables stayed."""
    rows = _rows(bank)
    published = ((REPO / "README.md").read_text()
                 + BANKS[bank].read_text())
    for row in rows:
        model = brt.DISPLAY[row.model]
        for line in published.splitlines():
            if not line.startswith(f"| {model} |") and f"| {model} |" not in line:
                continue
            if "bal_acc" in line or "IC95" in line:
                continue
        # the current balanced accuracy must be printed somewhere
        assert f"{row.metrics['balanced_accuracy']:.3f}" in published, (
            f"{bank}/{row.model}: bal_acc "
            f"{row.metrics['balanced_accuracy']:.3f} is absent from the READMEs")


def test_the_superseded_vit_b16_number_is_gone():
    published = ((REPO / "README.md").read_text()
                 + (REPO / "notebooks" / "baselines" / "README.md").read_text())
    stale = "| ViT-B/16 | Supervisado | 768 | 0.761 |"
    assert stale not in published, "the pre-augreg ViT-B/16 row is still published"
