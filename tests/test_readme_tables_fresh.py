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

sys.path.insert(0, str(REPO / "src"))

import build_results_tables as brt  # noqa: E402
from dermssl.finetune import protocol  # noqa: E402

# Banks whose tables are published. baselines_isic2020 left this map on
# 2026-09-04: the project moved to ISIC 2019 and its tables were retired from
# the README, so freshness no longer applies to it (its JSONs stay in results/
# as provenance). baselines_isic2019 joins automatically as soon as its JSONs
# land - see _published_banks below.
BANKS = {"baselines": REPO / "notebooks" / "baselines" / "README.md",
         "baselines_isic2019": REPO / "README.md"}


def _read_utf8(path: Path) -> str:
    """Read repository Markdown using its committed UTF-8 encoding."""
    return path.read_text(encoding="utf-8")


def _published_banks() -> list[str]:
    """Only COMPLETE banks are worth checking for freshness.

    A bank that is still running has some of its JSONs on disk, and demanding
    that a half-finished ranking be pasted into the README would be asking for
    a table that is wrong by construction: the ranking changes with every model
    that lands. ISIC 2019 joins this list on its eighth JSON, not before."""
    expected = len(protocol.MODELS)
    return [b for b in sorted(BANKS)
            if len(list((REPO / "results" / b).glob("results_*.json"))) == expected]


def _rows(bank: str) -> list[brt.Row]:
    return brt.rank(brt.load_bank(REPO / "results" / bank))


def _table_rows(markdown: str) -> set[str]:
    """Every data row of every markdown table, whitespace-normalised."""
    return {re.sub(r"\s+", " ", line).strip()
            for line in markdown.splitlines()
            if line.startswith("|") and not set(line) <= set("|- ")}


@pytest.mark.parametrize("bank", _published_banks())
def test_every_generated_row_appears_in_a_readme(bank):
    generated = _table_rows(brt.render(REPO / "results" / bank))
    published = _table_rows(_read_utf8(REPO / "README.md"))
    published |= _table_rows(_read_utf8(BANKS[bank]))
    missing = {r for r in generated if r not in published and "Modelo" not in r}
    assert not missing, (
        f"{bank}: these generated rows are not in any README - re-run "
        f"scripts/build_results_tables.py and paste: {sorted(missing)[:3]}")


@pytest.mark.parametrize("bank", _published_banks())
def test_readmes_carry_no_number_from_a_superseded_run(bank):
    """Guards the specific rot that happened on 2026-09-03: vit_b16 changed
    weights and vgg19's extraction point moved, but the tables stayed."""
    rows = _rows(bank)
    published = (_read_utf8(REPO / "README.md") + _read_utf8(BANKS[bank]))
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
    published = (
        _read_utf8(REPO / "README.md")
        + _read_utf8(REPO / "notebooks" / "baselines" / "README.md")
    )
    stale = "| ViT-B/16 | Supervisado | 768 | 0.761 |"
    assert stale not in published, "the pre-augreg ViT-B/16 row is still published"


def test_the_checkpoint_table_matches_the_protocol():
    """The hand-written version of this table claimed torchvision for four
    models that had already migrated to Hugging Face - including vit_b16,
    whose weight change is what flipped the comparison."""
    generated = _table_rows(brt.table_weights())
    published = _table_rows(_read_utf8(REPO / "notebooks" / "baselines" / "README.md"))
    missing = {r for r in generated if r not in published and "Modelo" not in r}
    assert not missing, (
        "the checkpoint table is stale - regenerate with "
        f"'build_results_tables.py --weights': {sorted(missing)[:2]}")


def test_no_readme_tells_the_reader_to_pick_a_p100():
    """Kaggle's PyTorch has no sm_60 kernels, so a P100 dies on the first
    forward. Any doc offering it as an option is a trap for the next person."""
    for path in (REPO / "notebooks" / "baselines" / "README.md",
                 REPO / "notebooks" / "finetune" / "README.md",
                 REPO / "cluster" / "finetune" / "README.md"):
        for line in _read_utf8(path).splitlines():
            if "P100" not in line:
                continue
            assert any(w in line for w in ("NO elegir", "no kernel image", "sm_60")), (
                f"{path.name}: line offers a P100 without warning: {line.strip()}")
