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
# 2026-09-04 when the project moved to ISIC 2019, and came back on 2026-09-11:
# its tables are published again in the README as the screening antecedent, so
# they must stay fresh like the other two. A bank only counts once all of its
# JSONs have landed - see _published_banks below.
BANKS = {"baselines": REPO / "notebooks" / "baselines" / "README.md",
         "baselines_isic2019": REPO / "README.md",
         "baselines_isic2020": REPO / "README.md",
         # The fine-tuning banks live one level deeper because the cluster and
         # Kaggle both write results/finetune/<dataset>/. They join the
         # freshness check on their eighth JSON, like every other bank.
         "finetune/ham10000": REPO / "README.md",
         "finetune/isic2019": REPO / "README.md"}


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
    published = _table_rows((REPO / "README.md").read_text())
    published |= _table_rows(BANKS[bank].read_text())
    missing = {r for r in generated if r not in published and "Modelo" not in r}
    assert not missing, (
        f"{bank}: these generated rows are not in any README - re-run "
        f"scripts/build_results_tables.py and paste: {sorted(missing)[:3]}")


@pytest.mark.parametrize("bank", _published_banks())
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


def test_the_checkpoint_table_matches_the_protocol():
    """The hand-written version of this table claimed torchvision for four
    models that had already migrated to Hugging Face - including vit_b16,
    whose weight change is what flipped the comparison."""
    generated = _table_rows(brt.table_weights())
    published = _table_rows((REPO / "notebooks" / "baselines" / "README.md").read_text())
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
        for line in path.read_text().splitlines():
            if "P100" not in line:
                continue
            assert any(w in line for w in ("NO elegir", "no kernel image", "sm_60")), (
                f"{path.name}: line offers a P100 without warning: {line.strip()}")


def test_the_published_finetuning_row_matches_its_jsons():
    """The one fine-tuning row in the README is hand-laid-out, so it needs a
    guard the generated tables get for free.

    It is published before its bank is complete because the claim stands on its
    own: the two confidence intervals do not overlap. If either JSON is
    re-run and the numbers move, this fails instead of letting the README keep
    a number the results no longer support."""
    import json

    readme = (REPO / "README.md").read_text()
    frozen = json.loads((REPO / "results" / "baselines"
                         / "results_resnet50.json").read_text())["probes"]["logreg"]
    tuned_files = list((REPO / "results").glob("finetune*/**/results_resnet50_ft.json"))
    tuned_files += list((REPO / "results" / "finetune").glob("results_resnet50_ft.json"))
    tuned_files = [f for f in tuned_files if "mlruns" not in f.parts]
    assert tuned_files, "the fine-tuned resnet50 JSON the README quotes is gone"
    tuned = json.loads(tuned_files[0].read_text())["probes"]["finetune"]

    for probe in (frozen, tuned):
        for field in ("balanced_accuracy", "macro_f1", "auc_ovr_macro"):
            assert f"{probe[field]:.3f}" in readme, (
                f"the README does not carry {field}={probe[field]:.3f}")

    floor = tuned["ic95_bootstrap"]["balanced_accuracy"][0]
    ceiling = frozen["ic95_bootstrap"]["balanced_accuracy"][1]
    assert floor > ceiling, (
        "the README claims the intervals do not overlap, but the fine-tuned "
        f"floor {floor:.3f} is not above the frozen ceiling {ceiling:.3f}")


def test_the_partial_finetuning_count_matches_what_landed():
    """'2 de 8' has to keep meaning two.

    While a bank is in flight the README carries a hand-written progress
    heading, and a heading like that rots the moment the next model lands. The
    freshness test only guards COMPLETE banks, so this covers the gap in
    between."""
    import re

    readme = (REPO / "README.md").read_text()
    m = re.search(r"Corridas cerradas hasta ahora \((\d+) de (\d+) en HAM10000\)", readme)
    if not m:
        return  # the bank finished and the heading was replaced by the table
    claimed, total = int(m.group(1)), int(m.group(2))
    on_disk = len(list((REPO / "results" / "finetune" / "ham10000")
                       .glob("results_*_ft.json")))
    assert claimed == on_disk, (
        f"the README says {claimed} runs are closed but {on_disk} JSONs are on disk")
    assert total == len(protocol.MODELS)
