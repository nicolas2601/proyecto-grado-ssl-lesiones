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
TABLES_README = REPO / "resultados" / "baselines" / "README.md"
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
         "baselines_isic2019": TABLES_README,
         "baselines_isic2020": TABLES_README,
         # The fine-tuning banks live one level deeper because the cluster and
         # Kaggle both write results/finetune/<dataset>/. They join the
         # freshness check on their eighth JSON, like every other bank.
         "finetune/ham10000": TABLES_README,
         "finetune/isic2019": TABLES_README}


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
    published = _table_rows(_read_utf8(TABLES_README))
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
    published = _read_utf8(BANKS[bank])
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
        _read_utf8(TABLES_README)
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


def _finetune_section() -> str:
    """Just the fine-tuning results section of the README.

    Located by intent, not by exact prose: the heading has been reworded twice
    and pinning the sentence made this guard fail on a rename instead of on a
    wrong number, which is the opposite of what it is for."""
    readme = TABLES_README.read_text()
    heads = [m for m in re.finditer(r"^## .*fine-tuning.*$", readme, re.M | re.I)
             if "sin fine-tuning" not in m.group(0).lower()]
    assert len(heads) == 1, (
        f"expected exactly one 'con fine-tuning' section, found "
        f"{[h.group(0) for h in heads]}")
    rest = readme[heads[0].end():]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _finetune_jsons(dataset: str):
    """Results of ONE dataset's fine-tuning bank.

    Scoping by dataset is the whole point. An earlier version globbed every
    fine-tuning JSON under results/, so the moment the first ISIC 2019 run
    landed it compared that alexnet (0.642) against the HAM10000 row (0.670)
    and reported a mismatch that did not exist. Worse than the false alarm is
    the false pass hiding behind it: had the two datasets happened to agree,
    the guard would have gone green while checking the wrong file."""
    root = REPO / "results" / "finetune" / dataset
    return [h for h in sorted(root.glob("results_*_ft*.json"))
            if "mlruns" not in h.parts]


# Where each fine-tuned dataset's FROZEN counterpart lives. The first version of
# the paired-table guard read every frozen number from results/baselines/, which
# is HAM10000: a correct ISIC 2019 table would have failed against the wrong
# bank, and a wrong one copied from HAM10000 would have passed.
FROZEN_BANK = {"ham10000": ("baselines", ""),
               "isic2019": ("baselines_isic2019", "_isic2019")}


def _dataset_sections() -> dict[str, str]:
    """Each published `### <dataset> con fine-tuning` block, by dataset.

    The heading is matched up to ' con fine-tuning', not with a no-space
    class: the first
    version captured only 'ISIC' out of 'ISIC 2019', so that section was never
    looked up and its rows went unchecked in silence. A block ends at the next
    ### heading of any kind, so a later table that happens to start with the
    same model name (the LR probe) cannot be mistaken for the paired row."""
    section = _finetune_section()
    found = {}
    heads = list(re.finditer(r"^### (.+?) con fine-tuning.*$", section, re.M))
    for head in heads:
        rest = section[head.end():]
        nxt = re.search(r"^### ", rest, re.M)
        body = rest if nxt is None else rest[:nxt.start()]
        key = head.group(1).lower().replace(" ", "")
        if key in FROZEN_BANK:
            found[key] = body
    return found


def _paired_row(section: str, display: str) -> list[str]:
    """The hand-laid-out row of the paired table, as a list of cells."""
    for line in section.splitlines():
        cells = [c.strip().strip("*") for c in line.split("|")]
        if len(cells) >= 7 and cells[1] == display:
            return cells
    raise AssertionError(f"no paired row published for {display}")


def test_every_published_finetuning_number_matches_its_json():
    """The paired table is laid out by hand, so it needs the guard the
    generated tables get for free - and its derived columns need checking too,
    because an arithmetic slip there is invisible against the JSONs."""
    import json

    sections = _dataset_sections()
    assert sections, "no '### <dataset> con fine-tuning' block found"
    checked = 0
    for dataset, section in sections.items():
        for path in _finetune_jsons(dataset):
            data = json.loads(path.read_text())
            assert data["dataset"].replace(" ", "") == dataset, (
                f"{path} sits under {dataset}/ but its JSON says "
                f"{data['dataset']}")
            model = data["modelo"]
            display = brt.DISPLAY.get(model, model)
            if f"| {display} |" not in section:
                continue  # not published yet; nothing to keep honest
            tuned = data["probes"]["finetune"]
            bank, suffix = FROZEN_BANK[dataset]
            frozen = json.loads((REPO / "results" / bank
                                 / f"results_{model}{suffix}.json").read_text())["probes"]
            frozen = frozen[max(frozen, key=lambda k: frozen[k]["balanced_accuracy"])]

            cells = _paired_row(section, display)
            assert cells[3] == f"{frozen['balanced_accuracy']:.3f}", (
                f"{dataset}/{model}: published frozen {cells[3]}, JSON says "
                f"{frozen['balanced_accuracy']:.3f}")
            assert cells[4] == f"{tuned['balanced_accuracy']:.3f}", (
                f"{dataset}/{model}: published fine-tuned {cells[4]}, JSON says "
                f"{tuned['balanced_accuracy']:.3f}")

            delta = tuned["balanced_accuracy"] - frozen["balanced_accuracy"]
            assert cells[5] == f"{delta:+.3f}", (
                f"{dataset}/{model}: published delta {cells[5]}, arithmetic "
                f"gives {delta:+.3f}")

            # Separation is checked in BOTH directions. The first version only
            # measured the fine-tuned floor against the frozen ceiling, which is
            # the right question for an improvement and the wrong one for a
            # drop: DINOv2 ViT-B/14 on HAM10000 sits entirely BELOW its frozen
            # interval (0.572 vs 0.576) and the README called that an overlap.
            up = (tuned["ic95_bootstrap"]["balanced_accuracy"][0]
                  - frozen["ic95_bootstrap"]["balanced_accuracy"][1])
            down = (frozen["ic95_bootstrap"]["balanced_accuracy"][0]
                    - tuned["ic95_bootstrap"]["balanced_accuracy"][1])
            if up > 0:
                expect, overlap = f"{up:+.3f}", False
            elif down > 0:
                expect, overlap = f"{-down:+.3f}", False
            else:
                expect, overlap = f"{up:+.3f}", True
            assert cells[6].startswith(expect), (
                f"{dataset}/{model}: published gap {cells[6]}, arithmetic "
                f"gives {expect}")
            assert ("solapan" in cells[6]) == overlap, (
                f"{dataset}/{model}: the row says {cells[6]!r} but up={up:+.4f} "
                f"down={down:+.4f}")
            if down > 0:
                assert "por debajo" in cells[6], (
                    f"{dataset}/{model}: separated downwards must say so")
            checked += 1
    assert checked, "no published fine-tuning row was verified"


def test_the_no_overlap_claim_is_true_where_the_readme_makes_it():
    """The README says resnet50's intervals separate. That must stay true."""
    import json

    frozen = json.loads((REPO / "results" / "baselines"
                         / "results_resnet50.json").read_text())["probes"]["logreg"]
    tuned = [p for p in _finetune_jsons("ham10000") if "resnet50" in p.name]
    assert tuned, "the fine-tuned resnet50 JSON the README quotes is gone"
    t = json.loads(tuned[0].read_text())["probes"]["finetune"]
    floor = t["ic95_bootstrap"]["balanced_accuracy"][0]
    ceiling = frozen["ic95_bootstrap"]["balanced_accuracy"][1]
    assert floor > ceiling, (
        f"the README claims separation but the fine-tuned floor {floor:.3f} "
        f"is not above the frozen ceiling {ceiling:.3f}")


def test_the_partial_finetuning_count_matches_what_landed():
    """'2 de 8' has to keep meaning two.

    While a bank is in flight the README carries a hand-written progress
    heading, and a heading like that rots the moment the next model lands. The
    freshness test only guards COMPLETE banks, so this covers the gap in
    between."""
    import re

    readme = TABLES_README.read_text()
    m = re.search(r"Corridas cerradas hasta ahora \((\d+) de (\d+) en HAM10000\)", readme)
    if not m:
        return  # the bank finished and the heading was replaced by the table
    claimed, total = int(m.group(1)), int(m.group(2))
    on_disk = len(list((REPO / "results" / "finetune" / "ham10000")
                       .glob("results_*_ft.json")))
    assert claimed == on_disk, (
        f"the README says {claimed} runs are closed but {on_disk} JSONs are on disk")
    assert total == len(protocol.MODELS)


def test_the_lr_probe_rows_are_published_and_fresh():
    """The probe has two models, so it never reaches _published_banks (which
    waits for eight). It still must not rot: its two rows decide how the DINOv2
    drop is read, and a stale 0.762 would be the worst number to leave wrong."""
    probe = REPO / "results" / "finetune_lrprobe"
    if len(list(probe.glob("results_*.json"))) < 2:
        pytest.skip("the LR probe has not landed")
    generated = _table_rows(brt.render(probe))
    published = _table_rows(TABLES_README.read_text())
    missing = {r for r in generated if r not in published and "Modelo" not in r}
    assert not missing, (
        "LR probe rows are not in the README - re-run "
        f"build_results_tables.py --results-dir results/finetune_lrprobe: {sorted(missing)[:3]}")
