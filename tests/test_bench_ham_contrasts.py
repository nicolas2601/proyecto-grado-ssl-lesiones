"""HAM10000 bench: proposed B71 canonical, F4 exploratory, E3b contrasts, and contrast-record consistency."""

from __future__ import annotations

import json

import pytest

from dermssl.bench import contrasts_load as L
from dermssl.bench import ham_rows as H
from dermssl.bench import render, table_extras
from dermssl.bench import rows as R

REPO = R.REPO
CANON = L.E3B_DIRS["canónico (batch 64)"]


def _rows() -> dict[str, R.Row]:
    return {r.name: r for r in R.dataset_rows(REPO, R.DATASETS["ham10000"])}


def test_b71_is_the_proposed_row_at_the_canonical_compute_and_cites_s0_s1_s2():
    row = _rows()["B71 (propuesto)"]
    assert row.block == R.FROZEN and row.segmentation == "no"
    assert "0,7106 / 0,7141 / 0,7089" in row.note and "S1" in row.note
    s1 = json.loads((REPO / H.COMPUTES[1][2]).read_text(encoding="utf-8"))["rows"]["B71"]["macro_f1"]
    assert row.values["macro_f1"].point == pytest.approx(s1, abs=1e-9)


def test_f4_is_a_separate_exploratory_row_never_ranked_with_the_proposed_one():
    row = _rows()["B71 + F4 (exploratorio)"]
    assert row.block == R.EXPLORATORY and "F4" in row.segmentation
    assert "no demostrable de forma robusta" in row.note and "2 de 3 cómputos" in row.note
    assert row.values["macro_f1"].point == pytest.approx(0.7441282873093702)
    assert not any(r.block == R.FROZEN and "F4" in r.name for r in _rows().values())


def test_ham_clean_cells_are_pending_with_reason_and_eta():
    rows = _rows()
    for name in ("B71-limpio (canónico)", "A71-limpio (canónico)"):
        assert rows[name].status == R.PENDING and "ETA" in rows[name].note and not rows[name].values


def test_b71_is_above_every_frozen_baseline_in_the_four_variants_and_f4_in_five_of_eight():
    above, raised = L.e3b_contrasts(REPO, CANON)
    for label in L.B71_VARIANTS.values():
        mine = [c for c in above if c.arm == label and c.metric == "macro_f1"]
        assert len(mine) == 8 and all(c.excludes_zero and c.direction == "arm_better" and c.delta > 0 for c in mine)
    hit = {c.arm for c in raised if c.metric == "macro_f1" and c.excludes_zero and c.delta > 0}
    assert hit == {"VGG19 + F4", "EfficientNet-B0 + F4", "ResNet-50 + F4", "SimSiam ResNet-50 + F4",
                   "DINOv2 ViT-S/14 + F4"}


def test_ham_section_keeps_the_frozen_only_caveat_and_the_legend():
    text = "\n".join(table_extras.ham_section(REPO))
    assert "CONGELADOS" in text and "fine-tuning sigue en la partición de Kaggle" in text
    assert "5 de 8" in text and "`*` = el IC pareado excluye 0" in text


def _all_contrasts():
    out = list(L.v71_contrasts(REPO))
    for rel in L.E3B_DIRS.values():
        out += [c for part in L.e3b_contrasts(REPO, rel) for c in part]
    for rel in L.SEED_FILES.values():
        out += L.f4_vs_b71(REPO, rel)
    for ds in ("isic2019", "isic2020"):
        out += [c for part in L.isic_all(REPO, ds).values() for c in part]
    return out


def test_every_contrast_is_internally_consistent():
    contrasts = _all_contrasts()
    assert len(contrasts) >= 48 + 2 * 120 + 3 * 6 + 2 * (6 + 18)  # ISIC cells keep landing
    for c in contrasts:
        assert c.delta == pytest.approx(c.point_arm - c.point_control, abs=1e-9)
        assert c.excludes_zero == (c.low > 0 or c.high < 0)
        assert (c.direction == "indistinguishable") == (not c.excludes_zero)


def test_the_robustness_json_matches_the_per_seed_runs():
    doc = json.loads((REPO / L.ROBUSTNESS).read_text(encoding="utf-8"))
    for seed, rel in L.SEED_FILES.items():
        mine = next(c for c in L.f4_vs_b71(REPO, rel) if c.metric == "macro_f1" and "todas" in c.arm)
        stored = doc["seeds"][str(seed)]["macro_f1"]
        assert mine.delta == pytest.approx(stored["delta"], abs=1e-12)
        assert (mine.low, mine.high) == pytest.approx(tuple(stored["ci95"]), abs=1e-12)


def test_markdown_renders_the_new_ham_blocks():
    text = render.render_markdown(REPO, R.DATASETS["ham10000"])
    assert "## Exploratorio (segmentación" in text and "## Contrastes pareados (partición v6/v7" in text
