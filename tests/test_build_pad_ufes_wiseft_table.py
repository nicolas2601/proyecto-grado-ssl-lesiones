"""scripts/build_pad_ufes_wiseft_table.py: the docs/30 8 table, generated from its JSON.

Rendered from a document built on synthetic probabilities; the committed table
is checked against the committed JSON once the pod run exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from test_pad_ufes_wiseft import _grid

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_readings as pr
from dermssl import pad_ufes_wiseft as pw
from dermssl import v71_protocol as vp
from scripts import build_pad_ufes_wiseft_table as bt

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "resultados" / "clasificacion" / "pad_ufes" / "pad_ufes_wiseft.json"
TABLE = RESULT.with_name("wiseft_alpha.md")


@pytest.fixture(scope="module")
def document():
    """Bin I: two patients with one class each, so some of its resamples have a single class."""
    mp = pytest.MonkeyPatch()
    mp.setattr(pr, "BOOTSTRAP", {"n_resamples": 20, "rng_seed": 20260923})
    n = 72
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    labels = np.array(["bcc"] * 6 + ["nv"] * 6 + [cycle[i % 5] for i in range(n - 12)])
    groups = np.array(["PA"] * 6 + ["PB"] * 6 + [f"P{i // 3}" for i in range(n - 12)])
    bins = np.array(["I"] * 12 + [("II", "III-VI", pu.FST_MISSING)[i % 3] for i in range(n - 12)])
    rng = np.random.default_rng(0)
    probas = {r: (lambda raw: raw / raw.sum(axis=1, keepdims=True))(rng.random((n, 7))) for r in pw.roles()}
    pad = {"ham_labels": labels, "groups": groups, "fst_bins": bins, "fst": [2] * n}
    provenance = {"code_revision": "f" * 40, "registration_commit": "r" * 40, "protocol": "p"}
    doc = pw.build_document(probas, pad, provenance, _grid())
    doc["first_pass_agreement"] = pw.first_pass_agreement(
        probas, {old: probas[new] for new, old in pw.FIRST_PASS_ROLES.items()})
    mp.undo()
    return json.loads(json.dumps(doc))


def _unescaped_pipes(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def test_every_reading_stratum_and_role_is_in_the_table(document):
    text = bt.render(document)
    for name in pu.READINGS:
        assert bt.READING_LABELS[name] in text
    for name in pu.FST_BINS:
        assert f"### {name}" in text
    for role in pw.roles():
        assert f"`{role}`" in text


def test_both_verdicts_are_shown_with_their_state(document):
    text = bt.render(document)
    for key in ("PAD-W1", "PAD-W2"):
        assert f"**{key}: {document['verdicts'][key][key]}**" in text


def test_pipes_inside_cells_are_escaped_so_every_row_keeps_its_columns(document):
    text = bt.render(document)
    assert "public@224\\|cls\\|none\\|id" in text
    for block in text.split("\n\n"):
        rows = [line for line in block.splitlines() if line.startswith("|")]
        if rows:
            assert len({_unescaped_pipes(line) for line in rows}) == 1, rows[0]


def test_the_ham_side_says_s43_has_no_curve(document):
    text = bt.render(document)
    assert document["ham_holdout"]["curves"]["A_s43"]["note"] in text
    ham = document["ham_holdout"]["curves"]["A_soup"]["points"]["0.8"]["macro_f1"]
    assert bt.num(ham) in text


def test_the_strata_show_k_over_the_resamples(document):
    text = bt.render(document)
    stratum = document["fst_strata"]["bins"]["I"]
    k = next(r["n_nonfinite"] for r in stratum["paired"] if r["metric"] == "auc")
    assert k > 0 and f"no calculable ({k}/20)" in text


def test_the_curve_table_reads_the_primary_points(document):
    text = bt.render(document)
    arms = document["readings"]["primary"]["arms"]
    assert bt.num(arms[vp.weights_name("A", "s43", 0.6)]["macro_f1"]) in text


def test_check_fails_when_the_table_is_stale(document, tmp_path):
    source = tmp_path / "doc.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "table.md"
    assert bt.main([str(source), "--out", str(out)]) == 0
    assert bt.main([str(source), "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8") + "x", encoding="utf-8")
    assert bt.main([str(source), "--out", str(out), "--check"]) == 1


@pytest.mark.skipif(not RESULT.exists(), reason="la curva WiSE-FT todavia no se corrio en el pod")
def test_the_committed_table_matches_the_committed_json():
    assert bt.main([str(RESULT), "--out", str(TABLE), "--check"]) == 0
