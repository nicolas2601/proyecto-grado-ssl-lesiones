"""scripts/build_pad_ufes_addendum_table.py: the docs/30 7 table, generated from its JSON.

Rendered from a document built on synthetic probabilities; the committed table
is checked against the committed JSON once the pod run exists.
"""

from __future__ import annotations

from dermssl.display_nomenclature import display_text
import json
import re
from pathlib import Path

import numpy as np
import pytest

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_addendum as pa
from dermssl import pad_ufes_readings as pr
from scripts import build_pad_ufes_addendum_table as bt

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "resultados" / "clasificacion" / "pad_ufes" / "pad_ufes_addendum_a71_s44.json"
TABLE = RESULT.with_name("adenda_a71_s44.md")


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
    probas = {r: (lambda raw: raw / raw.sum(axis=1, keepdims=True))(rng.random((n, 7))) for r in pa.roles()}
    pad = {"ham_labels": labels, "groups": groups, "fst_bins": bins, "fst": [2] * n}
    provenance = {"code_revision": "f" * 40, "registration_commit": "r" * 40,
                  "protocols": {role: "p" for role in pa.roles()}}
    doc = json.loads(json.dumps(pa.build_document(probas, pad, provenance)))
    mp.undo()
    return doc


def test_every_reading_stratum_and_model_is_in_the_table(document):
    text = bt.render(document)
    for name in pu.READINGS:
        assert bt.READING_LABELS[name] in text
    for name in pu.FST_BINS:
        assert f"### {name}" in text
    for role in pa.roles():
        assert display_text(f"`{role}`", code_labels=True) in text


def test_the_strata_show_k_over_the_resamples(document):
    text = bt.render(document)
    stratum = document["fst_strata"]["bins"]["I"]
    k = next(r["n_nonfinite"] for r in stratum["paired"] if r["metric"] == "auc")
    assert k > 0 and f"no calculable ({k}/20)" in text


def test_the_strata_arm_cells_show_k_exactly_over_1000(document):
    doc = json.loads(json.dumps(document))
    doc["bootstrap"]["n_resamples"] = 1000
    stratum = doc["fst_strata"]["bins"]["II"]
    arm = stratum["arms"][pa.A71]
    arm["n_nonfinite"]["ece"], arm["ci95"]["ece"] = 7, pa.NOT_CALCULABLE
    row = stratum["paired"][0]
    row.update(n_nonfinite=6, ci95_low=None, ci95_high=None, excludes_zero=None, reading=pa.NOT_CALCULABLE)
    text = bt.render(doc)
    assert "no calculable (7/1000)" in text and "no calculable (6/1000)" in text


def test_the_table_is_descriptive_only(document):
    text = bt.render(document)
    assert not re.search(r"\bmejora|\bgana|PAD-P", text, re.IGNORECASE)  # the FST note says "sin veredicto"
    labels = set(re.findall(r"\| (el IC excluye el 0|no se distingue|no calculable \(\d+/20\)) \|", text))
    assert labels and labels <= {"el IC excluye el 0", "no se distingue"} | {
        f"no calculable ({k}/20)" for k in range(1, 21)}


def test_check_fails_when_the_table_is_stale(document, tmp_path):
    source = tmp_path / "doc.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "table.md"
    assert bt.main([str(source), "--out", str(out)]) == 0
    assert bt.main([str(source), "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8") + "x", encoding="utf-8")
    assert bt.main([str(source), "--out", str(out), "--check"]) == 1


@pytest.mark.skipif(not RESULT.exists(), reason="la adenda todavia no se corrio en el pod")
def test_the_committed_table_matches_the_committed_json():
    assert bt.main([str(RESULT), "--out", str(TABLE), "--check"]) == 0
