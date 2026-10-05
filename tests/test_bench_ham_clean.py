"""HAM10000 clean-weights sensitivity in the bench: pending while the JSON is missing, generated once it exists.

The result JSON does not exist in the repository yet, so the measured state is proven on a SYNTHETIC fixture built
in a tmp copy of the results (never committed): produced by the real contrast and metric code, so the schema cannot
drift from what ``scripts/ham_clean_run.py`` writes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import v71_val  # noqa: E402

from dermssl import ham_clean as hc  # noqa: E402
from dermssl import v71_metrics as vm  # noqa: E402
from dermssl.bench import ablation_data, confusion, contrasts_doc, glossary, pending, render, rows as R  # noqa: E402

NAMES = ("B71-limpio (canónico)", "A71-limpio (canónico)")
THRESHOLDS = {"spec80": 0.2, "spec90": 0.3, "sens95": 0.05}
HAM = R.DATASETS["ham10000"]


def _by_name(repo):
    return {r.name: r for r in R.dataset_rows(repo, HAM)}


def synthetic_result(repo: Path) -> None:
    rng = np.random.RandomState(3)
    n, mel = 420, 4
    y = np.arange(n) % 7
    labels = (*hc.CLEAN, *hc.ORIGINAL, *hc.PUBLIC)
    probas = {k: rng.dirichlet(np.ones(7) * 0.4, n) * 0.5 + 0.5 * np.eye(7)[(y + (rng.rand(n) < 0.3)) % 7] for k in labels}
    results = {k: {"config": "x", "holdout_thresholds": THRESHOLDS,
                   **vm.val_entry_metrics(y, proba_full=probas[k], proba_threshold=probas[k], mel=mel,
                                          thresholds=THRESHOLDS, n_resamples=4)} for k in hc.CLEAN}
    groups = np.repeat(np.arange(n // 2), 2)
    paired = [row for a, c in hc.CONTRASTS for row in v71_val._contrast(
        a, c, y, probas, mel, 4, metrics=hc.CONTRAST_METRICS, groups=groups)]
    folder = repo / hc.DIR
    folder.mkdir(parents=True)
    np.savez_compressed(repo / hc.RESULT_NPZ, y_true=y, **probas)
    doc = {**hc.declarations(rehearsal=False), "split": "val", "n_eval": n, "results": results, "paired": paired,
           "bootstrap": {"n_resamples": 4, "rng_seed": 20260923, "paired": True, "grouped_by": "lesion_id"}}
    (repo / hc.RESULT_JSON).write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    shutil.copytree(REPO / "resultados", root / "resultados")
    shutil.copytree(REPO / "results", root / "results")
    synthetic_result(root)
    return root


# --- pending state: the real repository has no result JSON ---------------------------------------------------

def test_without_the_json_the_two_rows_are_pending_in_the_clean_block():
    rows = _by_name(REPO)
    for name in NAMES:
        r = rows[name]
        assert r.block == R.CLEAN_HAM and r.status == R.PENDING and not r.values and "sin ETA" in r.note
    assert not (REPO / hc.RESULT_JSON).exists()


def test_the_registered_rows_stay_in_the_frozen_block_labelled_as_original_weights():
    rows = _by_name(REPO)
    for name in ("B71 (propuesto)", "A71 (ablación sin término de tono)"):
        assert rows[name].block == R.FROZEN and rows[name].note.startswith("pesos originales (selección)")


def test_the_clean_block_comes_after_the_frozen_one_and_renders_as_its_own_section():
    assert R.BLOCKS.index(R.CLEAN_HAM) > R.BLOCKS.index(R.FROZEN)
    text = render.render_markdown(REPO, HAM)
    assert "## Pesos limpios (sensibilidad)" in text and text.index("## Congelados") < text.index("## Pesos limpios")


def test_pending_reason_is_derived_from_the_missing_file_everywhere():
    assert hc.RESULT_JSON in pending.HAM_CLEAN.reason
    table = contrasts_doc.pending_table(REPO)
    assert any(r[0] == "B71-limpio y A71-limpio contra el original en HAM10000" for r in table.rows)
    names = {(p["dataset"], p["modelo"]) for p in confusion.pending(REPO)}
    assert {("HAM10000", n) for n in NAMES} <= names


# --- measured state (synthetic fixture) -----------------------------------------------------------------------

def test_with_the_json_the_rows_are_measured_with_mcc_and_the_registered_row_is_untouched(measured):
    rows = _by_name(measured)
    for name in NAMES:
        r = rows[name]
        assert r.block == R.CLEAN_HAM and r.status == R.OK and set(r.values) == set(R.METRICS)
        assert r.sources == (hc.RESULT_JSON, hc.RESULT_NPZ) and r.n_val == 420 and not r.transductive
        assert "sensibilidad posterior" in r.note and "no es selección" in r.note
    registered = _by_name(REPO)["B71 (propuesto)"]
    assert rows["B71 (propuesto)"].values == registered.values and rows["B71 (propuesto)"].block == R.FROZEN


def test_the_clean_mcc_is_computed_from_the_stored_probabilities(measured):
    from dermssl.bench import metrics

    npz = np.load(measured / hc.RESULT_NPZ)
    cm = metrics.confusion_from_probabilities(npz["y_true"], npz["B71_clean"])
    assert _by_name(measured)["B71-limpio (canónico)"].values["mcc"].point == pytest.approx(metrics.mcc_from_confusion(cm))


def test_pending_disappears_only_with_the_json_and_comes_back_without_it(measured, tmp_path):
    assert not any(r[0].startswith("B71-limpio y A71-limpio contra el original en HAM10000")
                   for r in contrasts_doc.pending_table(measured).rows)
    assert not any(p["dataset"] == "HAM10000" and "limpio" in p["modelo"] for p in confusion.pending(measured))
    gone = tmp_path / "again"
    shutil.copytree(measured, gone)
    (gone / hc.RESULT_JSON).unlink()
    assert any(r[0].startswith("B71-limpio y A71-limpio contra el original en HAM10000")
               for r in contrasts_doc.pending_table(gone).rows)
    assert _by_name(gone)[NAMES[0]].status == R.PENDING


def test_contrasts_document_lists_the_five_pairs_with_mcc(measured):
    tabs, items = contrasts_doc.build(measured)
    mine = [t for t in tabs if t.label == "contrast-ham-clean"]
    assert len(mine) == 1 and len(mine[0].rows) == 5 * len(hc.CONTRAST_METRICS)
    assert "lesion_id" in mine[0].note and hc.RESULT_JSON in mine[0].note
    pairs = {(c.arm, c.control) for c in items if c.source == hc.RESULT_JSON}
    assert pairs == {("B71-limpio", "DINOv2 público (prot. B71)"), ("B71-limpio", "B71 original"),
                     ("A71-limpio", "DINOv2 público (prot. A71)"), ("A71-limpio", "A71 original"),
                     ("B71-limpio", "A71-limpio")}
    assert any(r[2] == "MCC" for r in mine[0].rows)
    assert not [t for t in contrasts_doc.build(REPO)[0] if t.label == "contrast-ham-clean"]


def test_ablation_has_the_clean_rows_and_its_own_contrast_table_with_mcc(measured):
    tabs = {t.label: t for t in ablation_data.tables(measured)}
    clean = tabs["ablation-clean"]
    assert any(r[0] == "HAM10000" and r[1] == NAMES[0] and r[3] != "pendiente" for r in clean.rows)
    mine = tabs["ablation-ham-clean"]
    assert {r[2] for r in mine.rows} >= {"MCC", "F1 macro"} and "sensibilidad posterior" in mine.note
    assert "ablation-ham-clean" not in {t.label for t in ablation_data.tables(REPO)}


def test_confusion_artifacts_exist_only_with_the_json(measured):
    ham = [a for a in confusion.artifacts(measured) if a["dataset"] == "HAM10000" and "limpio" in a["modelo"]]
    assert {a["modelo"] for a in ham} == set(NAMES)
    assert all(a["n"] == 420 and a["verificacion"]["accuracy_json"] > 0 for a in ham)
    assert not [a for a in confusion.artifacts(REPO) if a["dataset"] == "HAM10000" and "limpio" in a["modelo"]]


def test_reading_rules_and_glossary_explain_the_clean_block():
    text = "\n".join(glossary.GLOSSARY + glossary.reading_rules(REPO, R.DATASETS))
    assert "Pesos limpios (sensibilidad)" in text and "pesos originales (selección)" in text
    assert "no es selección" in text and "umbrales" in text
