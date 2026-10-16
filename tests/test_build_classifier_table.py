"""scripts/build_classifier_table.py on synthetic sources (no committed data)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import build_classifier_table as bct  # noqa: E402
from dermssl import per_class_metrics as pcm  # noqa: E402

CLASSES = list(pcm.HAM10000_CLASSES)
METRICS = {"macro_f1": 0.5, "balanced_accuracy": 0.4, "auc": 0.9, "ece": 0.05}


def _proba(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((21, 7))


def _doc(probas: dict, y_true: np.ndarray) -> dict:
    results = {}
    for role in bct.ROLES:
        macro = pcm.class_report(y_true, probas[role], CLASSES)["macro_f1"]
        rec = {**METRICS, "macro_f1": macro}
        results[role] = {**rec, "ci95": {m: [v - 0.01, v + 0.01] for m, v in rec.items()}}
    return {"arms": {"plain": "A*", "proposed": "B*", "public": "DINOv2 público"},
            "n_eval": len(y_true), "probe": "logreg", "results": results}


def _sources() -> dict:
    y_true = np.arange(21) % 7
    public = _proba(0)
    probas = {seed: {"y_true": y_true, "plain": _proba(seed), "proposed": _proba(seed + 100),
                     "public": public} for seed in (42, 43)}
    bank = {"split": {"classes": CLASSES, "val": 21},
            "rows": [{"backbone": b, "probe": "logreg", "fraction": 1.0, "seed": 0, **METRICS,
                      "accuracy": 0.6}
                     for b in bct.BACKBONES]}
    c1 = {f"{b}__none": {"backbone": b, "condition": "none", "metrics": {**METRICS, "accuracy": 0.6},
                         "gate": {"n_images": 21}} for b in bct.BACKBONES}
    docs = {seed: _doc(p, y_true) for seed, p in probas.items()}
    return {"bank": bank, "c1": c1, "docs": docs, "probas": probas}


def test_render_lists_every_class_and_the_melanoma_section():
    text = bct.render(_sources())
    for name in CLASSES:
        assert name in text
    assert "Sensibilidad melanoma" in text
    assert "AUC melanoma" in text
    assert "ECE: menor es mejor" in text


def test_render_has_one_row_per_v6_backbone_and_c1_cell():
    text = bct.render(_sources())
    for backbone in bct.BACKBONES:
        assert text.count(f"| {backbone} |") == 2


def test_v7_rows_carry_the_ci_and_the_seed_mean_has_none():
    text = bct.render(_sources())
    assert "[" in [ln for ln in text.splitlines() if "| v7 s42 | experimento de control SSL sin consistencia de tono |" in ln][0]
    mean_line = [ln for ln in text.splitlines() if "| media s42/s43 | experimento de control SSL sin consistencia de tono |" in ln][0]
    assert "[" not in mean_line


def test_public_arm_that_differs_between_seeds_fails_loudly():
    src = _sources()
    src["docs"][43]["results"]["public"]["auc"] = 0.1
    with pytest.raises(ValueError, match="público"):
        bct.render(src)


def test_public_probabilities_that_differ_between_seeds_fail_loudly():
    src = _sources()
    src["probas"][43]["public"] = _proba(9)
    with pytest.raises(ValueError, match="público"):
        bct.render(src)


def test_probas_that_disagree_with_the_json_macro_f1_fail():
    src = copy.deepcopy(_sources())
    src["docs"][42]["results"]["plain"]["macro_f1"] += 1e-6
    with pytest.raises(ValueError, match="macro-F1"):
        bct.render(src)


def test_unexpected_class_order_fails():
    src = _sources()
    src["bank"]["split"]["classes"] = CLASSES[::-1]
    with pytest.raises(ValueError, match="orden de clases"):
        bct.render(src)


def test_melanoma_rows_carry_a_bootstrap_ci_and_the_header_states_the_method():
    text = bct.render(_sources())
    assert f"{pcm.BOOTSTRAP_RESAMPLES} remuestras" in text
    assert f"semilla {pcm.BOOTSTRAP_SEED}" in text
    mel_rows = [ln for ln in text.splitlines() if ln.startswith("| experimento de control SSL sin consistencia de tono s42 |")]
    assert mel_rows[0].count("[") == 2  # sensitivity and AUC, both with a CI


def test_summary_has_accuracy_and_the_always_nv_reference_row():
    src = _sources()
    text = bct.render(src)
    header = [ln for ln in text.splitlines() if ln.startswith("| Versión |")][0]
    assert "Accuracy" in header
    nv_share = float((src["probas"][42]["y_true"] == CLASSES.index("nv")).mean())
    nv_row = [ln for ln in text.splitlines() if "| siempre nv |" in ln]
    assert len(nv_row) == 1 and bct._num(nv_share) in nv_row[0]
    v7_row = [ln for ln in text.splitlines() if ln.startswith("| v7 s42 | experimento de control SSL sin consistencia de tono |")][0]
    assert v7_row.count("[") == 5  # four JSON metrics plus accuracy, each with a CI
    assert "macro-F1" in text and "engaña" in text


def test_v6_and_c1_accuracy_come_from_their_json_without_ci():
    src = _sources()
    for row in src["bank"]["rows"]:
        row["accuracy"] = 0.7777
    text = bct.render(src)
    v6_row = [ln for ln in text.splitlines() if ln.startswith("| v6 |")][0]
    assert "0,7777 —" in v6_row
