"""The v7.1 table is generated from ``v71_final_eval.json`` and checked for freshness.

The committed JSON does not exist until the pod runs; until then the renderer
is pinned on a synthetic document, and the freshness check starts biting the
moment the JSON is committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_v71_table as bt  # noqa: E402

from dermssl import v71_plan as plan  # noqa: E402
from dermssl import v71_protocol as vp  # noqa: E402

V71_DIR = REPO / "resultados" / "clasificacion" / "v71"
METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "auc", "ece", "mel_recall", "mel_auc",
           "mel_sens_at_thr_spec80", "mel_spec_at_thr_spec80", "mel_sens_at_thr_spec90",
           "mel_spec_at_thr_spec90", "mel_sens_at_thr_sens95", "mel_spec_at_thr_sens95")


def _doc() -> dict:
    winners = {"A": vp.config_id("A_soup_a0.6", 448, "cls4", "la1.0", True),
               "B": vp.config_id("B_s42_a0.4", 224, "cls_attn", "balanced", False),
               "public": vp.config_id("public", 224, "cls_gem", "none", True)}
    entries = plan.val_entries({"winners": winners})
    results = {label: {"config": cid, **{m: 0.5 + k / 100 for m in METRICS},
                       "ci95": {m: [0.4, 0.6] for m in METRICS}}
               for k, (label, cid) in enumerate(entries.items())}
    paired = [{"arm": a, "control": c, "metric": m, "point_delta": 0.01, "ci95_low": -0.01,
               "ci95_high": 0.03, "direction": "indistinguishable"}
              for a, c in plan.VAL_CONTRASTS for m in ("macro_f1", "balanced_accuracy", "auc", "ece",
                                                       "mel_recall", "mel_auc")]
    return {"split": "val", "n_eval": 2004, "code_revision": "abc123", "entries": entries,
            "results": results, "paired": paired,
            "selection": {"winners": winners, "phase2": {"enabled": False,
                                                         "mel_sens_at_spec80_holdout": {"A": 0.72, "B": 0.66}}},
            "v7_reproduction_delta": {"A_v7": {"macro_f1": 0.0}},
            "note_operating_points": plan.OPERATING_POINT_PROTOCOL}


def test_render_names_every_entry_and_the_winners():
    text = bt.render(_doc())
    for label in plan.VAL_LABELS:
        assert f"| {bt.LABELS[label]} |" in text
    assert "A_soup_a0.6@448|cls4|la1.0|tta" in text
    assert "fase 2" in text.lower()
    assert plan.OPERATING_POINT_PROTOCOL in text  # the reader sees which probe each column used


def test_render_is_deterministic():
    assert bt.render(_doc()) == bt.render(_doc())


def test_check_detects_a_stale_table(tmp_path):
    import json

    doc_path, out = tmp_path / "v71_final_eval.json", tmp_path / "README.md"
    doc_path.write_text(json.dumps(_doc()), encoding="utf-8")
    assert bt.main([str(doc_path), "--out", str(out), "--check"]) == 1
    assert bt.main([str(doc_path), "--out", str(out)]) == 0
    assert bt.main([str(doc_path), "--out", str(out), "--check"]) == 0


@pytest.mark.skipif(not (V71_DIR / "v71_final_eval.json").exists(), reason="v7.1 not run on the pod yet")
def test_committed_v71_readme_matches_its_json():
    assert bt.main([str(V71_DIR / "v71_final_eval.json"), "--out", str(V71_DIR / "README.md"), "--check"]) == 0


def _row(arm, control, metric, delta=0.05, low=0.01, high=0.09):
    return {"arm": arm, "control": control, "metric": metric, "point_arm": 0.7, "point_control": 0.7 - delta,
            "point_delta": delta, "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0}


def _extra() -> dict:
    fair = [_row("A71", "public_A71", m) for m in ("mel_sens_at_thr_spec80", "mel_sens_at_thr_spec90",
                                                   "mel_spec_at_thr_sens95")]
    return {"note_verdict": "solo si excluye el 0",
            "fair_mel": {"note": "umbral propio", "contrasts": fair},
            "mel_to_nv": {"note": "mel a nv", "entries": {"A71": {"n_mel": 222, "count": 62, "rate": 62 / 222},
                                                          "A_v7_s43": {"n_mel": 222, "count": 90, "rate": 90 / 222}},
                          "contrasts": [_row("A71", "A_v7", "mel_to_nv_rate", -0.1, -0.16, -0.05)]},
            "s43": {"note": "la sopa contiene s43", "contrasts": [_row("A71", "A_v7_s43", "macro_f1", 0.03, -0.01, 0.07)]}}


def test_the_extra_tables_render_from_their_json():
    text = bt.render(_doc(), _extra())
    for heading in ("umbral propio", "Melanomas clasificados como nevus", "v7 s43"):
        assert heading in text
    assert "la sopa contiene s43" in text and "solo si excluye el 0" in text
    assert f"| {bt.LABELS['A_v7_s43']} | 90 | 0,4054 |" in text
    assert "| A* v7.1 − A* v7 s43 | Macro-F1 | 0,7000 | 0,6700 | 0,0300 | [-0,0100; 0,0700] | no |" in text
    assert "| sí |" in text


def test_without_extra_the_table_is_unchanged():
    assert "umbral propio" not in bt.render(_doc())


def test_main_reads_the_sibling_extra_json(tmp_path):
    import json

    doc_path, out = tmp_path / "v71_final_eval.json", tmp_path / "README.md"
    doc_path.write_text(json.dumps(_doc()), encoding="utf-8")
    (tmp_path / "extra_contrasts.json").write_text(json.dumps(_extra()), encoding="utf-8")
    assert bt.main([str(doc_path), "--out", str(out)]) == 0
    assert "la sopa contiene s43" in out.read_text(encoding="utf-8")


@pytest.mark.skipif(not (V71_DIR / "extra_contrasts.json").exists(), reason="extra contrasts not written")
def test_committed_readme_carries_the_extra_tables():
    assert "Melanomas clasificados como nevus" in (V71_DIR / "README.md").read_text(encoding="utf-8")
