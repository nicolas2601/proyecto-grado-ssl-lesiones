"""scripts/build_pad_ufes_table.py: the docs/30 2-3 first-pass table, generated from its JSON.

Rendered from a document that ``pad_ufes_eval.build_document`` builds on
synthetic probabilities (no PAD number is read here); the committed README is
checked against the committed JSON once the pod run exists.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_readings as pr
from scripts import build_pad_ufes_table as bt
from scripts import pad_ufes_eval as ev

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "resultados" / "clasificacion" / "pad_ufes" / "pad_ufes_eval.json"
README = RESULT.with_name("README.md")
N_RESAMPLES = 20


@pytest.fixture(scope="module")
def document():
    """Bin I: two patients with one class each, so some of its resamples miss a class."""
    mp = pytest.MonkeyPatch()
    mp.setattr(pr, "BOOTSTRAP", {"n_resamples": N_RESAMPLES, "rng_seed": 20260923})
    n = 72
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    labels = np.array(["bcc"] * 6 + ["nv"] * 6 + [cycle[i % 5] for i in range(n - 12)])
    groups = np.array(["PA"] * 6 + ["PB"] * 6 + [f"P{i // 3}" for i in range(n - 12)])
    bins = np.array(["I"] * 12 + [("II", "III-VI", pu.FST_MISSING)[i % 3] for i in range(n - 12)])
    rng = np.random.default_rng(0)
    probas = {r: (lambda raw: raw / raw.sum(axis=1, keepdims=True))(rng.random((n, 7))) for r in pr.roles()}
    fst = [1] * 12 + [(2, 3, None)[i % 3] for i in range(n - 12)]
    pad = {"ham_labels": labels, "groups": groups, "fst_bins": bins, "fst": fst}
    provenance = {"code_revision": "f" * 40, "registration_commit": "r" * 40,
                  "specs": {role: f"dinov2_vitb14:/w/{role}|x.pt" for role in pr.roles()}}
    doc = json.loads(json.dumps(ev.build_document(probas, pad, provenance)))
    mp.undo()
    return doc


def _row(entry: dict, arm: str, control: str, metric: str) -> dict:
    return next(r for r in entry["paired"]
                if r["arm"] == arm and r["control"] == control and r["metric"] == metric)


def test_the_header_carries_provenance_bootstrap_and_the_fst_caveats(document):
    text = bt.render(document)
    assert "docs/30" in text and "scripts/build_pad_ufes_table.py" in text
    assert f"semilla {pr.BOOTSTRAP['rng_seed']}" in text and f"{N_RESAMPLES} remuestras" in text
    assert "agrupado por paciente" in text
    assert document["fst_strata"]["note"] in text
    assert "PAD-UFES-20 no permite concluir sobre FST V-VI" in text
    full = document["readings"]["exploratory"]
    assert f"{full['n_images']} imágenes, {full['n_patients']} pacientes" in text


def test_the_thread_pin_comes_from_the_sidecar(document):
    env = {"env": {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}}
    assert "OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4, MKL_NUM_THREADS=4" in bt.render(document, env)
    assert "hilos no registrados" in bt.render(document, None)


def test_the_verdicts_are_copied_not_recomputed(document):
    doc = copy.deepcopy(document)
    for verdict, value in zip(bt.VERDICTS, ("cumplida", "no cumplida", "no cumplida", "cumplida")):
        doc["readings"]["primary"]["verdicts"][verdict] = value
    text = bt.render(doc)
    for verdict, value in zip(bt.VERDICTS, ("cumplida", "no cumplida", "no cumplida", "cumplida")):
        assert re.search(rf"\*\*{verdict}\*\*: {value} \(", text)


def test_the_verdict_rows_show_the_per_seed_deltas_used(document):
    text = bt.render(document)
    primary = document["readings"]["primary"]
    for seed in pr.SEEDS:
        row = _row(primary, pr.role("proposed", seed), pr.PUBLIC, "ece")
        delta = f"{row['point_delta']:.4f}".replace(".", ",")
        assert f"| PAD-P3 | s{seed} | `{row['arm']}` − `{row['control']}` | ECE | {delta} |" in text


def test_every_reading_stratum_and_model_is_in_the_table(document):
    text = bt.render(document)
    for name in pu.READINGS:
        assert bt.READING_LABELS[name] in text
    for name in document["fst_strata"]["bins"]:
        assert f"### {name}" in text
    for role in pr.roles():
        assert f"`{role}`" in text


def test_a_pipe_inside_a_cell_is_escaped(document):
    text = bt.render(document)
    assert "/w/public\\|x.pt" in text and "/w/public|x.pt" not in text


def test_contrast_labels_copy_the_direction_and_never_say_gana(document):
    doc = copy.deepcopy(document)
    rows = doc["readings"]["primary"]["paired"]
    rows[0]["direction"], rows[1]["direction"] = "arm_better", "control_better"
    text = bt.render(doc)
    assert not re.search(r"\bgana", text, re.IGNORECASE)
    assert f"mejora de `{rows[0]['arm']}` (IC excluye 0)" in text
    assert f"a favor de `{rows[1]['control']}` (IC excluye 0)" in text
    assert "indistinguible" in text


def test_mejora_is_never_written_outside_the_readings_with_verdict(document):
    doc = copy.deepcopy(document)
    for row in doc["readings"]["exploratory"]["paired"] + doc["fst_strata"]["bins"]["II"]["paired"]:
        row["direction"] = "arm_better"
    text = bt.render(doc)
    tail = text.split(f"## {bt.READING_LABELS['exploratory']}")[1]
    assert "mejora de" not in tail and "a favor de" in tail


def test_the_strata_show_k_over_the_resamples(document):
    text = bt.render(document)
    stratum = document["fst_strata"]["bins"]["I"]
    k = next(r["n_nonfinite"] for r in stratum["paired"] if r["metric"] == "auc")
    assert k > 0 and f"no calculable ({k}/{N_RESAMPLES})" in text


def test_k_is_rendered_exactly_over_1000(document):
    doc = copy.deepcopy(document)
    doc["bootstrap"]["n_resamples"] = 1000
    stratum = doc["fst_strata"]["bins"]["II"]
    arm = stratum["arms"][pr.PUBLIC]
    arm["n_nonfinite"]["auc"], arm["ci95"]["auc"] = 7, pr.NOT_CALCULABLE
    row = _row(stratum, pr.role("plain", 42), pr.PUBLIC, "ece")
    row.update(n_nonfinite=6, ci95_low=None, ci95_high=None, excludes_zero=None, direction=pr.NOT_CALCULABLE)
    text = bt.render(doc)
    assert "no calculable (7/1000)" in text and "no calculable (6/1000)" in text


def test_check_fails_when_the_table_is_stale(document, tmp_path):
    source = tmp_path / "pad_ufes_eval.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "README.md"
    assert bt.main([str(source), "--out", str(out)]) == 0
    assert bt.main([str(source), "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8") + "x", encoding="utf-8")
    assert bt.main([str(source), "--out", str(out), "--check"]) == 1


@pytest.mark.skipif(not RESULT.exists(), reason="la primera pasada todavia no se commiteo desde el pod")
def test_the_committed_readme_matches_the_committed_json():
    assert bt.main([str(RESULT), "--out", str(README), "--check"]) == 0


def test_main_reads_the_thread_pin_from_the_sidecar_next_to_the_json(document, tmp_path):
    source = tmp_path / "pad_ufes_eval.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    env = {"env": {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}}
    (tmp_path / "pad_ufes_eval.environment.json").write_text(json.dumps({"environment": env}), encoding="utf-8")
    out = tmp_path / "README.md"
    assert bt.main([str(source), "--out", str(out)]) == 0
    assert "OMP_NUM_THREADS=4" in out.read_text(encoding="utf-8")

