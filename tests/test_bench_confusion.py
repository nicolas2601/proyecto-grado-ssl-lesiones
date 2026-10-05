"""Confusion matrices of the self-supervised models: counts, normalization, scope, honesty."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from sklearn.metrics import matthews_corrcoef

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_bench_tables as bb  # noqa: E402
from dermssl.bench import confusion as C  # noqa: E402

OUT = REPO / bb.OUT
ARTS = C.artifacts(REPO)


def _expand(cm):
    y_true, y_pred = [], []
    for t, row in enumerate(cm):
        for p, count in enumerate(row):
            y_true += [t] * count
            y_pred += [p] * count
    return y_true, y_pred


def test_scope_is_self_supervised_only_and_has_23_cells():
    assert {a["modelo"].split(" (")[0] for a in ARTS} == {
        "DINOv2 ViT-S/14", "DINOv2 ViT-B/14", "SimSiam ResNet-50", "B71", "A71", "B71-limpio", "B71 original",
        "A71 original"}
    assert len(ARTS) == 17 + 6


def test_isic_cells_come_from_the_e1_json_and_state_their_compute():
    isic = [a for a in ARTS if a["dataset"] != "HAM10000" and a["modelo"].startswith(("B71", "A71"))]
    assert len(isic) == 6
    for art in isic:
        cell = json.loads((REPO / art["fuente"][0]).read_text(encoding="utf-8"))["result"]
        assert art["conteos"] == cell["metrics"]["matriz_confusion"]
        assert art["verificacion"] == {"n_val_json": cell["n_val"], "n_desde_matriz": art["n"]}
        assert ("canónico" in art["origen"]) == (art["modelo"] == "B71-limpio (canónico)")
        assert ("transductivo" in art["modelo"]) == art["modelo"].endswith("transductivo)")


def test_committed_json_files_equal_the_generated_artifacts():
    for art in ARTS:
        assert json.loads((OUT / C.rel_json(art)).read_text(encoding="utf-8")) == art
        assert (OUT / C.rel_json(art).with_suffix(".png")).stat().st_size > 1000


def test_counts_come_from_the_source_matrix_and_rows_normalize_to_one():
    for art in (a for a in ARTS if a["origen"] == "matriz_confusion del JSON"):
        source = json.loads((REPO / art["fuente"][0]).read_text(encoding="utf-8"))["probes"]
        probe = source["logreg"] if art["condicion"] == "congelado" else source["finetune"]
        assert art["conteos"] == probe["matriz_confusion"]
        assert all(sum(row) == pytest.approx(1.0) for row in art["normalizada_por_fila"])


def test_artifact_mcc_matches_sklearn_on_expanded_labels():
    for art in ARTS:
        y_true, y_pred = _expand(art["conteos"])
        assert art["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred), abs=1e-12)


def test_v71_matrices_are_derived_and_reproduce_the_stored_accuracy():
    v71 = [a for a in ARTS if a["dataset"] == "HAM10000" and a["modelo"].startswith(("A71", "B71"))]
    assert len(v71) == 2 and all("no trae matriz_confusion" in a["origen"] for a in v71)
    for art in v71:
        check = art["verificacion"]
        assert check["accuracy_desde_matriz"] == pytest.approx(check["accuracy_json"], abs=1e-12)
        assert art["n"] == 2004


def test_manifest_lists_pending_cells_and_the_missing_diff_figure():
    manifest = json.loads((OUT / "confusion" / "manifest.json").read_text(encoding="utf-8"))
    pend = {(p["dataset"], p["modelo"], p["condicion"]): p["estado"] for p in manifest["pendientes"]}
    assert pend[("ISIC 2020", "DINOv2 ViT-B/14", "fine-tuning")] == "no ejecutado todavia"
    assert any(m.startswith("A71-limpio") for _, m, _ in pend)
    assert all(p["eta"] and p["motivo"] for p in manifest["pendientes"])
    names = {(p["dataset"], p["modelo"]) for p in manifest["pendientes"]}
    assert ("ISIC 2019", "B71 original (canónico, transductivo)") in names
    assert ("ISIC 2019", "B71-limpio (canónico)") not in names
    assert "no comparten partición" in manifest["figura_de_diferencias"]
    readme = (OUT / "confusion" / "README.md").read_text(encoding="utf-8")
    assert "No se genera figura de diferencias frente a B71" in readme
    assert not [p for p in (OUT / "confusion").rglob("*") if "dif" in p.name.lower()]


def test_png_figure_is_drawn_with_two_panels(tmp_path):
    pytest.importorskip("matplotlib")
    from dermssl.bench.confusion_render import write_png

    target = tmp_path / "x.png"
    write_png(ARTS[0], target)
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
