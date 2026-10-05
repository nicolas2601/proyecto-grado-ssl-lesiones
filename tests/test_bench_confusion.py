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


def test_scope_is_self_supervised_only_and_has_25_cells():
    assert {a["modelo"].split(" (")[0] for a in ARTS} == {
        "DINOv2 ViT-S/14", "DINOv2 ViT-B/14", "SimSiam ResNet-50", "B71", "A71", "B71-limpio", "B71 original",
        "A71 original", "A71-limpio"}
    assert len(ARTS) >= 17 + 6 + 2  # grows as cells land; the model set above is the contract


def test_isic_cells_come_from_the_e1_json_and_state_their_compute():
    isic = [a for a in ARTS if a["dataset"] != "HAM10000" and a["modelo"].startswith(("B71", "A71"))]
    assert len(isic) == 12  # per dataset: 4 canonical + 2 batch-16 cells
    for art in isic:
        cell = json.loads((REPO / art["fuente"][0]).read_text(encoding="utf-8"))["result"]
        assert art["conteos"] == cell["metrics"]["matriz_confusion"]
        assert art["verificacion"] == {"n_val_json": cell["n_val"], "n_desde_matriz": art["n"]}
        assert ("canónico" in art["origen"]) == ("canónico" in art["modelo"])
        assert ("batch 16" in art["origen"]) == ("batch 16" in art["modelo"])
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
    done = {(a["dataset"], a["modelo"], a["condicion"]) for a in ARTS}
    assert not set(pend) & done, "a cell with a matrix must not also be pending"
    for done in ("DINOv2 ViT-B/14", "DINOv2 ViT-S/14"):
        assert ("ISIC 2020", done, "fine-tuning") not in pend
    assert all(p["eta"] and p["motivo"] for p in manifest["pendientes"])
    names = {(p["dataset"], p["modelo"]) for p in manifest["pendientes"]}
    assert names == {("HAM10000", "B71-limpio (canónico)"), ("HAM10000", "A71-limpio (canónico)")}
    assert all("no hay transductividad que corregir" in p["motivo"] for p in manifest["pendientes"])
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


SUPERVISED = ("alexnet", "vgg19", "resnet50", "efficientnet_b0", "vit_b16")
ALLOWED = {"dinov2_vits14", "dinov2_vitb14", "simsiam_resnet50", "b71", "a71", "b71limpio", "b71_original",
           "a71_original", "a71limpio", "b71_original_batch16", "a71_original_batch16"}


def test_allowed_models_are_exactly_the_self_supervised_ones_and_supervised_are_the_rest():
    from dermssl.bench.rows import MODELS

    assert set(C.SSL_MODELS) == {"dinov2_vits14", "dinov2_vitb14", "simsiam_r50"}
    assert set(MODELS) - set(C.SSL_MODELS) == set(SUPERVISED)


def test_no_committed_matrix_belongs_to_a_supervised_model():
    stems = {p.name.split("__")[0] for p in (OUT / "confusion").rglob("*.json") if p.name != "manifest.json"}
    assert stems <= ALLOWED, stems - ALLOWED
    assert not {p.name for p in (OUT / "confusion").rglob("*") if p.name.startswith(SUPERVISED)}


def test_supervised_finetune_results_never_get_a_matrix_nor_a_pending_cell(tmp_path):
    import shutil

    for sub in ("results/baselines_isic2020", "results/baselines", "results/baselines_isic2019",
                "results/finetune/ham10000", "results/finetune/isic2019", "resultados/banco_completo/ft_isic2020/json", "resultados/clasificacion/v71",
                "resultados/banco_completo/isic_eval"):
        shutil.copytree(REPO / sub, tmp_path / sub, dirs_exist_ok=True)
    src = REPO / "resultados/banco_completo/ft_isic2020/json/results_dinov2_vitb14_ft_isic2020.json"
    for model in SUPERVISED:  # the supervised fine-tuning results arrive as these files
        shutil.copy(src, tmp_path / f"resultados/banco_completo/ft_isic2020/json/results_{model}_ft_isic2020.json")
    models = {a["modelo"] for a in C.artifacts(tmp_path)}
    assert not models & {"AlexNet", "VGG19", "ResNet-50", "EfficientNet-B0", "ViT-B/16"}
    assert not {p["modelo"] for p in C.pending(tmp_path)} & {"AlexNet", "VGG19", "ResNet-50", "EfficientNet-B0", "ViT-B/16"}
