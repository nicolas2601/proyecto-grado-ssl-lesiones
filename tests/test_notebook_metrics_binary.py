"""The metric cell shared by every bank must work for BOTH task shapes:
7-class HAM10000 and binary ISIC 2020. Caught live on Kaggle (2026-09-03):
roc_auc_score(multi_class="ovr") with a 2-column proba raises on binary y."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_finetune_tests as ft  # noqa: E402


def _all_metrics(n_classes: int):
    ns = {"SEED": 42, "N": n_classes, "classes": [f"c{i}" for i in range(n_classes)],
          "np": np, "warnings": warnings, "__builtins__": __builtins__}
    exec(ft.metrics_lib_code().format(model="x", dataset="x", json_suffix=""), ns)
    return ns["all_metrics"]


@pytest.mark.parametrize("n_classes", [2, 7])
def test_all_metrics_runs_for_binary_and_multiclass(n_classes):
    rng = np.random.default_rng(0)
    y = rng.integers(0, n_classes, size=120)
    proba = rng.random((120, n_classes)); proba /= proba.sum(1, keepdims=True)
    m = _all_metrics(n_classes)(y, proba)
    assert 0.0 <= m["auc_ovr_macro"] <= 1.0
    assert len(m["matriz_confusion"]) == n_classes
    assert set(m["recall_por_clase"]) == {f"c{i}" for i in range(n_classes)}


def test_binary_auc_is_the_positive_class_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    proba = rng.random((200, 2)); proba /= proba.sum(1, keepdims=True)
    assert _all_metrics(2)(y, proba)["auc_ovr_macro"] == pytest.approx(
        roc_auc_score(y, proba[:, 1]))
