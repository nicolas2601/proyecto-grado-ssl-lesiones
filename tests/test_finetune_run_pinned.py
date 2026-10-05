"""run() results on the tiny CPU smoke are pinned, so refactoring its loops cannot drift."""

import json

import pytest

torch = pytest.importorskip("torch")

import ft_support  # noqa: E402

from dermssl.finetune import torch_run  # noqa: E402
from dermssl.finetune.tracking import CompositeTracker  # noqa: E402

GOLDEN = {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.3333333333333333,
          "auc_ovr_macro": 1.0, "ece": 0.3548644483089447}


def test_smoke_metrics_do_not_drift(tmp_path, monkeypatch):
    ft_support.patch_for_cpu(monkeypatch)
    root = ft_support.make_isic2020(tmp_path)
    path = torch_run.run(ft_support.smoke_cfg(root, tmp_path), CompositeTracker([]))
    doc = json.loads(path.read_text())
    metrics = doc["probes"]["finetune"]
    for key, value in GOLDEN.items():
        assert metrics[key] == pytest.approx(value, abs=1e-6), key
    assert metrics["matriz_confusion"] == [[20, 0], [20, 0]]
    assert doc["mejor_epoca"] == 1
