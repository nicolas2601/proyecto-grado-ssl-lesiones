"""Fine-tuning on ISIC 2020 (docs/33 G1.3): split identity, guards and a CPU smoke."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image



# ---------------------------------------------------------------- synthetic ISIC 2020
@pytest.fixture
def isic2020_root(tmp_path):
    rows = [{"isic_id": f"ISIC_{k:07d}", "patient_id": f"IP_{k // 3:04d}",
             "target": int(k % 7 == 0)} for k in range(900)]
    folder = tmp_path / "isic2020"
    (folder / "train-image").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "train-metadata.csv", index=False)
    for row in rows:
        Image.new("RGB", (16, 16), (row["target"] * 200, 40, 90)).save(
            folder / "train-image" / f"{row['isic_id']}.jpg")
    return tmp_path


def _cfg(root, tmp_path, **extra):
    from dermssl.finetune.config import FinetuneConfig
    return FinetuneConfig(model="alexnet", dataset="isic2020", data_root=str(root),
                          output_dir=str(tmp_path / "out"), num_workers=0,
                          precision="fp32", **extra)


@pytest.fixture
def no_committed_check(monkeypatch):
    from dermssl.finetune import split_guard
    monkeypatch.setattr(split_guard, "expectation_for", lambda dataset: None)


def test_ft_split_equals_the_isic_eval_split(isic2020_root, tmp_path, monkeypatch,
                                             no_committed_check):
    pytest.importorskip("torch")
    from dermssl.finetune import torch_run
    from dermssl.isic_eval import splits as sp

    monkeypatch.setattr(sp, "verify_split", lambda ds, expected: {})
    folder = isic2020_root / "isic2020"
    ref = sp.load_isic2020(folder / "train-metadata.csv", folder / "train-image")
    data = torch_run.load_split(_cfg(isic2020_root, tmp_path))
    assert data.tr_ids == ref.train_ids
    assert data.va_ids == ref.val_ids
    assert data.classes == ["benign", "malignant"]


# ---------------------------------------------------------------- guards
def test_wrong_sklearn_version_is_a_typed_error(isic2020_root, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import sklearn
    from dermssl.finetune import split_guard, torch_run

    monkeypatch.setattr(sklearn, "__version__", "1.8.0")
    with pytest.raises(split_guard.SklearnVersionError, match="1.7.2"):
        torch_run.load_split(_cfg(isic2020_root, tmp_path))


def test_a_split_that_is_not_the_committed_one_is_refused(isic2020_root, tmp_path,
                                                          monkeypatch):
    pytest.importorskip("torch")
    import sklearn
    from dermssl.finetune import torch_run
    from dermssl.isic_eval.splits import SplitMismatchError

    monkeypatch.setattr(sklearn, "__version__", "1.7.2")
    with pytest.raises(SplitMismatchError):
        torch_run.load_split(_cfg(isic2020_root, tmp_path))


def test_only_isic2020_is_checked_against_a_committed_split():
    from dermssl.finetune import split_guard
    from dermssl.isic_eval import splits as sp

    assert split_guard.expectation_for("isic2020") is sp.ISIC2020_EXPECTED
    assert split_guard.expectation_for("ham10000") is None
    assert split_guard.expectation_for("isic2019") is None


# ---------------------------------------------------------------- CPU smoke
def test_cpu_smoke_writes_a_binary_results_json(isic2020_root, tmp_path, monkeypatch,
                                                no_committed_check):
    torch = pytest.importorskip("torch")
    from dermssl.finetune import torch_run
    from dermssl.finetune.tracking import CompositeTracker

    tiny = lambda name: (torch.nn.Sequential(  # noqa: E731
        torch.nn.Conv2d(3, 256, 3, stride=8), torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1)), torch_run.transforms.InterpolationMode.BILINEAR)
    monkeypatch.setattr(torch_run, "load_backbone", tiny)
    monkeypatch.setattr(torch_run, "bootstrap_ci",
                        lambda y, p, seed, n_resamples=20: {"balanced_accuracy": [0, 1],
                                                            "macro_f1": [0, 1]})
    path = torch_run.run(_cfg(isic2020_root, tmp_path, smoke=True), CompositeTracker([]))

    assert path.name == "results_alexnet_ft_isic2020.json"
    data = json.loads(path.read_text())
    assert data["dataset"] == "isic2020" and data["smoke"] is True
    assert "patient_id" in data["split"]
    probes = data["probes"]["finetune"]
    assert set(probes["recall_por_clase"]) == {"benign", "malignant"}
    assert np.array(probes["matriz_confusion"]).shape == (2, 2)
    assert {"balanced_accuracy", "macro_f1", "auc_ovr_macro", "ece",
            "ic95_bootstrap"} <= set(probes)
    assert (path.parent / "ft_alexnet_isic2020.pth").exists()


def test_results_json_carries_the_per_epoch_curve(isic2020_root, tmp_path, monkeypatch,
                                                  no_committed_check, capsys):
    import re

    torch = pytest.importorskip("torch")
    from dermssl.finetune import protocol, torch_run
    from dermssl.finetune.tracking import CompositeTracker

    monkeypatch.setattr(protocol, "SMOKE_EPOCHS", 3)
    tiny = lambda name: (torch.nn.Sequential(  # noqa: E731
        torch.nn.Conv2d(3, 256, 3, stride=8), torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1)), torch_run.transforms.InterpolationMode.BILINEAR)
    monkeypatch.setattr(torch_run, "load_backbone", tiny)
    monkeypatch.setattr(torch_run, "bootstrap_ci",
                        lambda y, p, seed, n_resamples=20: {"balanced_accuracy": [0, 1],
                                                            "macro_f1": [0, 1]})
    path = torch_run.run(_cfg(isic2020_root, tmp_path, smoke=True), CompositeTracker([]))
    data = json.loads(path.read_text())

    curve = data["curva_epocas"]
    assert data["epocas_corridas"] == len(curve) == 3
    assert [row["epoca"] for row in curve] == [1, 2, 3]
    printed = re.findall(r"epoca (\d+)/3 loss=([\d.]+) val_bal_acc=([\d.]+)",
                         capsys.readouterr().out)
    assert len(printed) == 3
    for row, (n, loss, ba) in zip(curve, printed):
        assert row["epoca"] == int(n)
        assert f"{row['train_loss']:.4f}" == loss
        assert f"{row['val_balanced_accuracy']:.4f}" == ba
    assert {"mejor_epoca", "probes", "protocolo", "split"} <= set(data)
