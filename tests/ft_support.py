"""Shared synthetic ISIC 2020 data and a tiny backbone for the fine-tuning CPU tests."""

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from dermssl.finetune import split_guard, torch_run  # noqa: E402
from dermssl.finetune.config import FinetuneConfig  # noqa: E402


def make_isic2020(root):
    rows = [{"isic_id": f"ISIC_{k:07d}", "patient_id": f"IP_{k // 3:04d}",
             "target": int(k % 7 == 0)} for k in range(900)]
    folder = root / "data" / "isic2020"
    (folder / "train-image").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "train-metadata.csv", index=False)
    for row in rows:
        Image.new("RGB", (16, 16), (row["target"] * 200, 40, 90)).save(
            folder / "train-image" / f"{row['isic_id']}.jpg")
    return root / "data"


def tiny_backbone(name, pretrained=True):
    return (torch.nn.Sequential(torch.nn.Conv2d(3, 256, 3, stride=8),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1)),
            torch_run.transforms.InterpolationMode.BILINEAR)


def patch_for_cpu(monkeypatch):
    monkeypatch.setattr(split_guard, "expectation_for", lambda dataset: None)
    monkeypatch.setattr(torch_run, "load_backbone", tiny_backbone)
    monkeypatch.setattr(torch_run, "bootstrap_ci",
                        lambda y, p, seed, n_resamples=20: {"balanced_accuracy": [0, 1],
                                                            "macro_f1": [0, 1]})


def smoke_cfg(root, tmp_path):
    return FinetuneConfig(model="alexnet", dataset="isic2020", data_root=str(root),
                          output_dir=str(tmp_path / "out"), num_workers=0,
                          precision="fp32", smoke=True)
