"""Generate one self-contained Kaggle FINE-TUNING notebook per baseline model.

Second stage of the baseline bank: the frozen-backbone tests measured the
representations as-is; these notebooks fine-tune the SAME 8 backbones
end-to-end on the medical data (linear head + full backbone, weighted CE,
AdamW, cosine schedule, AMP) and report the SAME metric suite, so both banks
are directly comparable cell-by-cell.

Output:
  notebooks/finetune/<model>/finetune_<model>.ipynb              (HAM10000)
  notebooks/finetune_isic2020/<model>/finetune_<model>_isic2020.ipynb

Everything shared with the frozen bank (model list, data/split cells, backbone
loaders, metric functions, report, MLflow) is imported from
build_baseline_tests.py - one source of truth for the protocol.

Run: python scripts/build_finetune_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path

from build_baseline_tests import (
    BACKBONE_CODE,
    DATASETS,
    MLFLOW_CODE,
    MODELS,
    PROBES_CODE,
    REPORT_CODE,
    TITLES,
    make_cell,
)

HEADER_FT_MD = """\
# Fine-tuning de baseline: {title}

Segunda etapa del banco (Objetivo 2): el mismo backbone del test congelado,
ahora **fine-tuning end-to-end** sobre **{dataset_title}** (cabeza lineal +
backbone completo). Mismo split anti-fuga y mismas metricas que el banco
congelado -> comparacion directa congelado-vs-fine-tuned por modelo.

Protocolo identico para los 8 modelos: AdamW (LR backbone 5e-5, LR cabeza
5e-4), weighted cross-entropy, coseno, AMP fp16, mejor epoca por balanced
accuracy de validacion.

**Como correr en Kaggle**
1. Add Input: {input_step}
2. Settings: GPU T4 + Internet ON (descarga de pesos).
3. Primero `SMOKE = True` (1 epoca con subconjunto; numeros NO validos).
4. Luego `SMOKE = False` -> Save & Run All (~1-2 h en T4 segun modelo).
5. Si hay OOM (vgg19 / vit_b16 / dinov2_vitb14): bajar `BATCH` a 16.

Resultado: `/kaggle/working/results_{model}{json_suffix}.json` +
checkpoint `ft_{model}{json_suffix}.pth` + tabla impresa.
"""

CONFIG_FT_CODE = """\
MODEL = "{model}"   # generado por scripts/build_finetune_tests.py - no editar a mano
DATASET = "{dataset}"
JSON_SUFFIX = "{json_suffix}"
SMOKE = True        # True = 1 epoca con subconjunto (numeros NO validos)
BATCH = 32          # fine-tuning necesita memoria para gradientes; OOM -> 16
EPOCHS = 8
LR_BACKBONE = 5e-5
LR_HEAD = 5e-4
SEED = 42
KNN_K = 20          # sin uso aqui; se mantiene por compatibilidad de celdas
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
print("modelo:", MODEL, "| device:", device, "| smoke:", SMOKE)
"""

TRAIN_FT_CODE = """\
# --- Fine-tuning end-to-end: se descongela el backbone y se agrega cabeza
# lineal. El assert de la celda anterior verifico el estado CONGELADO de
# carga; aqui empieza el entrenamiento real. ---
import copy
from sklearn.metrics import balanced_accuracy_score

EPOCHS_RUN = 1 if SMOKE else EPOCHS
for p in model.parameters():
    p.requires_grad = True

# La celda anterior imprime "entrenables=0" porque comparte codigo con el banco
# CONGELADO. Ese numero ya no vale aqui: se re-imprime despues de descongelar
# para que el log no haga dudar de si esto entreno de verdad.
_entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert _entrenables > 0, "el backbone sigue congelado: esto NO es fine-tuning"
print(f"{{MODEL}}: backbone DESCONGELADO, entrenables={{_entrenables:,}}")

class FTNet(nn.Module):
    def __init__(self, backbone, dim, n_cls):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, n_cls)
    def forward(self, x):
        return self.head(self.backbone(x))

net = FTNet(model, DIM, N).to(device)

tf_train = transforms.Compose([
    transforms.Resize(int(RES * 1.14), interpolation=INTERP),
    transforms.RandomCrop(RES),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
tf_eval = transforms.Compose([
    transforms.Resize(int(RES * 1.14), interpolation=INTERP),
    transforms.CenterCrop(RES),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class FTDataset(Dataset):
    def __init__(self, id_list, tf_):
        self.ids, self.tf = id_list, tf_
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(paths[self.ids[i]]).convert("RGB")
        return self.tf(img), id2label[self.ids[i]]

dl_tr = DataLoader(FTDataset(tr_ids, tf_train), batch_size=BATCH, shuffle=True,
                   num_workers=2, pin_memory=True, drop_last=True)
dl_va = DataLoader(FTDataset(va_ids, tf_eval), batch_size=BATCH, shuffle=False,
                   num_workers=2, pin_memory=True)

counts = np.bincount(y_tr, minlength=N).astype(float)
class_w = torch.tensor(counts.sum() / (N * counts), dtype=torch.float32,
                       device=device)
criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = torch.optim.AdamW([
    {{"params": net.backbone.parameters(), "lr": LR_BACKBONE}},
    {{"params": net.head.parameters(), "lr": LR_HEAD}},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, EPOCHS_RUN * len(dl_tr)))
use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

best_ba, best_sd = -1.0, None
t0 = time.time()
for epoch in range(EPOCHS_RUN):
    net.train(); running, seen = 0.0, 0
    for xb, yb in dl_tr:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(net(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update(); scheduler.step()
        running += loss.item() * len(yb); seen += len(yb)
    net.eval(); preds = []
    with torch.no_grad():
        for xb, _ in dl_va:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = net(xb.to(device, non_blocking=True))
            preds.append(out.float().argmax(1).cpu().numpy())
    ba = balanced_accuracy_score(y_va, np.concatenate(preds))
    if ba > best_ba:
        best_ba, best_sd = ba, copy.deepcopy(net.state_dict())
    print(f"epoca {{epoch + 1}}/{{EPOCHS_RUN}} loss={{running / max(1, seen):.4f}} "
          f"val_bal_acc={{ba:.4f}} mejor={{best_ba:.4f}} ({{time.time() - t0:.0f}}s)")

net.load_state_dict(best_sd)  # evaluacion final sobre la MEJOR epoca
print(f"fine-tuning listo: mejor bal_acc={{best_ba:.4f}} en {{time.time() - t0:.0f}}s")
"""

EVAL_FT_CODE = """\
# --- Evaluacion final del modelo fine-tuned (mismas metricas del banco) ---
net.eval(); probs = []
with torch.no_grad():
    for xb, _ in dl_va:
        out = net(xb.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
proba_va = np.concatenate(probs)

results = {{}}
results["finetune"] = all_metrics(y_va, proba_va)

dest = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
ckpt = dest / f"ft_{{MODEL}}{{JSON_SUFFIX}}.pth"
torch.save({{"modelo": MODEL, "dataset": DATASET, "smoke": SMOKE,
            "clases": classes, "state_dict": net.state_dict()}}, ckpt)
print("checkpoint:", ckpt)
"""


def metrics_lib_code() -> str:
    """The metric-function half of the frozen bank's probes cell.

    Everything above the probe-fitting marker defines bootstrap_ci/ece/
    all_metrics; the fitting part (kNN/logreg/prototypes) does not apply to a
    fine-tuned classifier. Slicing keeps one source of truth for the metrics.
    """
    marker = "Xn_tr, Xn_va = normalize"
    assert marker in PROBES_CODE, "build_baseline_tests.PROBES_CODE cambio de forma"
    return PROBES_CODE.split(marker)[0].replace(
        "# --- 3 probes + todas las metricas de literatura ---",
        "# --- Metricas de literatura (identicas al banco congelado) ---",
    )


def mlflow_ft_code() -> str:
    """Same MLflow cell as the frozen bank, pointed at its own experiment."""
    assert 'f"baselines-{{DATASET}}"' in MLFLOW_CODE, "MLFLOW_CODE cambio de forma"
    return MLFLOW_CODE.replace('f"baselines-{{DATASET}}"', 'f"finetune-{{DATASET}}"')


def report_ft_code() -> str:
    """Same report cell, but the CI line reads the finetune probe, not logreg."""
    assert 'results["logreg"]["ic95_bootstrap"]' in REPORT_CODE, (
        "REPORT_CODE cambio de forma")
    return (REPORT_CODE
            .replace('results["logreg"]["ic95_bootstrap"]',
                     'results["finetune"]["ic95_bootstrap"]')
            .replace('print("logreg IC95 bootstrap:',
                     'print("finetune IC95 bootstrap:'))


def build(model: str, dataset: str) -> dict:
    spec = DATASETS[dataset]
    fmt = {"model": model, "title": TITLES[model], "dataset": dataset,
           "dataset_title": spec["dataset_title"], "input_step": spec["input_step"],
           "json_suffix": "_ft" + spec["json_suffix"]}
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "cells": [
            make_cell("markdown", HEADER_FT_MD.format(**fmt)),
            make_cell("code", CONFIG_FT_CODE.format(**fmt)),
            make_cell("code", spec["data_code"](fmt)),
            make_cell("code", BACKBONE_CODE.format(**fmt)),
            make_cell("code", metrics_lib_code().format(**fmt)),
            make_cell("code", TRAIN_FT_CODE.format(**fmt)),
            make_cell("code", EVAL_FT_CODE.format(**fmt)),
            make_cell("code", report_ft_code().format(**fmt)),
            make_cell("code", mlflow_ft_code().format(**fmt)),
        ],
    }


def main() -> None:
    import py_compile
    import tempfile

    root = Path(__file__).resolve().parent.parent / "notebooks"
    for dataset, spec in DATASETS.items():
        out_dir = spec["out_dir"].replace("baselines", "finetune")
        for model in MODELS:
            nb = build(model, dataset)
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                    fh.write("".join(cell["source"]))
                    tmp = fh.name
                py_compile.compile(tmp, doraise=True)
            out = (root / out_dir / model
                   / f"finetune_{model}{spec['file_suffix']}.ipynb")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
            print("wrote", out.relative_to(root.parent))


if __name__ == "__main__":
    main()
