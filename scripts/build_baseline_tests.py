"""Generate one self-contained Kaggle test notebook per baseline model.

Output:
  notebooks/baselines/<model>/test_<model>.ipynb            (HAM10000)
  notebooks/baselines_isic2020/<model>/test_<model>_isic2020.ipynb

Each notebook is independent (no dermssl import) so any team member can
upload a single file to Kaggle and run that model's test. All notebooks share
the same template; only the MODEL and DATASET constants change, which
guarantees an identical protocol across models AND datasets.

Run: python scripts/build_baseline_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path

MODELS = [
    "alexnet",
    "vgg19",
    "efficientnet_b0",
    "resnet50",
    "vit_b16",
    "simsiam_r50",
    "dinov2_vits14",
    "dinov2_vitb14",
]

TITLES = {
    "alexnet": "AlexNet (2012) — supervisado, ImageNet-1k",
    "vgg19": "VGG19 (2014) — supervisado, ImageNet-1k",
    "efficientnet_b0": "EfficientNet-B0 (2019) — supervisado, ImageNet-1k",
    "resnet50": "ResNet-50 (2015) — supervisado, ImageNet-1k",
    "vit_b16": "ViT-B/16 (2021) — supervisado (Attention-Transformer)",
    "simsiam_r50": "SimSiam ResNet-50 (2021) — SSL sobre ImageNet-1k sin etiquetas",
    "dinov2_vits14": "DINOv2 ViT-S/14 (2023) — SSL sobre LVD-142M",
    "dinov2_vitb14": "DINOv2 ViT-B/14 (2023) — SSL sobre LVD-142M "
                     "(par justo de tamano vs ViT-B/16 supervisado)",
}

HEADER_MD = """\
# Test de baseline: {title}

Protocolo identico para los 8 modelos del banco de pruebas (Objetivo 2):
backbone **congelado** (cero fine-tuning) sobre **{dataset_title}**, split
por grupo (anti-fuga), tres probes (kNN, regresion logistica, prototipos)
y todas las metricas: accuracy, balanced accuracy, F1/precision/recall macro
y por clase, AUC (OvR macro), ECE, matriz de confusion e IC bootstrap 95%.

**Como correr en Kaggle**
1. Add Input: {input_step}
2. Settings: GPU T4 + Internet ON (descarga de pesos).
3. Primero `SMOKE = True` (valida el pipeline en minutos; numeros NO validos).
4. Luego `SMOKE = False` -> Save & Run All.

Resultado: `/kaggle/working/results_{model}{json_suffix}.json` + tabla impresa.
"""

CONFIG_CODE = """\
MODEL = "{model}"   # generado por scripts/build_baseline_tests.py - no editar a mano
DATASET = "{dataset}"
JSON_SUFFIX = "{json_suffix}"
SMOKE = True        # True = validacion rapida del pipeline (numeros NO validos)
BATCH = 64
SEED = 42
KNN_K = 20
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

DATA_CODE = """\
# --- HAM10000 desde /kaggle/input, split anti-fuga por lesion_id ---
from sklearn.model_selection import StratifiedGroupKFold

root = Path("/kaggle/input")
meta_csv = next(root.glob("**/HAM10000_metadata.csv"))
meta = pd.read_csv(meta_csv)
paths = {{p.stem: p for p in root.glob("**/*.jpg")}}
meta = meta[meta.image_id.isin(paths)].reset_index(drop=True)
classes = sorted(meta.dx.unique())
cls2idx = {{c: i for i, c in enumerate(classes)}}
id2label = {{r.image_id: cls2idx[r.dx] for r in meta.itertuples()}}
N = len(classes)
print("clases:", classes); print(meta.dx.value_counts().to_string())

ids = meta.image_id.tolist()
y = np.array([id2label[i] for i in ids])
groups = meta.lesion_id.values
tr_idx, va_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                      .split(ids, y, groups))
tr_ids = [ids[i] for i in tr_idx]; va_ids = [ids[i] for i in va_idx]
lk = meta.set_index("image_id")
shared = set(lk.loc[tr_ids, "lesion_id"]) & set(lk.loc[va_ids, "lesion_id"])
assert not shared, f"LEAKAGE: {{len(shared)}} lesiones compartidas"

if SMOKE:  # subconjunto estratificado: humo en minutos
    rng = np.random.default_rng(SEED)
    def subsample(id_list, frac, min_pc):
        yy = np.array([id2label[i] for i in id_list]); keep = []
        for c in range(N):
            idx = np.flatnonzero(yy == c)
            take = max(min_pc, int(len(idx) * frac))
            keep.extend(rng.choice(idx, size=min(take, len(idx)), replace=False))
        return [id_list[i] for i in sorted(keep)]
    tr_ids = subsample(tr_ids, 0.08, 12); va_ids = subsample(va_ids, 0.08, 6)

y_tr = np.array([id2label[i] for i in tr_ids])
y_va = np.array([id2label[i] for i in va_ids])
SPLIT_DESC = "StratifiedGroupKFold por lesion_id (0 lesiones compartidas)"
print(f"train={{len(tr_ids)}} val={{len(va_ids)}} | lesiones compartidas=0")
"""

DATA_CODE_ISIC2019MC = """\
# --- ISIC 2019 (andrewmvd/isic-2019) desde /kaggle/input: tarea MULTICLASE de
# 8 clases, la misma del challenge oficial. Es el UNICO banco cuya tarea coincide
# con el ancla de Gessert (0.636 de balanced accuracy en el test oficial), asi
# que es el unico comparable de frente contra ella.
#
# LIMITE que viaja con estos numeros: este release NO trae columna de paciente
# (y el patient_id del RESIZED es sintetico), asi que NO hay split anti-fuga por
# paciente. El split es estratificado simple. Dos imagenes del mismo paciente
# pueden caer en train y en val, lo que sesga las metricas hacia arriba. Por eso
# este banco NO es comparable 1:1 con HAM10000 ni con ISIC 2020, que si agrupan
# por lesion_id y patient_id reales. ---
from sklearn.model_selection import StratifiedKFold

root = Path("/kaggle/input")
gt_csv = next(root.glob("**/ISIC_2019_Training_GroundTruth.csv"))
meta = pd.read_csv(gt_csv)
id_col = meta.columns[0]                      # "image"
# UNK sale en cero en todo el release: conservarla inventaria una novena clase
# que nunca ocurre y hundiria todas las metricas macro por un factor 9/8.
onehot = meta.drop(columns=[id_col] + [c for c in ("UNK",) if c in meta.columns])
classes = list(onehot.columns)
N = len(classes)
labels = onehot.idxmax(axis=1)                # one-hot -> nombre de clase
cls2idx = {{c: i for i, c in enumerate(classes)}}
paths = {{p.stem: p for p in root.glob("**/*.jpg")}}
meta = meta.assign(_label=labels)
meta = meta[meta[id_col].astype(str).isin(paths)].reset_index(drop=True)
id2label = {{str(r): cls2idx[l] for r, l in zip(meta[id_col], meta._label)}}
print("clases:", classes); print(meta._label.value_counts().to_string())

ids = meta[id_col].astype(str).tolist()
y = np.array([id2label[i] for i in ids])
tr_idx, va_idx = next(StratifiedKFold(5, shuffle=True, random_state=SEED)
                      .split(ids, y))
tr_ids = [ids[i] for i in tr_idx]; va_ids = [ids[i] for i in va_idx]
assert not set(tr_ids) & set(va_ids), "una imagen quedo en train y en val"

if SMOKE:  # subconjunto estratificado: humo en minutos
    rng = np.random.default_rng(SEED)
    def subsample(id_list, frac, min_pc):
        yy = np.array([id2label[i] for i in id_list]); keep = []
        for c in range(N):
            idx = np.flatnonzero(yy == c)
            take = max(min_pc, int(len(idx) * frac))
            keep.extend(rng.choice(idx, size=min(take, len(idx)), replace=False))
        return [id_list[i] for i in sorted(keep)]
    tr_ids = subsample(tr_ids, 0.08, 12); va_ids = subsample(va_ids, 0.08, 6)

y_tr = np.array([id2label[i] for i in tr_ids])
y_va = np.array([id2label[i] for i in va_ids])
SPLIT_DESC = ("StratifiedKFold estratificado simple SIN grupo de paciente "
              "(el release no trae patient_id) - riesgo de fuga entre imagenes "
              "del mismo paciente, metricas con sesgo optimista")
print(f"train={{len(tr_ids)}} val={{len(va_ids)}} | SIN agrupacion por paciente")
"""

DATA_CODE_ISICBIN = """\
# --- ISIC {year} RESIZED desde /kaggle/input: tarea BINARIA (0 benigno / 1 maligno).
# Gotcha documentado (docs/15): los datasets RESIZED solo traen target binario;
# la variante multiclase de ISIC 2019 requiere andrewmvd/isic-2019. ---
from sklearn.model_selection import StratifiedGroupKFold

root = Path("/kaggle/input")
meta_csv = [p for p in root.glob("**/train-metadata.csv") if "{year}" in str(p)]
assert meta_csv, "no encontre train-metadata.csv del dataset ISIC {year}"
meta = pd.read_csv(meta_csv[0])
id_col = next(c for c in ("isic_id", "image_name", "image_id", "image")
              if c in meta.columns)
assert "target" in meta.columns, f"sin columna target: {{list(meta.columns)}}"
paths = {{p.stem: p for p in root.glob("**/*.jpg") if "{year}" in str(p)}}
meta = meta[meta[id_col].astype(str).isin(paths)].reset_index(drop=True)
classes = ["benign", "malignant"]; N = 2
id2label = {{str(r): int(t) for r, t in zip(meta[id_col], meta.target)}}
print(f"ISIC {year}: n={{len(meta)}}"); print(meta.target.value_counts().to_string())

ids = meta[id_col].astype(str).tolist()
y = np.array([id2label[i] for i in ids])
if "{use_patient_groups}" == "yes" and "patient_id" in meta.columns:
    groups = meta.patient_id.astype(str).values
    SPLIT_DESC = "StratifiedGroupKFold por patient_id (0 pacientes compartidos)"
else:
    # ISIC 2019 RESIZED: patient_id es sintetico ("randomly generated") ->
    # no hay grupo real; split estratificado simple, documentado como limite.
    groups = np.array(ids)
    SPLIT_DESC = "estratificado simple (sin grupo real disponible en RESIZED)"
tr_idx, va_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                      .split(ids, y, groups))
tr_ids = [ids[i] for i in tr_idx]; va_ids = [ids[i] for i in va_idx]
g = dict(zip(ids, groups))
shared = {{g[i] for i in tr_ids}} & {{g[i] for i in va_ids}}
assert not shared, f"LEAKAGE: {{len(shared)}} grupos compartidos"

if SMOKE:  # subconjunto estratificado: humo en minutos
    rng = np.random.default_rng(SEED)
    def subsample(id_list, frac, min_pc):
        yy = np.array([id2label[i] for i in id_list]); keep = []
        for c in range(N):
            idx = np.flatnonzero(yy == c)
            take = max(min_pc, int(len(idx) * frac))
            keep.extend(rng.choice(idx, size=min(take, len(idx)), replace=False))
        return [id_list[i] for i in sorted(keep)]
    tr_ids = subsample(tr_ids, 0.04, 40); va_ids = subsample(va_ids, 0.04, 20)

y_tr = np.array([id2label[i] for i in tr_ids])
y_va = np.array([id2label[i] for i in va_ids])
print(f"train={{len(tr_ids)}} val={{len(va_ids)}} | {{SPLIT_DESC}}")
"""

BACKBONE_CODE = """\
# --- Backbone CONGELADO (nunca se entrena). Fuente de pesos: Hugging Face Hub
# via timm (decision del equipo con la Dra. Karen, 2026-08-28). Los tags
# .tv_in1k / .tv2_in1k de timm sirven LOS MISMOS tensores que torchvision ->
# resnet50 sigue comparable con las corridas previas del banco. OJO: los mismos
# tensores NO implican el mismo punto de extraccion — ver el caso vgg19 abajo.
# Excepciones documentadas (sin pesos oficiales en HF): alexnet (torchvision)
# y simsiam_r50 (checkpoint oficial de FAIR). model.eval() SIEMPRE:
# requires_grad=False no congela BatchNorm. ---
RES, INTERP = 224, transforms.InterpolationMode.BILINEAR

# Ancho de features esperado: GAP del ultimo bloque conv (CNN) o token CLS
# (ViT). Se asserta abajo — un cambio de punto de extraccion rompe la
# comparabilidad del banco y tiene que fallar fuerte, no en silencio.
FEATURE_DIMS = {{
    "alexnet": 256, "vgg19": 512, "efficientnet_b0": 1280, "resnet50": 2048,
    "vit_b16": 768, "simsiam_r50": 2048, "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}}

TIMM_HF_TAGS = {{
    "vgg19": "vgg19.tv_in1k",                       # = torchvision IMAGENET1K_V1
    "efficientnet_b0": "efficientnet_b0.ra_in1k",   # receta original de timm
    "resnet50": "resnet50.tv2_in1k",                # = torchvision IMAGENET1K_V2
    "vit_b16": "vit_base_patch16_224.augreg_in1k",  # supervisado SOLO in1k (fair)
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
}}

try:
    import timm
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

def gap(features):
    return nn.Sequential(features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

def load_backbone(name):
    global RES, INTERP
    if name == "alexnet":
        # Sin pesos oficiales en HF Hub: se mantiene torchvision, documentado.
        from torchvision.models import AlexNet_Weights, alexnet
        return gap(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features)
    if name == "simsiam_r50":
        # Checkpoint oficial de FAIR (no publicado en HF). Pendiente: mirror
        # en el repo HF del equipo para fijar revision.
        from torchvision.models import resnet50
        url = ("https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/"
               "pretrain/checkpoint_0099.pth.tar")
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")["state_dict"]
        sd = {{k.replace("module.encoder.", ""): v for k, v in sd.items()
              if k.startswith("module.encoder") and "fc" not in k}}
        m = resnet50(weights=None); m.fc = nn.Identity()
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, f"claves inesperadas: {{unexpected[:5]}}"
        return m
    tag = TIMM_HF_TAGS[name]
    if name == "efficientnet_b0":
        INTERP = transforms.InterpolationMode.BICUBIC  # su distribucion de entrenamiento
    kwargs = {{"img_size": RES}} if "patch14" in tag else {{}}
    m = timm.create_model(tag, pretrained=True, num_classes=0, **kwargs)
    if name == "vgg19":
        # num_classes=0 deja las pre-logits de 4096 (fc6/fc7 de la cabeza); el
        # banco lee toda CNN en el GAP de su ultimo bloque conv.
        return gap(m.features)
    return m

model = load_backbone(MODEL)
for p in model.parameters():
    p.requires_grad = False
model.eval().to(device)
with torch.no_grad():
    DIM = model(torch.zeros(1, 3, RES, RES, device=device)).shape[-1]
trainable = sum(p.requires_grad for p in model.parameters())
assert trainable == 0, "el backbone NO esta congelado"
assert DIM == FEATURE_DIMS[MODEL], (
    f"{{MODEL}}: se extrajeron {{DIM}} dims, el protocolo fija {{FEATURE_DIMS[MODEL]}}. "
    "El punto de extraccion cambio; la corrida NO es comparable con el banco.")
print(f"{{MODEL}}: dim={{DIM}} res={{RES}} interp={{INTERP}} entrenables=0")
"""

EXTRACT_CODE = """\
# --- Extraccion de features (sin aumentos: reproducible) ---
tf = transforms.Compose([
    transforms.Resize(int(RES * 1.14), interpolation=INTERP),
    transforms.CenterCrop(RES),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class ImgDataset(Dataset):
    def __init__(self, id_list):
        self.ids = id_list
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(paths[self.ids[i]]).convert("RGB")
        return tf(img), id2label[self.ids[i]]

@torch.no_grad()
def extract(id_list):
    dl = DataLoader(ImgDataset(id_list), batch_size=BATCH, shuffle=False,
                    num_workers=2, pin_memory=True)
    feats, labels = [], []
    for xb, yb in dl:
        out = model(xb.to(device, non_blocking=True))
        feats.append(out.float().cpu().numpy()); labels.append(yb.numpy())
    return np.concatenate(feats), np.concatenate(labels)

t0 = time.time()
X_tr, y_tr2 = extract(tr_ids); X_va, y_va2 = extract(va_ids)
assert (y_tr2 == y_tr).all() and (y_va2 == y_va).all()
print(f"features: train={{X_tr.shape}} val={{X_va.shape}} en {{time.time()-t0:.0f}}s")
"""

PROBES_CODE = """\
# --- 3 probes + todas las metricas de literatura ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

def bootstrap_ci(y_true, pred, n_resamples=1000):
    # IC 95 percentil sobre remuestreo del set de validacion. La diferencia
    # entre "0.629" y "0.629 [0.61, 0.65]" es lo primero que pregunta un jurado.
    rng = np.random.default_rng(SEED)
    idx_all = np.arange(len(y_true)); ba, f1v = [], []
    for _ in range(n_resamples):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        ba.append(balanced_accuracy_score(y_true[idx], pred[idx]))
        f1v.append(f1_score(y_true[idx], pred[idx], average="macro"))
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {{"balanced_accuracy": pct(ba), "macro_f1": pct(f1v)}}

def ece(y_true, proba, bins=15):
    conf = proba.max(1); pred = proba.argmax(1)
    ok = (pred == y_true).astype(float); edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(tot)

def all_metrics(y_true, proba):
    pred = proba.argmax(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {{
            "accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro"),
            "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
            # binario: sklearn exige la proba de la clase positiva (1-D);
            # multiclase: OvR macro sobre la matriz completa.
            "auc_ovr_macro": (roc_auc_score(y_true, proba[:, 1]) if N == 2 else
                              roc_auc_score(y_true, proba, multi_class="ovr",
                                            average="macro", labels=list(range(N)))),
            "ece": ece(y_true, proba),
            "recall_por_clase": {{classes[c]: float(recall_score(
                y_true, pred, labels=[c], average="macro", zero_division=0))
                for c in range(N)}},
            "precision_por_clase": {{classes[c]: float(precision_score(
                y_true, pred, labels=[c], average="macro", zero_division=0))
                for c in range(N)}},
            "f1_por_clase": {{classes[c]: float(f1_score(
                y_true, pred, labels=[c], average="macro", zero_division=0))
                for c in range(N)}},
            "matriz_confusion": confusion_matrix(
                y_true, pred, labels=list(range(N))).tolist(),
            "ic95_bootstrap": bootstrap_ci(y_true, pred),
        }}

Xn_tr, Xn_va = normalize(X_tr), normalize(X_va)   # L2: protocolo DINO para kNN
results = {{}}

knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine", weights="distance")
knn.fit(Xn_tr, y_tr)
results["knn"] = all_metrics(y_va, knn.predict_proba(Xn_va))

logreg = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
logreg.fit(X_tr, y_tr)
results["logreg"] = all_metrics(y_va, logreg.predict_proba(X_va))

protos = normalize(np.stack([Xn_tr[y_tr == c].mean(0) for c in range(N)]))
sims = Xn_va @ protos.T
proba_proto = np.exp(sims * 10); proba_proto /= proba_proto.sum(1, keepdims=True)
results["prototype"] = all_metrics(y_va, proba_proto)
"""

REPORT_CODE = """\
# --- Reporte final + JSON ---
out = {{
    "modelo": MODEL, "dataset": DATASET, "smoke": SMOKE,
    "n_train": len(tr_ids), "n_val": len(va_ids),
    "dim_features": int(DIM), "resolucion": RES, "semilla": SEED,
    "split": SPLIT_DESC,
    "probes": results,
}}
dest = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
(dest / f"results_{{MODEL}}{{JSON_SUFFIX}}.json").write_text(json.dumps(out, indent=2))

hdr = ["probe", "bal_acc", "macro_f1", "precision", "recall", "auc", "acc", "ece"]
print(("ATENCION: numeros de SMOKE, no validos\\n" if SMOKE else "") + "-" * 86)
print(f"{{MODEL}} | val n={{len(va_ids)}}")
print("{{:<11}}{{:>9}}{{:>10}}{{:>11}}{{:>9}}{{:>8}}{{:>8}}{{:>8}}".format(*hdr))
for probe, m in results.items():
    print("{{:<11}}{{:>9.4f}}{{:>10.4f}}{{:>11.4f}}{{:>9.4f}}{{:>8.4f}}{{:>8.4f}}{{:>8.4f}}".format(
        probe, m["balanced_accuracy"], m["macro_f1"], m["macro_precision"],
        m["macro_recall"], m["auc_ovr_macro"], m["accuracy"], m["ece"]))
print("-" * 86)
ci = results["logreg"]["ic95_bootstrap"]
print("logreg IC95 bootstrap: bal_acc {{:.3f}}-{{:.3f}} | F1 macro {{:.3f}}-{{:.3f}}".format(
    *ci["balanced_accuracy"], *ci["macro_f1"]))
if DATASET == "ham10000":
    print("ancla de literatura: ganador ISIC 2019 (Gessert) = 0.636 balanced accuracy")
print("guardado:", dest / f"results_{{MODEL}}{{JSON_SUFFIX}}.json")
"""


MLFLOW_CODE = """\
# --- Trazabilidad MLflow (no fatal: si falta, el test ya corrio igual) ---
# Mismo criterio que v6 (docs/16): backend de archivos, sin servidor.
# En Kaggle: los runs quedan en /kaggle/working/mlruns (descargables como Output).
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # MLflow 3.x: file store pide opt-in
try:
    import mlflow
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlflow-skinny"],
                   check=False)
    try:
        import mlflow
    except ImportError:
        mlflow = None
        print("AVISO: mlflow no disponible - el test corrio igual; JSON ya guardado")

if mlflow is not None:
    mlflow.set_tracking_uri(f"file:{{dest / 'mlruns'}}")
    mlflow.set_experiment(f"baselines-{{DATASET}}")
    with mlflow.start_run(run_name=f"{{MODEL}}{{'-smoke' if SMOKE else ''}}"):
        mlflow.log_params({{
            "modelo": MODEL, "dataset": DATASET, "smoke": SMOKE, "semilla": SEED,
            "resolucion": RES, "dim_features": int(DIM), "n_train": len(tr_ids),
            "n_val": len(va_ids), "knn_k": KNN_K, "split": SPLIT_DESC,
        }})
        for probe, m in results.items():
            for name, val in m.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"{{probe}}.{{name}}", val)
        mlflow.log_artifact(str(dest / f"results_{{MODEL}}{{JSON_SUFFIX}}.json"))
    print("MLflow: run registrado en", dest / "mlruns")
    print("ver UI local: MLFLOW_ALLOW_FILE_STORE=true mlflow ui "
          "--backend-store-uri file:" + str(dest / "mlruns"))
"""


def make_cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


DATASETS = {
    "ham10000": {
        "dataset_title": "HAM10000 (7 clases)",
        "input_step": "`Skin Cancer MNIST: HAM10000` (kmader/skin-cancer-mnist-ham10000).",
        "json_suffix": "",
        "data_code": lambda fmt: DATA_CODE.format(**fmt),
        "out_dir": "baselines",
        "file_suffix": "",
    },
    "isic2020": {
        "dataset_title": "ISIC 2020 RESIZED (binaria: benigno/maligno)",
        "input_step": "`ISIC 2020 JPG 256x256 RESIZED` (el mismo del pool SSL).",
        "json_suffix": "_isic2020",
        "data_code": lambda fmt: DATA_CODE_ISICBIN.format(
            year="2020", use_patient_groups="yes", **fmt),
        "out_dir": "baselines_isic2020",
        "file_suffix": "_isic2020",
    },
    "isic2019": {
        "dataset_title": "ISIC 2019 (multiclase, 8 clases del challenge oficial)",
        "input_step": "`ISIC 2019` (andrewmvd/isic-2019), el release con ground truth multiclase.",
        "json_suffix": "_isic2019",
        "data_code": lambda fmt: DATA_CODE_ISIC2019MC.format(**fmt),
        "out_dir": "baselines_isic2019",
        "file_suffix": "_isic2019",
    },
}


def build(model: str, dataset: str) -> dict:
    spec = DATASETS[dataset]
    fmt = {"model": model, "title": TITLES[model], "dataset": dataset,
           "dataset_title": spec["dataset_title"], "input_step": spec["input_step"],
           "json_suffix": spec["json_suffix"]}
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "cells": [
            make_cell("markdown", HEADER_MD.format(**fmt)),
            make_cell("code", CONFIG_CODE.format(**fmt)),
            make_cell("code", spec["data_code"](fmt)),
            make_cell("code", BACKBONE_CODE.format(**fmt)),
            make_cell("code", EXTRACT_CODE.format(**fmt)),
            make_cell("code", PROBES_CODE.format(**fmt)),
            make_cell("code", REPORT_CODE.format(**fmt)),
            make_cell("code", MLFLOW_CODE.format(**fmt)),
        ],
    }


def main() -> None:
    import py_compile
    import tempfile

    root = Path(__file__).resolve().parent.parent / "notebooks"
    for dataset, spec in DATASETS.items():
        for model in MODELS:
            nb = build(model, dataset)
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                    fh.write("".join(cell["source"]))
                    tmp = fh.name
                py_compile.compile(tmp, doraise=True)
            out = (root / spec["out_dir"] / model
                   / f"test_{model}{spec['file_suffix']}.ipynb")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
            print("wrote", out.relative_to(root.parent))


if __name__ == "__main__":
    main()
