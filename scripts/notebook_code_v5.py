"""Executable cells for the v5 notebook (the torch-dependent half).

The numpy-only logic is NOT written here: it is inlined from ``src/dermssl/`` by
``build_notebook_v5.py``, so the code that ships in the notebook is byte-for-byte
the code covered by ``tests/``.

Cells are stored as raw strings so that escape sequences meant for the notebook
(``\n`` inside an f-string, for example) survive this file untouched.
"""

CONFIG_CODE = r'''import gc, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # una imagen truncada NO debe tumbar el run
T_START = time.time()

def elapsed():
    return f"{(time.time() - T_START) / 60:.1f} min"

# ----------------------------------------------------------------------------
# PERILLAS. Lo unico que se toca a mano.
# ----------------------------------------------------------------------------
SMOKE        = True          # True = validacion rapida del pipeline (numeros NO validos)
DATA_BACKEND = "auto"        # "auto" | "kaggle-input" | "kagglehub" | "local"
LOCAL_ROOT   = None          # ruta de datos ya bajados (para el cluster)
BASE_RES     = 224           # se ajusta por backbone (ViT patch 14 vs 16)
BATCH        = 64
NUM_WORKERS  = 4             # Kaggle da ~4 vCPU con GPU
MAX_HOURS    = 10.5          # guarda de tiempo (Kaggle corta a las 12 h)

# Backbones a comparar. Ninguno se entrena: solo se leen sus features.
#   ssl_*  -> pre-entrenado SIN etiquetas  (la hipotesis)
#   sup_*  -> pre-entrenado CON etiquetas  (el control)
BACKBONES = [
    "ssl_dinov2_vits14",   # DINOv2 ViT-S/14 - barato, cabe en T4 de sobra
    "sup_vit_b16",         # ViT-B/16 ImageNet supervisado - CONTROL de arquitectura
    "sup_resnet50",        # ResNet-50 ImageNet supervisado - linea base de v4
    "sup_alexnet",         # AlexNet 2012 - piso historico de la escalera
    "sup_vgg19",           # VGG19 2014 - profundidad bruta (sin BN, como el paper)
    "sup_effnet_b0",       # EfficientNet-B0 2019 - eficiencia (5.3M params)
]
# Cinco probes. Los "_balanced" reponderan por frecuencia de clase: se reportan AL
# LADO de los normales, no en su lugar, porque son estimadores distintos y cual
# conviene es una pregunta empirica que la corrida responde.
PROBES    = ["knn", "knn_balanced", "prototype", "logreg", "logreg_balanced"]
FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.50, 1.00)
SEEDS     = [42, 1234, 2026]

if SMOKE:
    BACKBONES = ["ssl_dinov2_vits14", "sup_resnet50"]
    PROBES    = ["knn", "knn_balanced", "logreg", "logreg_balanced"]
    FRACTIONS = (0.05, 1.00)
    SEEDS     = [42]
    print("*** MODO SMOKE: los numeros NO son validos como resultado. "
          "Solo prueba que el pipeline corre entero. ***")

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("./out")
WORK.mkdir(parents=True, exist_ok=True)
CACHE_DIR = WORK / "features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = device.type == "cuda"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

print("torch:", torch.__version__, "| device:", device, "| GPUs:", torch.cuda.device_count())
print(f"backbones={BACKBONES}\nprobes={PROBES}\nfracciones={FRACTIONS} | semillas={SEEDS}")'''

DATA_CODE = r'''# index_pool=False: el pool sin etiquetas NO se usa con el backbone congelado, y
# recorrer las ~48k imagenes de ISIC costo 157 de los 192 s de la corrida del 29-07.
# Se activa (index_pool=True) recien en el pre-entrenamiento continuo.
data = resolve_datasets(backend=DATA_BACKEND, local_root=LOCAL_ROOT, index_pool=False)
print(json.dumps(data.summary(), indent=2))

meta = pd.read_csv(data.ham_metadata_csv)
meta = meta[meta.image_id.isin(data.ham_images)].reset_index(drop=True)
classes = sorted(meta.dx.unique())
cls2idx = {c: i for i, c in enumerate(classes)}
id2label = {r.image_id: cls2idx[r.dx] for r in meta.itertuples()}
N_CLASSES = len(classes)
print("\nclases:", classes)
print(meta.dx.value_counts().to_string())'''

SPLIT_CODE = r'''from sklearn.model_selection import StratifiedGroupKFold

all_ids = meta.image_id.tolist()
y_all   = np.array([id2label[i] for i in all_ids])
groups  = meta.set_index("image_id").loc[all_ids, "lesion_id"].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = next(sgkf.split(all_ids, y_all, groups))
tr_ids = [all_ids[i] for i in tr_idx]
va_ids = [all_ids[i] for i in va_idx]

lookup  = meta.set_index("image_id")
overlap = set(lookup.loc[tr_ids, "lesion_id"]) & set(lookup.loc[va_ids, "lesion_id"])
assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} lesiones en train y val a la vez"

if SMOKE:  # subconjunto estratificado para que el humo corra en minutos
    y_tr_full = np.array([id2label[i] for i in tr_ids])
    y_va_full = np.array([id2label[i] for i in va_ids])
    tr_ids = [tr_ids[i] for i in stratified_regime_indices(y_tr_full, 0.08, seed=42, min_per_class=12)]
    va_ids = [va_ids[i] for i in stratified_regime_indices(y_va_full, 0.08, seed=42, min_per_class=6)]

y_train = np.array([id2label[i] for i in tr_ids])
y_val   = np.array([id2label[i] for i in va_ids])
print(f"train={len(tr_ids)} | val={len(va_ids)} | lesiones compartidas={len(overlap)} (debe ser 0)")
print("regimenes de etiquetas disponibles:")
for row in describe_regimes(y_train, build_regimes(y_train, FRACTIONS, seed=SEEDS[0])):
    print(f"  {row['fraction']:>5.0%} -> {row['n_labels']:>5} etiquetas "
          f"({row['pct_of_train']:>5.1f}% del train) | clase mas chica: {row['min_class_count']}")'''

BACKBONE_CODE = r'''from torchvision.models import (
    AlexNet_Weights, EfficientNet_B0_Weights, ResNet50_Weights, VGG19_Weights,
    ViT_B_16_Weights, alexnet, efficientnet_b0, resnet50, vgg19, vit_b_16)

BACKBONE_SPECS = {
    "ssl_dinov2_vits14": {"patch_name": "dinov2_vits14", "labels_used": 0,
                          "family": "vit", "pretrain": "DINOv2 LVD-142M (sin etiquetas)"},
    "ssl_dinov2_vitb14": {"patch_name": "dinov2_vitb14", "labels_used": 0,
                          "family": "vit", "pretrain": "DINOv2 LVD-142M (sin etiquetas)"},
    "sup_vit_b16":       {"patch_name": "vit_base_patch16_224", "labels_used": 1_281_167,
                          "family": "vit", "pretrain": "ImageNet-1k supervisado"},
    "sup_resnet50":      {"patch_name": "resnet50", "labels_used": 1_281_167,
                          "family": "cnn", "pretrain": "ImageNet-1k supervisado"},
    "ssl_simsiam_r50":   {"patch_name": "resnet50", "labels_used": 1_281_167,
                          "family": "cnn", "pretrain": "SimSiam sobre init ImageNet (v4)"},
    "sup_alexnet":       {"patch_name": "alexnet", "labels_used": 1_281_167,
                          "family": "cnn", "pretrain": "ImageNet-1k supervisado (variante 2014)"},
    "sup_vgg19":         {"patch_name": "vgg19", "labels_used": 1_281_167,
                          "family": "cnn", "pretrain": "ImageNet-1k supervisado"},
    "sup_effnet_b0":     {"patch_name": "efficientnet_b0", "labels_used": 1_281_167,
                          "family": "cnn", "pretrain": "ImageNet-1k supervisado (AutoAugment)"},
}

# Punto de extraccion unico para CNNs sin avgpool+fc integrados: GAP del ultimo
# bloque convolucional (mismo protocolo que resnet50 con fc=Identity). Mezclar
# fc7 (4096-d) en unas y pooling en otras compararia protocolos, no modelos.
def _gap_wrapper(features_module):
    return nn.Sequential(features_module, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

def _load_dinov2(name, res):
    """Tres rutas, en orden de preferencia. Devuelve (modelo, como_se_cargo)."""
    errors = []
    size = "small" if "vits" in name else "base"
    try:  # 1) timm: preinstalado en Kaggle, pesos desde Hugging Face
        import timm
        model = timm.create_model(f"vit_{size}_patch14_dinov2.lvd142m",
                                  pretrained=True, num_classes=0, img_size=res)
        return model, "timm/lvd142m"
    except Exception as exc:
        errors.append(f"timm: {exc}")
    try:  # 2) torch.hub oficial de facebookresearch
        model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
        return model, "torch.hub/facebookresearch"
    except Exception as exc:
        errors.append(f"torch.hub: {exc}")
    local = list(Path("/kaggle/input").glob(f"**/{name}*.pth")) if Path("/kaggle/input").is_dir() else []
    if local:  # 3) checkpoint subido como Input (para sesiones sin internet)
        import timm
        model = timm.create_model(f"vit_{size}_patch14_dinov2", pretrained=False,
                                  num_classes=0, img_size=res)
        model.load_state_dict(torch.load(local[0], map_location="cpu"), strict=False)
        return model, f"local:{local[0].name}"
    raise RuntimeError(
        "No se pudo cargar DINOv2. Activa Internet en Settings, o sube los pesos como "
        "Input dataset. Intentos:\n  - " + "\n  - ".join(errors))

def build_frozen_backbone(key, res):
    """Devuelve (modelo_en_eval, dim_features, fuente). NUNCA entrena."""
    if key.startswith("ssl_dinov2"):
        model, source = _load_dinov2(BACKBONE_SPECS[key]["patch_name"], res)
    elif key == "sup_vit_b16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()          # se queda el token CLS crudo
        source = "torchvision/IMAGENET1K_V1"
    elif key in ("sup_resnet50", "ssl_simsiam_r50"):
        if key == "ssl_simsiam_r50":
            ckpts = list(Path("/kaggle/input").glob("**/backbone_domain_adaptive_r50.pth"))
            if not ckpts:
                raise FileNotFoundError("falta backbone_domain_adaptive_r50.pth (salida de v4)")
            model = resnet50(weights=None)
            model.load_state_dict(torch.load(ckpts[0], map_location="cpu"), strict=False)
            source = f"v4:{ckpts[0].name}"
        else:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            source = "torchvision/IMAGENET1K_V2"
        model.fc = nn.Identity()
    elif key == "sup_alexnet":
        model = _gap_wrapper(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features)
        source = "torchvision/IMAGENET1K_V1 (One-weird-trick 2014)"
    elif key == "sup_vgg19":
        model = _gap_wrapper(vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features)
        source = "torchvision/IMAGENET1K_V1 (sin BN, como el paper)"
    elif key == "sup_effnet_b0":
        model = _gap_wrapper(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features)
        source = "torchvision/IMAGENET1K_V1"
    else:
        raise ValueError(f"backbone desconocido: {key}")

    for p in model.parameters():
        p.requires_grad = False              # congelado, de forma explicita y verificable
    model.eval().to(device)
    with torch.no_grad():
        dim = model(torch.zeros(1, 3, res, res, device=device)).shape[-1]
    return model, int(dim), source

# Resolucion por backbone: se ajusta al tamano de parche y se reporta el cambio.
# torchvision/vit_b_16 lleva la resolucion incrustada en sus pesos: solo acepta 224.
FIXED_RES = {"sup_vit_b16": 224}
RES_BY_BACKBONE = {}
for key in BACKBONES:
    if key in FIXED_RES:
        res, note = FIXED_RES[key], f"fija en {FIXED_RES[key]} (pesos de torchvision)"
    else:
        res, note = resolve_resolution(BASE_RES, BACKBONE_SPECS[key]["patch_name"])
    RES_BY_BACKBONE[key] = res
    print(f"{key:<20} res={res:<4} patch={patch_size_for(BACKBONE_SPECS[key]['patch_name'])} | {note}")
print(f"\n[control de la trampa] 256 con parche 14 -> {resolve_resolution(256, 'dinov2_vitb14')[1]}")'''

EXTRACT_CODE = r'''cache = FeatureCache(CACHE_DIR)

# EfficientNet-B0 fue entrenado con remuestreo BICUBIC; evaluarlo con BILINEAR
# lo saca de su distribucion (regla: preprocesamiento por modelo, docs/17).
INTERP_BY_BACKBONE = {"sup_effnet_b0": transforms.InterpolationMode.BICUBIC}

def eval_transform(res, interp=transforms.InterpolationMode.BILINEAR):
    """Sin aumentos ni aleatoriedad: la extraccion debe ser reproducible."""
    return transforms.Compose([
        transforms.Resize(int(res * 1.14), interpolation=interp),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] imagen ilegible {path}: {exc} (placeholder negro)")
        return Image.new("RGB", (224, 224), (0, 0, 0))

class ImageIdDataset(Dataset):
    def __init__(self, ids, tf):
        self.ids, self.tf = ids, tf
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        image_id = self.ids[i]
        return self.tf(load_rgb(data.ham_images[image_id])), id2label[image_id]

@torch.no_grad()
def extract(model, ids, res, interp=transforms.InterpolationMode.BILINEAR):
    dl = DataLoader(ImageIdDataset(ids, eval_transform(res, interp)), batch_size=BATCH,
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    feats, labels = [], []
    for images, batch_labels in dl:
        images = images.to(device, non_blocking=True)
        if AMP:
            with torch.amp.autocast("cuda"):
                out = model(images)
        else:
            out = model(images)
        feats.append(out.float().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(feats), np.concatenate(labels)

FEATURES, EXTRACTION_LOG = {}, []
for key in BACKBONES:
    res = RES_BY_BACKBONE[key]
    interp = INTERP_BY_BACKBONE.get(key, transforms.InterpolationMode.BILINEAR)
    keys = {s: cache_key(key, s, res, ids) for s, ids in (("train", tr_ids), ("val", va_ids))}
    need_gpu = not all(cache.has(k) for k in keys.values())

    model = source = None
    if need_gpu:
        model, dim, source = build_frozen_backbone(key, res)
        frozen = sum(p.requires_grad for p in model.parameters())
        assert frozen == 0, f"{key}: {frozen} parametros entrenables - el backbone NO esta congelado"
        print(f"\n[{key}] cargado de {source} | dim={dim} | res={res} | parametros entrenables=0")

    per_split, wall = {}, 0.0
    for split, ids in (("train", tr_ids), ("val", va_ids)):
        t0 = time.time()
        entry = cache.get_or_extract(
            keys[split], lambda ids=ids, res=res, interp=interp: extract(model, ids, res, interp),
            meta={"backbone": key, "split": split, "resolution": res, "source": source},
        )
        took = time.time() - t0
        wall += took
        per_split[split] = entry
        print(f"  {split:<5} n={len(entry.features):<5} dim={entry.features.shape[1]:<5} "
              f"{'extraido' if need_gpu else 'cacheado'} en {took:.1f}s")

    FEATURES[key] = {"train": (per_split["train"].features, per_split["train"].labels),
                     "val":   (per_split["val"].features,   per_split["val"].labels)}
    EXTRACTION_LOG.append({
        "backbone": key, "resolution": res, "source": source,
        "dim": int(per_split["val"].features.shape[1]),
        "images": len(tr_ids) + len(va_ids),
        "wall_seconds": round(wall, 2),
        "images_per_second": round((len(tr_ids) + len(va_ids)) / wall, 1) if wall > 0 else None,
        "from_cache": not need_gpu,
        "labels_used_in_pretraining": BACKBONE_SPECS[key]["labels_used"],
    })
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\nfeatures listas para", list(FEATURES), "|", elapsed())'''

PROBE_CODE = r'''rows = []
budget_hit = False
for key in BACKBONES:
    x_tr, y_tr = FEATURES[key]["train"]
    x_va, y_va = FEATURES[key]["val"]
    for seed in SEEDS:
        regimes = build_regimes(y_tr, FRACTIONS, seed=seed, min_per_class=2)
        for frac, idx in regimes.items():
            for probe in PROBES:
                if (time.time() - T_START) / 3600 > MAX_HOURS:
                    budget_hit = True
                    break
                t0 = time.time()
                try:
                    proba = run_probe(probe, x_tr[idx], y_tr[idx], x_va, N_CLASSES)
                    stats = summarize(y_va, proba, N_CLASSES)
                    error = None
                except Exception as exc:  # un probe caido no debe tumbar la rejilla
                    stats = {"macro_f1": float("nan"), "accuracy": float("nan"),
                             "auc": float("nan"), "ece": float("nan")}
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"[WARN] {key}/{probe}/{frac:.0%}/s{seed} fallo -> {error}")
                rows.append({"backbone": key, "supervision": "ssl" if key.startswith("ssl_") else "supervised",
                             "probe": probe, "fraction": frac, "seed": seed,
                             "n_labels": int(len(idx)), "seconds": round(time.time() - t0, 3),
                             "error": error, **stats})
if budget_hit:
    print(f"[GUARDA DE TIEMPO] {elapsed()} paso de {MAX_HOURS} h -> rejilla truncada. "
          f"{len(rows)} combinaciones completadas (corte explicito, no silencioso).")

assert rows, ("la rejilla quedo vacia: ninguna combinacion corrio. "
              "Revisa MAX_HOURS, BACKBONES y PROBES antes de seguir.")
grid = pd.DataFrame(rows)
print(f"\n{len(grid)} combinaciones | fallos: {int(grid.error.notna().sum())} | {elapsed()}")'''

REPORT_CODE = r'''ok = grid[grid.error.isna()]
agg = (ok.groupby(["backbone", "supervision", "probe", "fraction"])
         .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
              auc_mean=("auc", "mean"), auc_std=("auc", "std"),
              n_labels=("n_labels", "max"), seeds=("seed", "nunique"))
         .reset_index().sort_values(["probe", "fraction", "auc_mean"], ascending=[True, True, False]))

pd.set_option("display.width", 170)
pd.set_option("display.max_rows", 300)
# std con 1 semilla es NaN a proposito: no se puede estimar dispersion de una muestra.
agg["metrica_titular"] = agg.probe.map(headline_metric)
print("=== macro-F1 y AUC por backbone / probe / regimen "
      "(std NaN = una sola semilla) ===")
print(agg.round(4).to_string(index=False))

print("\n=== ADVERTENCIA DE LECTURA ===")
print(f"  Probes dominados por el prior de clase: {PRIOR_DOMINATED_PROBES}")
print("  En esos probes el argmax sigue la clase mayoritaria (nv es ~65 %), asi que su")
print("  macro-F1 NO se cita como resultado: se lee su AUC, que es por ranking.")
print(f"  Probes sin calibracion real (su ECE no es una afirmacion): {UNCALIBRATED_PROBES}")

# --- delta SSL vs mejor supervisado, la comparacion titular -----------------
deltas = []
for probe in sorted(ok.probe.unique()):
    for frac in sorted(ok.fraction.unique()):
        sel = agg[(agg.probe == probe) & (agg.fraction == frac)]
        ssl_rows = sel[sel.supervision == "ssl"]
        sup_rows = sel[sel.supervision == "supervised"]
        if ssl_rows.empty or sup_rows.empty:
            continue
        metric = f"{headline_metric(probe)}_mean"
        best_ssl = ssl_rows.loc[ssl_rows[metric].idxmax()]
        best_sup = sup_rows.loc[sup_rows[metric].idxmax()]
        deltas.append({"probe": probe, "fraction": frac, "metrica": headline_metric(probe),
                       "mejor_ssl": best_ssl.backbone, "valor_ssl": round(best_ssl[metric], 4),
                       "mejor_sup": best_sup.backbone, "valor_sup": round(best_sup[metric], 4),
                       "delta": round(best_ssl[metric] - best_sup[metric], 4)})
delta_df = pd.DataFrame(deltas)
if len(delta_df):
    print("\n=== SSL vs mejor supervisado (en la metrica titular de cada probe) ===")
    print(delta_df.to_string(index=False))
    wins = int((delta_df.delta > 0).sum())
    print(f"\n  el SSL gana en {wins}/{len(delta_df)} combinaciones probe x regimen")

# --- compuertas de aceptacion (logica en dermssl.gates, cubierta por tests) --
gate_list = evaluate_gates(
    backend=data.backend,
    ham_images=len(data.ham_images),
    lesion_overlap=len(overlap),
    rows_total=len(grid),
    rows_ok=len(ok),
    frozen_verified=True,          # el assert de la celda de extraccion ya lo garantizo
    backbones=BACKBONES,
    grid_rows=json.loads(ok.to_json(orient="records")),
    smoke=SMOKE,
)

# --- equidad por fototipo (Fase 2: fst_manifest.csv, generado en Kaggle por --
# torch_unet.py, Fase 3 - dermssl.fitzpatrick). Si el manifiesto todavia no
# existe (por ejemplo, antes de correr la Fase 3), este bloque no toca
# gate_list: la corrida se comporta exactamente como v5 (design.md, tabla
# "Error Handling": "Manifest missing -> ... v5 behavior unchanged").
FST_MANIFEST_PATH = WORK / "fst_manifest.csv"
if FST_MANIFEST_PATH.is_file():
    fst_manifest = pd.read_csv(FST_MANIFEST_PATH)
    fst_by_id = fst_manifest.set_index("image_id")["fst_bucket"]

    def fst_bucket_or_none(value):
        """Keep every validation row; map explicit/missing values to unavailable."""
        if pd.isna(value) or str(value).strip().lower() in ("", "unavailable", "nan"):
            return None
        try:
            bucket = int(value)
        except (TypeError, ValueError):
            return None
        return bucket if 1 <= bucket <= 6 else None

    ssl_key = next((k for k in BACKBONES if k.startswith("ssl_")), None)
    strat = {}
    if ssl_key is not None and "logreg_balanced" in PROBES and 1.0 in FRACTIONS:
        x_tr, y_tr = FEATURES[ssl_key]["train"]
        x_va, y_va = FEATURES[ssl_key]["val"]
        proba_va = run_probe("logreg_balanced", x_tr, y_tr, x_va, N_CLASSES)
        joined = [
            (label, proba, fst_bucket_or_none(fst_by_id.get(img_id, None)))
            for img_id, label, proba in zip(va_ids, y_va, proba_va)
        ]
        if joined:
            labels_j, proba_j, buckets_j = zip(*joined)
            strat = stratified_summary(
                np.array(labels_j), np.array(proba_j), np.array(buckets_j), N_CLASSES
            )
    gate_list.append(fairness_gate(strat))

print()
print(report(gate_list))

summary = {
    "mode": "SMOKE (numeros NO validos)" if SMOKE else "corrida real",
    "notebook": "proyecto_ssl_lesiones_v5",
    "claim": "ninguna etiqueta toca el extractor de features; backbone congelado",
    "elapsed": elapsed(),
    "data": {"backend": data.backend, "ham_images": len(data.ham_images),
             "pool_images": data.summary()["pool_images"], "train": len(tr_ids),
             "val": len(va_ids), "classes": classes, "lesion_overlap": int(len(overlap))},
    "backbones": {k: {**BACKBONE_SPECS[k], "resolution": RES_BY_BACKBONE[k]} for k in BACKBONES},
    "reporting_rules": {"prior_dominated_probes": list(PRIOR_DOMINATED_PROBES),
                        "uncalibrated_probes": list(UNCALIBRATED_PROBES)},
    "extraction": EXTRACTION_LOG,
    "grid": json.loads(grid.to_json(orient="records")),
    "aggregate": json.loads(agg.to_json(orient="records")),
    "ssl_vs_supervised": deltas,
    "gates": [g.as_dict() for g in gate_list],
    "all_gates_pass": bool(all_blocking_pass(gate_list)),
}
out_path = WORK / "RESULTS_SUMMARY_v5.json"
out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
grid.to_csv(WORK / "probe_grid_v5.csv", index=False)
print(f"\nguardado -> {out_path} y probe_grid_v5.csv | {elapsed()}")'''
