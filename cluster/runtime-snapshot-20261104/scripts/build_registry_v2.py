import json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from evaluate_overlay_readout import load_labels
R = Path("results")
night = json.load(open(R/"ssl-night-comparison-20261026-v1/final-night-report.json"))
comp = {c["representation"]: c for c in night["separate_public_encoder_comparators"]}
def night_delta(d):
    return None if d is None else (d["point"], d["ci95"][0], d["ci95"][1])
base = json.load(open(R/"baseline-current-recipe-seed42-evaluation-v1/balanced-linear-reader-pad-adapt.json"))
cls = base["classes"]; C = np.array(cls); groups = np.asarray(base["groups"])
rows = [{"image_id": i} for i in base["image_ids"]]
y = load_labels("PAD", rows, Path("/home/jovyan/work/pad_ufes_20/metadata.csv"), "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527")
CANCER = {"BCC", "MEL", "SCC"}
yb = np.array([1 if c in CANCER else 0 for c in y])
def auc(p, lab=None):
    idx = [i for i, c in enumerate(cls) if c in CANCER]
    lab = yb if lab is None else lab
    return roc_auc_score(lab, p[:, idx].sum(1))
pb = np.array(base["runs"][0]["oof_probabilities"])
ug = np.unique(groups); idxg = {g: np.where(groups == g)[0] for g in ug}
rng = np.random.default_rng(0); S = [np.concatenate([idxg[g] for g in rng.choice(ug, len(ug))]) for _ in range(500)]
auc_base = auc(pb)
new = json.load(open(R/"dino-ibot-evaluation-v1/PAD-readout.json"))
pn = np.array(new["runs"][0]["oof_probabilities"])
dauc = [auc(pn[s], yb[s]) - auc(pb[s], yb[s]) for s in S]
rows_out = []
def add(name, internal, typ, pad, ham, dpad, dham, gpu, seeds, status, section):
    rows_out.append({"name": name, "internal": internal, "type": typ, "section": section, "pad_macro_f1": pad,
                     "ham_macro_f1": ham, "delta_pad": dpad, "delta_ham": dham, "gpu_h": gpu, "seeds": seeds, "status": status})
S_SSL, S_SUP, S_IT, S_CL, S_SEMI, S_NF = "SSL puro", "supervisado", "imagen-texto y +metadatos", "clásico", "semi-supervisado", "no factible"
add("SimSiam sobre DINOv2 (base)", "baseline-current-recipe", "SSL puro", 0.6063, 0.5140, None, None, None, "42 (CV 3)", "referencia", S_SSL)
add("Tasa de aprendizaje ×4", "learning-rate-x4", "SSL puro", 0.6090, 0.5183, (0.0026, -0.0108, 0.0159), (0.0043, -0.0076, 0.0171), None, "42", "v1 terminado", S_SSL)
add("Menos repeticiones de celular", "fewer-phone-repeats", "SSL puro", 0.5975, 0.5108, (-0.0089, -0.0195, 0.0008), (-0.0032, -0.0152, 0.0072), None, "42", "v1 terminado", S_SSL)
add("Misma cuota de datos sin SkinDisNet", "same-quota-without-skindisnet", "SSL puro", 0.5937, 0.5191, (-0.0126, -0.0251, -0.0019), (0.0051, -0.0060, 0.0155), None, "42", "v1 terminado", S_SSL)
add("Celular + SkinDisNet", "plus-skindisnet-phone", "SSL puro", 0.5940, 0.5050, (-0.0123, -0.0258, -0.0010), (-0.0090, -0.0211, 0.0026), None, "42", "v1 terminado", S_SSL)
add("Estilo ExPLoRA (LoRA r=64)", "simsiam-explora-style", "SSL puro", 0.6010, 0.5186, (-0.0053, -0.0122, 0.0009), (0.0046, -0.0080, 0.0159), None, "42", "v1 terminado", S_SSL)
add("SimSiam con predictor constante (tasa de aprendizaje)", "simsiam-constant-predictor-lr", "SSL puro", 0.6007, 0.5144, (-0.0057, -0.0152, 0.0033), (0.0004, -0.0074, 0.0093), None, "42", "v1 terminado", S_SSL)
add("Entrenamiento largo 12 000 pasos", "long-dose-baseline-12000", "SSL puro", 0.6023, 0.5169, (-0.0040, -0.0164, 0.0061), (0.0029, -0.0094, 0.0142), None, "42", "v1 terminado", S_SSL)
add("Entrenamiento largo 6000 pasos", "long-dose-baseline-12000-step6000", "SSL puro", 0.6096, 0.5113, (0.0033, -0.0055, 0.0122), (-0.0027, -0.0151, 0.0086), None, "42", "v1 terminado", S_SSL)
add("DINOv2 público congelado (comparador)", "public-dinov2", "SSL puro (público)", 0.5840, 0.5049, (-0.0224, -0.0369, -0.0105), (-0.0091, -0.0267, 0.0078), None, "42", "v1 comparador", S_SSL)
add("DINOv2-large congelado (público, solo imagen)", "dinov2-large", "SSL puro (público)", 0.5922, None, night_delta(comp["dinov2-large"]["image_delta_vs_adapted_SSL"]) and (comp["dinov2-large"]["image_delta_vs_adapted_SSL"]["point"], comp["dinov2-large"]["image_delta_vs_adapted_SSL"]["ci95"][0], comp["dinov2-large"]["image_delta_vs_adapted_SSL"]["ci95"][1]), None, None, "42 (CV 3)", "v1 comparador", S_SSL)
add("Distilación con predictor (memoria corregida)", "simsiam-distill-with-predictor-memoryfix", "SSL puro", 0.5740, 0.4620, (-0.0323, -0.0617, -0.0062), (-0.0520, -0.0802, -0.0220), 1.1355, "42", "v2 terminado", S_SSL)
add("Recorte aleatorio escala 0,4", "crop-scale-0.4", "SSL puro", 0.5911, 0.5068, (-0.0152, -0.0277, -0.0043), (-0.0072, -0.0232, 0.0052), 1.1328, "42", "v2 terminado", S_SSL)
add("SimSiam sobre DINOv2, imagen 336", "one-factor-res336", "SSL puro", 0.5979, 0.5127, (-0.0050, -0.0240, 0.0100), (-0.0014, -0.0183, 0.0111), 2.2, "42", "v3 terminado (GPU-h estimada)", S_SSL)
add("DINO+iBOT continuado (desvío: pos_embed y cls_token congelados)", "dino-ibot-ema-continued", "SSL puro", 0.6100, 0.5158, (0.0090, -0.0230, 0.0420), (0.0331, 0.0041, 0.0660), 2.3, "42", "v3 terminado (GPU-h estimada)", S_SSL)
add("AUC cáncer vs no cáncer (SimSiam base)", "auc-cancer-baseline", "métrica binaria", round(auc_base, 4), None, None, None, None, "42 (OOF)", "referencia de AUC", S_SSL)
add("AUC cáncer vs no cáncer (DINO+iBOT)", "auc-cancer-dino-ibot", "métrica binaria", round(auc(pn), 4), None, (round(auc(pn) - auc_base, 4), float(np.percentile(dauc, 2.5)), float(np.percentile(dauc, 97.5))), None, None, "42 (OOF)", "ΔAUC pareado por paciente", S_SSL)
for name, key in [("SigLIP2 so400m (imagen-texto, solo imagen)", "siglip2-so400m"), ("SigLIP2 base (imagen-texto, solo imagen)", "siglip2-base"),
                  ("DINOv2-large + SigLIP2 (solo imagen)", "dinov2-large-plus-siglip2"), ("SimSiam adaptado + SigLIP2 (solo imagen)", "adapted-simsiam-plus-siglip2")]:
    c = comp[key]; io = c["image_only"]; d = c.get("image_delta_vs_adapted_SSL")
    add(name, key, "imagen-texto", round(io["macro_f1"], 4), None, (d["point"], d["ci95"][0], d["ci95"][1]) if d else None, None, None, "42 (CV 3)", "v1 comparador (no SSL puro)", S_IT)
for name, key in [("SigLIP2 so400m + metadatos", "siglip2-so400m"), ("SigLIP2 base + metadatos", "siglip2-base"),
                  ("DINOv2-large + metadatos", "dinov2-large"), ("DINOv2-large + SigLIP2 + metadatos", "dinov2-large-plus-siglip2")]:
    c = comp[key]; im = c.get("image_plus_metadata")
    if not im: continue
    d = c.get("metadata_delta_vs_adapted_SSL_metadata")
    add(name, key + "-metadata", "imagen-texto + metadatos" if "siglip" in key else "SSL + metadatos", round(im["macro_f1"], 4), None,
        (d["point"], d["ci95"][0], d["ci95"][1]) if d else None, None, None, "42 (CV 3)", "v1 comparador (+metadatos)", S_IT)
add("Clásico LBP+GLCM+color, lector balanceado", "classic-primary-balanced-logistic", "clásico", 0.3078, None, (-0.2950, -0.3390, -0.2530), None, None, "42 (CV 3)", "v3 terminado; intento previo 0,257 descartado (lector mal calibrado)", S_CL)
add("Clásico LBP+GLCM+color, RBF-SVM balanceado", "classic-secondary-rbf-svm", "clásico", 0.2887, None, (-0.3290, -0.3750, -0.2860), None, None, "42 (CV 3)", "v3 terminado", S_CL)
add("MLP balanceado sobre SimSiam", "supervised-mlp-balanced", "supervisado", 0.5871, 0.4838, (-0.0068, -0.0398, 0.0295), (-0.0305, -0.0706, 0.0154), None, "42 (CV 3)", "v3 terminado; supervisado, no SSL", S_SUP)
add("Ensemble LR + MLP sobre SimSiam", "supervised-ensemble-lr-mlp", "supervisado", 0.5965, 0.5042, (0.0005, -0.0284, 0.0295), (-0.0084, -0.0456, 0.0325), None, "42 (CV 3)", "v3 terminado; supervisado, no SSL", S_SUP)
add("FixMatch / Mean Teacher (semi-supervisado)", "semi-supervised-pending", "semi-supervisado", None, None, None, None, None, "—", "no ejecutado (pendiente)", S_SEMI)
for name, why in [("DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)", "repo gated manual; licencia 'other'; términos no aceptados"),
                  ("MedSigLIP (google/medsiglip-448)", "gated automático; términos de Google no aceptados"),
                  ("Derm Foundation (Google)", "gated; no verificado"),
                  ("DermLIP ViT-B/16 (redlessone)", "gated automático; CC-BY-NC-4.0 (no comercial)"),
                  ("MONET (suinleelab/monet)", "repo abierto sin licencia verificable (sin tag ni texto de licencia)"),
                  ("PanDerm", "pesos y licencia no verificados"),
                  ("ConvMAE, SparK, Context Encoders, RBM/DBN, GAN/BiGAN, difusión", "arquitectura incompatible con ViT-B/14 o costo prohibitivo"),
                  ("SLIP, CLIP-Lite, DeCLIP", "usan texto; fuera de imagen sola")]:
    add(name, "no-factible", "no factible", None, None, None, None, None, "—", "no factible: " + why, S_NF)
notes = ["Δ PAD/HAM: IC95 pareado por paciente (500). v1 y comparadores: IC del reporte nocturno. v2: IC de reactivación. v3: semilla 20261025.",
         "HAM: CSV re-hasheado 06aee5, equivalencia verificada por reproducción exacta del baseline; digest histórico d5045e no recuperado.",
         "GPU-h v3 son estimaciones, no del ledger exacto. Freeze-4, OOB, BYOL, Barlow y VICReg siguen en curso y no entran en esta tabla."]
json.dump({"rows": rows_out, "notes": notes}, open("results/campaign-20261027-v3/registro-brazos-parcial.json", "w"), indent=2, ensure_ascii=False)
print("rows", len(rows_out), "auc_base", round(auc_base, 4), "auc_new", round(auc(pn), 4))
