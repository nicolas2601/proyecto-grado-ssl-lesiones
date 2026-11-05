import json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from evaluate_overlay_readout import load_labels
R = Path("results")
night = json.load(open(R/"ssl-night-comparison-20261026-v1/final-night-report.json"))
comp = {c["representation"]: c for c in night["separate_public_encoder_comparators"]}
def d3(x):
    return None if x is None else (x["point"], x["ci95"][0], x["ci95"][1])
base = json.load(open(R/"baseline-current-recipe-seed42-evaluation-v1/balanced-linear-reader-pad-adapt.json"))
cls = base["classes"]; CANCER = {"BCC", "MEL", "SCC"}; cidx = [i for i, c in enumerate(cls) if c in CANCER]
g = np.asarray(base["groups"])
y = load_labels("PAD", [{"image_id": i} for i in base["image_ids"]], Path("/home/jovyan/work/pad_ufes_20/metadata.csv"), "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527")
yb = np.array([1 if c in CANCER else 0 for c in y])
def auc(p, lab=yb): return roc_auc_score(lab, p[:, cidx].sum(1))
base_auc = np.mean([auc(np.array(r["oof_probabilities"])) for r in base["runs"]])
dino = json.load(open(R/"dino-ibot-evaluation-v1/PAD-readout.json"))
dino_auc = np.mean([auc(np.array(r["oof_probabilities"])) for r in dino["runs"]])
pb = np.array(base["runs"][0]["oof_probabilities"]); pn = np.array(dino["runs"][0]["oof_probabilities"])
ug = np.unique(g); idxg = {k: np.where(g == k)[0] for k in ug}
rng = np.random.default_rng(0); S = [np.concatenate([idxg[k] for k in rng.choice(ug, len(ug))]) for _ in range(500)]
dd = [auc(pn[s], yb[s]) - auc(pb[s], yb[s]) for s in S]
def gpu(path):
    try:
        return float(json.load(open(path))["allocated_gpu_hours"])
    except Exception:
        return None
rows = []
def add(name, internal, typ, section, pad, ham, dpad, dham, gpu_h, seeds, status):
    rows.append({"name": name, "internal": internal, "type": typ, "section": section, "pad_macro_f1": pad,
                 "ham_macro_f1": ham, "delta_pad": dpad, "delta_ham": dham, "gpu_h": gpu_h, "seeds": seeds, "status": status})
S_SSL, S_SUP, S_IT, S_CL, S_SEMI, S_NF = "SSL puro", "supervisado", "imagen-texto y +metadatos", "clásico", "semi-supervisado", "no factible"
NOLEDGER = "no en ledger v1"
add("SimSiam sobre DINOv2 (base)", "baseline-current-recipe", "SSL puro", S_SSL, 0.6063, 0.5140, None, None, 1.25, "3★", "referencia")
add("Tasa de aprendizaje ×4", "learning-rate-x4", "SSL puro", S_SSL, 0.6090, 0.5183, (0.0026, -0.0108, 0.0159), (0.0043, -0.0076, 0.0171), 1.25, "3★", "v1 terminado")
add("Menos repeticiones de celular", "fewer-phone-repeats", "SSL puro", S_SSL, 0.5975, 0.5108, (-0.0089, -0.0195, 0.0008), (-0.0032, -0.0152, 0.0072), 1.03, "3★", "v1 terminado")
add("Misma cuota de datos sin SkinDisNet", "same-quota-without-skindisnet", "SSL puro", S_SSL, 0.5937, 0.5191, (-0.0126, -0.0251, -0.0019), (0.0051, -0.0060, 0.0155), 1.07, "3★", "v1 terminado")
add("Celular + SkinDisNet", "plus-skindisnet-phone", "SSL puro", S_SSL, 0.5940, 0.5050, (-0.0123, -0.0258, -0.0010), (-0.0090, -0.0211, 0.0026), None, "3★", "v1 terminado (GPU-h sin entrada en ledger)")
add("Estilo ExPLoRA (LoRA r=64)", "simsiam-explora-style", "SSL puro", S_SSL, 0.6010, 0.5186, (-0.0053, -0.0122, 0.0009), (0.0046, -0.0080, 0.0159), None, "3★", "v1 terminado")
add("SimSiam con predictor constante (tasa de aprendizaje)", "simsiam-constant-predictor-lr", "SSL puro", S_SSL, 0.6007, 0.5144, (-0.0057, -0.0152, 0.0033), (0.0004, -0.0074, 0.0093), None, "3★", "v1 terminado")
add("Entrenamiento largo 12 000 pasos", "long-dose-baseline-12000", "SSL puro", S_SSL, 0.6023, 0.5169, (-0.0040, -0.0164, 0.0061), (0.0029, -0.0094, 0.0142), None, "3★", "v1 terminado")
add("Entrenamiento largo 6000 pasos", "long-dose-baseline-12000-step6000", "SSL puro", S_SSL, 0.6096, 0.5113, (0.0033, -0.0055, 0.0122), (-0.0027, -0.0151, 0.0086), None, "3★", "v1 terminado")
add("DINOv2 público congelado (comparador)", "public-dinov2", "SSL puro (público)", S_SSL, 0.5840, 0.5049, (-0.0224, -0.0369, -0.0105), (-0.0091, -0.0267, 0.0078), None, "3★", "v1 comparador")
add("DINOv2-large congelado (público, solo imagen)", "dinov2-large", "SSL puro (público)", S_SSL, 0.5922, "no medido en HAM", d3(comp["dinov2-large"]["image_delta_vs_adapted_SSL"]), None, None, "3★", "v1 comparador")
add("Distilación con predictor (memoria corregida)", "simsiam-distill-with-predictor-memoryfix", "SSL puro", S_SSL, 0.5740, 0.4620, (-0.0323, -0.0617, -0.0062), (-0.0520, -0.0802, -0.0220), 1.14, "3★", "v2 terminado")
add("Recorte aleatorio escala 0,4", "crop-scale-0.4", "SSL puro", S_SSL, 0.5911, 0.5068, (-0.0152, -0.0277, -0.0043), (-0.0072, -0.0232, 0.0052), 1.13, "3★", "v2 terminado")
add("SimSiam sobre DINOv2, imagen 336", "one-factor-res336", "SSL puro", S_SSL, 0.5979, 0.5127, (-0.0050, -0.0240, 0.0100), (-0.0014, -0.0183, 0.0111), gpu("runs/one-factor-res336-seed42-run-v1/result.json"), "3★", "v3 terminado")
add("DINO+iBOT continuado (desvío: pos_embed y cls_token congelados)", "dino-ibot-ema-continued", "SSL puro", S_SSL, 0.6100, 0.5158, (0.0090, -0.0230, 0.0420), (0.0331, 0.0041, 0.0660), 2.30, "3★", "v3 terminado (GPU-h estimada)")
add("SimSiam sobre DINOv2, bloques congelados 4", "one-factor-freeze4", "SSL puro", S_SSL, 0.6016, 0.5190, (-0.0005, -0.0118, 0.0111), (0.0068, -0.0086, 0.0201), gpu("runs/one-factor-freeze4-seed42-run-v1/result.json"), "3★", "v3 terminado (Δ seed 20261025; media 3 semillas −0,005 ★)")
add("SimSiam + regularización fuera de batch (OOB)", "simsiam-oob", "SSL puro", S_SSL, 0.5896, 0.5313, (-0.0084, -0.0361, 0.0176), (0.0223, -0.0087, 0.0496), gpu("runs/oob-simsiam-seed42-run-v1/result.json"), "3★", "v3 terminado (Δ seed 20261025; media 3 semillas −0,017 ★)")
add("BYOL continuado (EMA)", "simsiam-byol", "SSL puro", S_SSL, None, None, None, None, None, "—", "en curso (GPU0)")
add("Barlow Twins (cabeza sobre DINOv2)", "simsiam-barlow", "SSL puro", S_SSL, None, None, None, None, None, "—", "en curso (GPU1)")
add("VICReg (cabeza sobre DINOv2)", "simsiam-vicreg", "SSL puro", S_SSL, None, None, None, None, None, "—", "en cola tras Barlow")
add("AUC cáncer vs no cáncer (SimSiam base)", "auc-cancer-baseline", "métrica binaria", S_SSL, round(base_auc, 3), None, None, None, None, "3★", "referencia de AUC (seed 20261025: 0,884)")
add("AUC cáncer vs no cáncer (DINO+iBOT)", "auc-cancer-dino-ibot", "métrica binaria", S_SSL, round(dino_auc, 3), None, (round(dino_auc - base_auc, 4), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))), None, None, "3★", "ΔAUC pareado (semilla 20261025)")
for name, key in [("SigLIP2 so400m (imagen-texto, solo imagen)", "siglip2-so400m"), ("SigLIP2 base (imagen-texto, solo imagen)", "siglip2-base"),
                  ("DINOv2-large + SigLIP2 (solo imagen)", "dinov2-large-plus-siglip2"), ("SimSiam adaptado + SigLIP2 (solo imagen)", "adapted-simsiam-plus-siglip2")]:
    c = comp[key]
    add(name, key, "imagen-texto", S_IT, round(c["image_only"]["macro_f1"], 4), "no medido en HAM", d3(c.get("image_delta_vs_adapted_SSL")), None, None, "3★", "v1 comparador (no SSL puro)")
for name, key in [("SigLIP2 so400m + metadatos", "siglip2-so400m"), ("SigLIP2 base + metadatos", "siglip2-base"),
                  ("DINOv2-large + metadatos", "dinov2-large"), ("DINOv2-large + SigLIP2 + metadatos", "dinov2-large-plus-siglip2")]:
    c = comp[key]
    add(name, key + "-metadata", "imagen-texto + metadatos" if "siglip" in key else "SSL + metadatos", S_IT, round(c["image_plus_metadata"]["macro_f1"], 4), "no medido en HAM", d3(c.get("metadata_delta_vs_adapted_SSL_metadata")), None, None, "3★", "v1 comparador (+metadatos)")
add("Clásico LBP+GLCM+color, lector balanceado", "classic-primary-balanced-logistic", "clásico", S_CL, 0.3078, "no medido en HAM", (-0.2950, -0.3390, -0.2530), None, None, "3★", "v3 terminado; intento previo 0,257 descartado")
add("Clásico LBP+GLCM+color, RBF-SVM balanceado", "classic-secondary-rbf-svm", "clásico", S_CL, 0.2887, "no medido en HAM", (-0.3290, -0.3750, -0.2860), None, None, "3★", "v3 terminado")
add("MLP balanceado sobre SimSiam", "supervised-mlp-balanced", "supervisado", S_SUP, 0.5871, 0.4838, (-0.0068, -0.0398, 0.0295), (-0.0305, -0.0706, 0.0154), None, "3★", "v3 terminado; supervisado, no SSL")
add("Ensemble LR + MLP sobre SimSiam", "supervised-ensemble-lr-mlp", "supervisado", S_SUP, 0.5965, 0.5042, (0.0005, -0.0284, 0.0295), (-0.0084, -0.0456, 0.0325), None, "3★", "v3 terminado; supervisado, no SSL")
add("FixMatch / Mean Teacher (semi-supervisado)", "semi-supervised-pending", "semi-supervisado", S_SEMI, None, None, None, None, None, "—", "no ejecutado (pendiente)")
for name, why in [("DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)", "repo gated manual; licencia 'other'; términos no aceptados"),
                  ("MedSigLIP (google/medsiglip-448)", "gated automático; términos de Google no aceptados"),
                  ("Derm Foundation (Google)", "gated; no verificado"),
                  ("DermLIP ViT-B/16 (redlessone)", "gated automático; CC-BY-NC-4.0 (no comercial)"),
                  ("MONET (suinleelab/monet)", "repo abierto sin licencia verificable"),
                  ("PanDerm", "pesos y licencia no verificados"),
                  ("ConvMAE, SparK, Context Encoders, RBM/DBN, GAN/BiGAN, difusión", "arquitectura incompatible con ViT-B/14 o costo"),
                  ("SLIP, CLIP-Lite, DeCLIP", "usan texto; fuera de imagen sola")]:
    add(name, "no-factible", "no factible", S_NF, None, None, None, None, None, "—", "no factible: " + why)
notes = ["★ = valor de macro-F1 o AUC es media de 3 semillas CV (20261025/26/27); el Δ e IC son pareados por paciente (500 bootstraps) sobre la semilla 20261025, salvo v1 (reporte nocturno) y v2 (reactivación).",
         "Δ de freeze-4 y OOB en media de 3 semillas (★): −0,005 y −0,017 en PAD; +0,005 y +0,017 en HAM.",
         "HAM: CSV re-hasheado 06aee5, equivalencia verificada por reproducción exacta del baseline; digest histórico d5045e no recuperado.",
         "GPU-h: v1 del ledger (total 11,41 h); v2 y v3 de allocated_gpu_hours de cada corrida; DINO+iBOT estimado (sin receipt de tiempo). Sin entrada en ledger: celular+SkinDisNet y brazos v1 restantes."]
json.dump({"rows": rows, "notes": notes}, open("results/campaign-20261027-v3/registro-brazos-v3.json", "w"), indent=2, ensure_ascii=False)
print("rows", len(rows), "base_auc", round(base_auc, 4), "dino_auc", round(dino_auc, 4))
