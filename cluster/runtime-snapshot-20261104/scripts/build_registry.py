import json
HAMNOTE = "HAM: CSV re-hasheado 06aee5, equivalencia verificada por reproducción exacta del baseline; digest histórico d5045e no recuperado"
def D(p, lo, hi): return (p, lo, hi)
rows = []
def add(name, internal, typ, pad, ham, dpad, dham, gpu, seeds, status, section):
    rows.append({"name": name, "internal": internal, "type": typ, "section": section, "pad_macro_f1": pad,
                 "ham_macro_f1": ham, "delta_pad": dpad, "delta_ham": dham, "gpu_h": gpu, "seeds": seeds, "status": status})
S = "SSL puro"; C = "clásico"; SUP = "supervisado"
add("SimSiam sobre DINOv2 (base)", "baseline-current-recipe", "SSL puro", 0.6063, 0.5140, None, None, None, "42 (CV 3)", "referencia", S)
add("Tasa de aprendizaje ×4", "learning-rate-x4", "SSL puro", 0.6090, 0.5183, D(0.0026, -0.0108, 0.0159), D(0.0043, -0.0076, 0.0171), None, "42", "v1 terminado", S)
add("Menos repeticiones de fono", "fewer-phone-repeats", "SSL puro", 0.5975, 0.5108, D(-0.0089, -0.0195, 0.0008), D(-0.0032, -0.0152, 0.0072), None, "42", "v1 terminado", S)
add("Cuota igual sin SkinDisNet", "same-quota-without-skindisnet", "SSL puro", 0.5937, 0.5191, D(-0.0126, -0.0251, -0.0019), D(0.0051, -0.0060, 0.0155), None, "42", "v1 terminado", S)
add("Fono + SkinDisNet", "plus-skindisnet-phone", "SSL puro", 0.5940, 0.5050, D(-0.0123, -0.0258, -0.0010), D(-0.0090, -0.0211, 0.0026), None, "42", "v1 terminado", S)
add("SimSiam explora-style (LoRA)", "simsiam-explora-style", "SSL puro", 0.6010, 0.5186, D(-0.0053, -0.0122, 0.0009), D(0.0046, -0.0080, 0.0159), None, "42", "v1 terminado", S)
add("SimSiam predictor constante, lr", "simsiam-constant-predictor-lr", "SSL puro", 0.6007, 0.5144, D(-0.0057, -0.0152, 0.0033), D(0.0004, -0.0074, 0.0093), None, "42", "v1 terminado", S)
add("Dosis larga 12000", "long-dose-baseline-12000", "SSL puro", 0.6023, 0.5169, D(-0.0040, -0.0164, 0.0061), D(0.0029, -0.0094, 0.0142), None, "42", "v1 terminado", S)
add("Dosis larga 6000", "long-dose-baseline-12000-step6000", "SSL puro", 0.6096, 0.5113, D(0.0033, -0.0055, 0.0122), D(-0.0027, -0.0151, 0.0086), None, "42", "v1 terminado", S)
add("DINOv2 público (comparador)", "public-dinov2", "SSL puro (público)", 0.5840, 0.5049, D(-0.0224, -0.0369, -0.0105), D(-0.0091, -0.0267, 0.0078), None, "42", "v1 comparador", S)
add("SimSiam + distilación con predictor (memoria corregida)", "simsiam-distill-with-predictor-memoryfix", "SSL puro", 0.5740, 0.4620, D(-0.0323, -0.0617, -0.0062), D(-0.0520, -0.0802, -0.0220), 1.1355, "42", "v2 terminado", S)
add("Recorte aleatorio escala 0,4", "crop-scale-0.4", "SSL puro", 0.5911, 0.5068, D(-0.0152, -0.0277, -0.0043), D(-0.0072, -0.0232, 0.0052), 1.1328, "42", "v2 terminado", S)
add("SimSiam sobre DINOv2, imagen 336", "one-factor-res336", "SSL puro", 0.5979, 0.5127, D(-0.0050, -0.0240, 0.0100), D(-0.0014, -0.0183, 0.0111), 2.2, "42", "v3 terminado (GPU-h estimada)", S)
add("DINO+iBOT continuado", "dino-ibot-ema-continued", "SSL puro", 0.6100, 0.5158, D(0.0090, -0.0230, 0.0420), D(0.0331, 0.0041, 0.0660), 2.3, "42", "v3 terminado (GPU-h estimada)", S)
add("Clásico LBP+GLCM+color, lector balanceado", "classic-primary-balanced-logistic", "clásico", 0.3078, None, D(-0.2950, -0.3390, -0.2530), None, None, "42", "v3 terminado; intento previo 0,257 descartado", C)
add("Clásico LBP+GLCM+color, RBF-SVM balanceado", "classic-secondary-rbf-svm", "clásico", 0.2887, None, D(-0.3290, -0.3750, -0.2860), None, None, "42", "v3 terminado", C)
add("MLP balanceado sobre SimSiam (supervisado)", "supervised-mlp-balanced", "supervisado", 0.5871, 0.4838, D(-0.0068, -0.0398, 0.0295), D(-0.0305, -0.0706, 0.0154), None, "42", "v3 terminado; pista supervisada", SUP)
add("Ensemble LR + MLP sobre SimSiam (supervisado)", "supervised-ensemble-lr-mlp", "supervisado", 0.5965, 0.5042, D(0.0005, -0.0284, 0.0295), D(-0.0084, -0.0456, 0.0325), None, "42", "v3 terminado; pista supervisada", SUP)
notes = ["Una semilla SSL; bootstrap condicional; sin validación clínica.", HAMNOTE,
         "Δ: IC95 pareado por paciente (500 bootstraps) sobre OOF de la semilla 20261025 salvo v1 (night report) y v2 (reactivación, IC del reporte).",
         "GPU-h v3 son estimaciones, no del ledger exacto. Excluidos por no terminados: freeze-4, OOB, BYOL, Barlow, VICReg."]
json.dump({"rows": rows, "notes": notes}, open("results/campaign-20261027-v3/registro-brazos-parcial.json", "w"), indent=2, ensure_ascii=False)
print("rows", len(rows))
