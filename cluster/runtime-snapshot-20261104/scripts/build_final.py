import json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from evaluate_overlay_readout import load_labels
R = Path("results")
B = {"PAD": "baseline-current-recipe-seed42-evaluation-v1/balanced-linear-reader-pad-adapt.json",
     "HAM": "baseline-current-recipe-seed42-evaluation-v1/balanced-linear-reader-ham-dev.json"}
META = {"PAD": ("/home/jovyan/work/pad_ufes_20/metadata.csv", "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"),
        "HAM": ("/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv", "06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5")}
def stats(kind, arm_json):
    base = json.load(open(R / B[kind])); new = json.load(open(R / arm_json))
    cls = base["classes"]; C = np.array(cls); g = np.asarray(base["groups"])
    y = load_labels(kind, [{"image_id": i} for i in base["image_ids"]], Path(META[kind][0]), META[kind][1])
    mf = lambda yt, p: f1_score(yt, C[np.argmax(p, 1)], labels=cls, average="macro", zero_division=0)
    ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    rng = np.random.default_rng(0); S = [np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]) for _ in range(500)]
    nm = [mf(y, np.array(r["oof_probabilities"])) for r in new["runs"]]
    bm = [mf(y, np.array(r["oof_probabilities"])) for r in base["runs"]]
    pn = np.array(new["runs"][0]["oof_probabilities"]); pb = np.array(base["runs"][0]["oof_probabilities"])
    ds = [mf(y[s], pn[s]) - mf(y[s], pb[s]) for s in S]
    return {"mean": float(np.mean(nm)), "d_mean": float(np.mean(nm) - np.mean(bm)), "d_seed0": float(nm[0] - bm[0]),
            "ic": (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))}
ARMS = [
 ("SimSiam sobre DINOv2 (base)", "baseline-current-recipe", "SSL puro", None, None, 1.25, "referencia", None),
 ("BYOL continuado (EMA)", "byol", "SSL puro", "byol-evaluation-v1/PAD-readout.json", "byol-evaluation-v1/HAM-readout-06aee5.json", 1.39, "v3 terminado", None),
 ("Barlow Twins (semilla SSL 42)", "barlow", "SSL puro", "barlow-evaluation-v1/PAD-readout.json", "barlow-evaluation-v1/HAM-readout-06aee5.json", 1.31, "v3 terminado", None),
 ("SimSiam bloques congelados 4", "freeze4", "SSL puro", "freeze4-evaluation-v1/PAD-readout.json", "freeze4-evaluation-v1/HAM-readout-06aee5.json", None, "v3 terminado", None),
 ("SimSiam + OOB", "oob", "SSL puro", "oob-evaluation-v1/PAD-readout.json", "oob-evaluation-v1/HAM-readout-06aee5.json", None, "v3 terminado", None),
 ("DINOv2-small congelado (público)", "dinov2-small", "SSL puro", "public-small-giant-v1/dinov2-small/PAD-readout.json", "public-small-giant-v1/dinov2-small/HAM-readout-06aee5.json", None, "público, Apache-2.0", None),
 ("DINOv2-giant congelado (público)", "dinov2-giant", "SSL puro", "public-small-giant-v1/dinov2-giant/PAD-readout.json", "public-small-giant-v1/dinov2-giant/HAM-readout-06aee5.json", None, "público, Apache-2.0", None),
 ("OpenCLIP ViT-B/16 (imagen-texto, solo imagen)", "openclip-b16", "imagen-texto y +metadatos", "openclip-b16-evaluation-v1/PAD-readout.json", "openclip-b16-evaluation-v1/HAM-readout-06aee5.json", None, "público MIT, safetensors", None),
 ("OpenCLIP ViT-L/14 (imagen-texto, solo imagen)", "openclip-l14", "imagen-texto y +metadatos", "openclip-l14-evaluation-v1/PAD-readout.json", "openclip-l14-evaluation-v1/HAM-readout-06aee5.json", None, "público MIT, safetensors", None),
]
rows = []
for name, internal, section, pad_json, ham_json, gpu, status, _ in ARMS:
    if pad_json is None:
        rows.append({"name": name, "internal": internal, "type": section, "section": section, "pad_macro_f1": 0.6063, "ham_macro_f1": 0.5140,
                     "delta_pad": None, "delta_ham": None, "gpu_h": gpu, "seeds": "3★", "status": status}); continue
    p = stats("PAD", pad_json); h = stats("HAM", ham_json)
    rows.append({"name": name, "internal": internal, "type": section, "section": section,
                 "pad_macro_f1": round(p["mean"], 4), "ham_macro_f1": round(h["mean"], 4),
                 "delta_pad": (round(p["d_mean"], 4), round(p["ic"][0], 4), round(p["ic"][1], 4)),
                 "delta_ham": (round(h["d_mean"], 4), round(h["ic"][0], 4), round(h["ic"][1], 4)),
                 "gpu_h": gpu, "seeds": "3★", "status": f"{status}; seed 20261025: ΔPAD {p['d_seed0']:+.3f}, ΔHAM {h['d_seed0']:+.3f}"})
out = Path("results/campaign-20261027-v3/registro-final.json")
notes = ["Convención de Δ: media de 3 semillas CV (20261025/26/27) con IC95 pareado por paciente (500) sobre la semilla 20261025 (IC de la semilla; el punto es la media de 3).",
         "Semillas: '3★' = 3 semillas de CV del lector (mismo encoder). Semillas SSL: 1 por brazo salvo réplicas de Barlow (43/44 en curso).",
         "HAM: CSV re-hasheado 06aee5, equivalencia verificada por reproducción exacta del baseline; digest histórico d5045e no recuperado.",
         "GPU-h v3 son estimaciones salvo los de allocated_gpu_hours. Sin tabla: MULAN (hook pendiente), BiomedCLIP (saltado: solo pickle .bin), ViT-L Barlow (no construido)."]
out.write_text(json.dumps({"rows": rows, "notes": notes}, indent=2, ensure_ascii=False))
for r in rows: print(r["name"], r["pad_macro_f1"], r["delta_pad"], r["ham_macro_f1"], r["delta_ham"])
