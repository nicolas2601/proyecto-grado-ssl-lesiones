"""Bootstrap pareado de la diferencia por imagen entre el estudiante final y los pisos."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path.home() / "work" / "proyectonp"
RNG = np.random.default_rng(42)
B = 10000


def load(path, field="dice"):
    out = {}
    for row in csv.DictReader(open(path)):
        try:
            v = float(row[field])
            if v == v:
                out[row["image_id"]] = v
        except (ValueError, KeyError):
            pass
    return out


def paired(a, b, label):
    ids = sorted(set(a) & set(b))
    x = np.array([a[i] for i in ids])
    y = np.array([b[i] for i in ids])
    d = x - y
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sign = "EXCLUYE 0" if (lo > 0 or hi < 0) else "incluye 0"
    print(f"{label:38s} n={len(ids):5d}  dif={d.mean():+.4f}  IC95=[{lo:+.4f}, {hi:+.4f}]  {sign}")


for metric in ["dice", "hd95"]:
    print(f"\n===== {metric} =====")
    for s in ["isic2018", "ham10000"]:
        est = load(ROOT / f"salidas/final/evaluacion_{s}/per_image.csv", metric)
        for m in ["rnorm", "xnorm"]:
            piso = load(ROOT / f"baselines_v2/evaluacion_{s}_{m}/per_image.csv", metric)
            paired(est, piso, f"{s}: estudiante - {m}")
