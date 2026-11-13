"""Modality-centering probe: x' = x - mean_mod(train) + mean(train), modality from metadata (oracle)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/jovyan/work/nevus-fix-20261113/scripts")
import nevus_study as S
from metrics import evaluate

feats = Path("/home/jovyan/work/nevus-fix-20261113/feats")
train = S.concat([S.load(feats, n) for n in S.TRAIN])
sets = {n: S.load(feats, n) for n in S.REPORT if (feats / f"{'milk' if n.startswith('milk_') else n}.npz").exists()}
mu = train["x"].mean(0)
mus = {m: train["x"][train["mod"] == m].mean(0) for m in ("clin", "derm")}
def center(d):
    x = d["x"].copy()
    for m in ("clin", "derm"):
        x[d["mod"] == m] += mu - mus[m]
    return x
for spec in ("cell:2:3", "cell:1:10"):
    scheme, boost, C = spec.split(":")
    t = dict(train); t["x"] = center(train)
    model = S.fit(t, scheme, float(boost), float(C))
    out = {n: round(evaluate(d["y"], model.predict_proba(center(d)))["balanced_accuracy"], 3) for n, d in sets.items()}
    print(spec, "centered", json.dumps(out), flush=True)
