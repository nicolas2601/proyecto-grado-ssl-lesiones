#!/usr/bin/env python3
"""Nevus-collapse study: shipped heads vs multi-domain linear readers on the frozen SSL encoder.

Labels only enter the linear reader; the MoCo/DINOv2 encoder is frozen. MILK10k is split by lesion
(train/dev/test = 60/20/20 by SHA256 of lesion_id). Readers are selected on milk_dev only; every other
external set is report-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import labels as L
from metrics import evaluate, project_pad6

TRAIN = ("ham_fit", "ham_dev", "isic2019_noham", "pad_adapt", "milk_train")
REPORT = ("milk_dev", "milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")


def split_of(lesion: str) -> str:
    bucket = int(hashlib.sha256(lesion.encode()).hexdigest(), 16) % 10
    return "milk_train" if bucket < 6 else ("milk_dev" if bucket < 8 else "milk_test")


def load(feats: Path, name: str):
    source = "milk" if name.startswith("milk_") else name
    path = feats / f"{source}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    table = L.SOURCES[source]()
    keep = [i for i, img in enumerate(data["image_ids"].tolist()) if img in table]
    ids = [data["image_ids"][i] for i in keep]
    if source == "milk":
        keep = [i for i, img in zip(keep, ids) if split_of(table[img][2]) == name]
        ids = [data["image_ids"][i] for i in keep]
    rows = [table[i] for i in ids]
    return {"x": data["features"][keep], "ham7": data["ham7"][keep], "pad6": data["pad6"][keep],
            "y": np.asarray([r[0] for r in rows]), "mod": np.asarray([r[1] for r in rows]),
            "group": np.asarray([r[2] for r in rows]), "source": np.asarray([name] * len(rows))}


def weights(y7: np.ndarray, mod: np.ndarray, scheme: str, clin_boost: float) -> np.ndarray:
    """class: balanced over HAM7; cell: each (modality, class) cell gets equal mass."""
    w = np.ones(len(y7))
    if scheme == "class":
        for c in np.unique(y7):
            w[y7 == c] = len(y7) / (len(np.unique(y7)) * (y7 == c).sum())
    else:
        cells = sorted(set(zip(mod.tolist(), y7.tolist())))
        for m, c in cells:
            mask = (mod == m) & (y7 == c)
            w[mask] = len(y7) / (len(cells) * mask.sum())
    w[mod == "clin"] *= clin_boost
    return w * len(w) / w.sum()


def fit(train, scheme, clin_boost, C):
    y7 = np.asarray([L.U8_TO_HAM7[u] for u in train["y"]])
    model = LogisticRegression(C=C, max_iter=3000, tol=1e-6)
    model.fit(train["x"], y7, sample_weight=weights(y7, train["mod"], scheme, clin_boost))
    if tuple(model.classes_) != L.HAM7:
        raise ValueError("training pool lacks a HAM7 class")
    return model


def report_all(sets, probs_fn):
    out = {}
    for name, d in sets.items():
        out[name] = {"all": evaluate(d["y"], probs_fn(d))}
        for m in ("clin", "derm"):
            mask = d["mod"] == m
            if mask.sum() >= 20 and mask.sum() < len(mask):
                sub = {k: v[mask] for k, v in d.items()}
                out[name][m] = evaluate(sub["y"], probs_fn(sub))
    return out


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", default="class:1:1,cell:1:1,cell:2:1,cell:1:10,cell:2:10,cell:1:100")
    parser.add_argument("--train", default=",".join(TRAIN))
    parser.add_argument("--save-best", type=Path)
    parser.add_argument("--no-baselines", action="store_true", help="features from a new encoder have no heads")
    args = parser.parse_args()
    train_parts = [p for p in (load(args.feats, n) for n in args.train.split(",")) if p is not None]
    train = concat(train_parts)
    sets = {n: d for n in REPORT if (d := load(args.feats, n)) is not None}
    result = {"train_counts": {n: int((train["source"] == n).sum()) for n in np.unique(train["source"])},
              "baselines": {} if args.no_baselines else {
                  "shipped_HAM7": report_all(sets, lambda d: d["ham7"]),
                  "shipped_PAD6": report_all(sets, lambda d: project_pad6(d["pad6"]))},
              "readers": {}}
    models = {}
    for spec in args.grid.split(","):
        scheme, boost, C = spec.split(":")
        model = fit(train, scheme, float(boost), float(C))
        models[spec] = model
        result["readers"][spec] = report_all(sets, lambda d, m=model: m.predict_proba(d["x"]))
        brief = {n: round(r["all"]["balanced_accuracy"], 3) for n, r in result["readers"][spec].items()}
        print(json.dumps({"spec": spec, "bacc": brief}), flush=True)
    if "milk_dev" not in sets:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        return
    score = {s: r["milk_dev"]["clin"]["balanced_accuracy"] + 0.5 * r["milk_dev"]["derm"]["balanced_accuracy"]
             for s, r in result["readers"].items()}
    best = max(score, key=score.get)
    result["selection"] = {"rule": "max dev_clin_bacc + 0.5*dev_derm_bacc on milk_dev", "scores": score, "best": best}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    if args.save_best:
        m = models[best]
        np.savez(args.save_best, weight=m.coef_.astype(np.float32), bias=m.intercept_.astype(np.float32),
                 classes=np.asarray(m.classes_), spec=best)
    print(json.dumps({"best": best, "out": str(args.out)}))


if __name__ == "__main__":
    main()
