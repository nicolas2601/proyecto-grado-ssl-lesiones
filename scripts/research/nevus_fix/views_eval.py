#!/usr/bin/env python3
"""Cheaper test-time recipes: shipped HAM7 head on the mean of a subset of D4 views vs all eight.

Reads per-view CLS4 from extract_views.py. Reports balanced accuracy, malignant AUC, argmax agreement
and max probability change against the shipped 8-view recipe. Compute cost scales with the number of
views (one ViT pass each).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import labels as L
import nevus_study as S
from metrics import evaluate

SUBSETS = {"d4-8": (0, 1, 2, 3, 4, 5, 6, 7), "id+hflip+rot180+rot180flip": (0, 1, 4, 5), "rot4": (0, 2, 4, 6),
           "id+hflip": (0, 1), "id+rot180": (0, 4), "id": (0,)}
SETS = ("milk_dev", "milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")


def rows_for(views_dir: Path, feats_dir: Path, name: str):
    """Per-view array aligned with the rows S.load keeps (labels, modality, group)."""
    d = S.load(feats_dir, name)
    source = "milk" if name.startswith("milk_") else name
    shipped_ids = np.load(feats_dir / f"{source}.npz")["image_ids"]
    v = np.load(views_dir / f"{source}.npz")
    if not (v["image_ids"] == shipped_ids).all():
        raise SystemExit(f"{name}: view ids differ from shipped feature ids")
    index = {img: k for k, img in enumerate(shipped_ids.tolist())}
    pos = [index[i] for i in _kept_ids(feats_dir, name)]
    return d, v["views"][pos].astype(np.float32)


def _kept_ids(feats_dir, name):
    source = "milk" if name.startswith("milk_") else name
    ids = np.load(feats_dir / f"{source}.npz")["image_ids"].tolist()
    table = L.SOURCES[source]()
    kept = [i for i in ids if i in table]
    if source == "milk":
        kept = [i for i in kept if S.split_of(table[i][2]) == name]
    return kept


def probs(head, x):
    z = x @ head["weight"].T + head["bias"]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    head = np.load(args.head)
    report = {}
    for name in SETS:
        d, per_view = rows_for(args.views, args.feats, name)
        ref = None
        report[name] = {}
        for tag, idx in SUBSETS.items():
            x = per_view[:, idx].mean(1)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            p = probs(head, x)
            ref = p if ref is None else ref
            r = evaluate(d["y"], p)
            out = {"views": len(idx), "bacc": r["balanced_accuracy"], "malignant_auc": r["malignant_auc"],
                   "nv_recall": r["recall"].get("nv"), "mel_recall": r["recall"].get("mel"),
                   "agree_vs_d4": float((p.argmax(1) == ref.argmax(1)).mean()),
                   "max_dprob_vs_d4": float(np.abs(p - ref).max())}
            mask = d["mod"] == "clin"
            if mask.sum() >= 20 and (~mask).sum() >= 20:
                out["bacc_clin"] = evaluate(d["y"][mask], p[mask])["balanced_accuracy"]
                out["bacc_derm"] = evaluate(d["y"][~mask], p[~mask])["balanced_accuracy"]
            report[name][tag] = out
    dev = report["milk_dev"]
    for tag in SUBSETS:
        dev[tag]["dev_score"] = dev[tag]["bacc_clin"] + 0.5 * dev[tag]["bacc_derm"]
        print(json.dumps({"subset": tag, "views": len(SUBSETS[tag]), "dev_score": round(dev[tag]["dev_score"], 4),
                          "bacc": {n: round(report[n][tag]["bacc"], 3) for n in SETS},
                          "agree": {n: round(report[n][tag]["agree_vs_d4"], 3) for n in SETS}}), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
