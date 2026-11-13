#!/usr/bin/env python3
"""Paired bootstrap between two test-time recipes built from per-view features.

A recipe is one or more components ``views_dir|subset|head.npz`` joined by ``+``; with several
components the prediction is the minimum-entropy one per image. Example:
  --base "../views|d4-8|../results/head-cell23.npz"
  --cand "../views|id|../results/head-views-id.npz+../views-c70|id|../results/head-views-id.npz"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import labels as L
from metrics import evaluate
from view_ensemble import combine
from views_eval import SUBSETS, probs, rows_for

SETS = ("milk_dev", "milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")
DRAWS = 2000


def predict(recipe: str, feats: Path, name: str):
    out, ref = [], None
    for component in recipe.split("+"):
        views_dir, subset, head = component.split("|")
        d, per_view = rows_for(Path(views_dir), feats, name)
        x = per_view[:, SUBSETS[subset]].mean(1)
        out.append(probs(np.load(head), x / np.linalg.norm(x, axis=1, keepdims=True)))
        ref = d if ref is None else ref
    return ref, out[0] if len(out) == 1 else combine(out, "minent")


def bacc(y, p):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--cand", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    report = {"base": args.base, "cand": args.cand, "sets": {}}
    for name in SETS:
        d, pa = predict(args.base, args.feats, name)
        _, pb = predict(args.cand, args.feats, name)
        y = np.asarray([L.HAM7.index(L.U8_TO_HAM7[u]) for u in d["y"]])
        diffs = []
        for _ in range(DRAWS):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == len(np.unique(y)):
                diffs.append(bacc(y[i], pb[i]) - bacc(y[i], pa[i]))
        ra, rb = evaluate(d["y"], pa), evaluate(d["y"], pb)
        report["sets"][name] = {
            "n": int(len(y)), "bacc": [bacc(y, pa), bacc(y, pb)], "diff": bacc(y, pb) - bacc(y, pa),
            "ci95": np.percentile(diffs, [2.5, 97.5]).tolist(), "malignant_auc": [ra["malignant_auc"], rb["malignant_auc"]],
            "nv_recall": [ra["recall"].get("nv"), rb["recall"].get("nv")],
            "mel_recall": [ra["recall"].get("mel"), rb["recall"].get("mel")],
            "agree": float((pa.argmax(1) == pb.argmax(1)).mean())}
        r = report["sets"][name]
        print(f"{name:12s} n={r['n']:5d} bacc {r['bacc'][0]:.3f}->{r['bacc'][1]:.3f} diff {r['diff']:+.3f} "
              f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] malAUC {r['malignant_auc'][0] or 0:.3f}->"
              f"{r['malignant_auc'][1] or 0:.3f} mel {r['mel_recall'][0] or 0:.2f}->{r['mel_recall'][1] or 0:.2f} "
              f"nv {r['nv_recall'][0] or 0:.2f}->{r['nv_recall'][1] or 0:.2f} agree {r['agree']:.2f}", flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
