#!/usr/bin/env python3
"""Refit the v10.4 HAM7 reader (cell:2:3) on features built from a subset of D4 views.

A cheaper test-time recipe needs a reader fit on the same recipe. Selection score on milk_dev
(clinical bacc + 0.5 * dermoscopic bacc); every other set is report-only.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import nevus_study as S
from metrics import evaluate
from views_eval import SUBSETS, rows_for

SPEC = ("cell", 2.0, 3.0)
REPORT = ("milk_dev", "milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")


def subset_features(views_dir, feats_dir, name, idx):
    d, per_view = rows_for(views_dir, feats_dir, name)
    x = per_view[:, idx].mean(1)
    d["x"] = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return d


def run(job):
    tag, views_dir, feats_dir, out_dir = job
    idx = SUBSETS[tag]
    train = S.concat([subset_features(views_dir, feats_dir, n, idx) for n in S.TRAIN])
    model = S.fit(train, *SPEC)
    result = {}
    for name in REPORT:
        d = subset_features(views_dir, feats_dir, name, idx)
        p = model.predict_proba(d["x"])
        result[name] = {"all": evaluate(d["y"], p)}
        mask = d["mod"] == "clin"
        if mask.sum() >= 20 and (~mask).sum() >= 20:
            result[name]["clin"] = evaluate(d["y"][mask], p[mask])
            result[name]["derm"] = evaluate(d["y"][~mask], p[~mask])
    dev = result["milk_dev"]
    score = dev["clin"]["balanced_accuracy"] + 0.5 * dev["derm"]["balanced_accuracy"]
    np.savez(out_dir / f"head-views-{tag}.npz", weight=model.coef_.astype(np.float32),
             bias=model.intercept_.astype(np.float32), classes=np.asarray(model.classes_), views=np.asarray(idx))
    return tag, score, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subsets", default="id,id+hflip,rot4,d4-8")
    args = parser.parse_args()
    jobs = [(t, args.views, args.feats, args.out_dir) for t in args.subsets.split(",")]
    report = {}
    with ProcessPoolExecutor(len(jobs)) as pool:
        for tag, score, result in pool.map(run, jobs):
            report[tag] = {"views": len(SUBSETS[tag]), "dev_score": score, "sets": result}
            print(json.dumps({"subset": tag, "views": len(SUBSETS[tag]), "dev_score": round(score, 4),
                              "bacc": {n: round(r["all"]["balanced_accuracy"], 3) for n, r in result.items()},
                              "nv": {n: round(r["all"]["recall"].get("nv", float("nan")), 2) for n, r in result.items()},
                              "mel": {n: round(r["all"]["recall"].get("mel", float("nan")), 2)
                                      for n, r in result.items()}}), flush=True)
    (args.out_dir / "views-refit.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
