#!/usr/bin/env python3
"""Target-domain adaptation estimate: add labelled DDI and/or BOSQUE rows to the reader pool.

The added set is scored only on held-out folds (5 folds grouped by patient/identity), so its numbers
estimate what a reader trained with that data would do on new patients from the same source. This
removes the set as an untouched external test, so these numbers are reported separately from the
frozen external evaluation. Encoder frozen; labels only enter the linear reader.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import nevus_study as S
from metrics import evaluate

SPEC = ("cell", 2.0, 3.0)
SEED = 20261113


def take(d, idx):
    return {k: v[idx] for k, v in d.items()}


def fold_job(args):
    feats, targets, k, train_idx, held = args
    pool = [S.load(feats, n) for n in S.TRAIN]
    parts = [take(S.load(feats, t), train_idx[t]) for t in targets]
    model = S.fit(S.concat(pool + parts), *SPEC)
    return k, {t: model.predict_proba(S.load(feats, t)["x"][held[t]]) for t in targets}


def folds(d):
    order = np.random.default_rng(SEED).permutation(len(d["y"]))
    split = list(GroupKFold(5).split(order, d["y"][order], d["group"][order]))
    return [(order[a], order[b]) for a, b in split]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--targets", required=True, help="comma-separated: ddi,bosque")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets = args.targets.split(",")
    data = {t: S.load(args.feats, t) for t in targets}
    split = {t: folds(data[t]) for t in targets}
    jobs = [(args.feats, targets, k, {t: split[t][k][0] for t in targets}, {t: split[t][k][1] for t in targets})
            for k in range(5)]
    oof = {t: np.zeros((len(data[t]["y"]), 7)) for t in targets}
    with ProcessPoolExecutor(5) as pool:
        for k, probs in pool.map(fold_job, jobs):
            for t in targets:
                oof[t][split[t][k][1]] = probs[t]
    report = {"spec": ":".join(map(str, SPEC)), "targets": targets, "cv": {}}
    for t in targets:
        r = evaluate(data[t]["y"], oof[t])
        report["cv"][t] = r
        print(json.dumps({"target": t, "bacc": round(r["balanced_accuracy"], 3), "malignant_auc": round(r["malignant_auc"], 3),
                          "recall": {c: round(v, 2) for c, v in r["recall"].items()}}), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
