#!/usr/bin/env python3
"""Paired bootstrap of HAM7 balanced accuracy: shipped view vs a two-view min-entropy combination."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import labels as L
import nevus_study as S
from metrics import evaluate
from view_ensemble import combine, probabilities

SETS = ("milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")
DRAWS = 2000


def bacc(y, p):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def view(spec):
    feats, head = spec.split(":")
    return Path(feats), np.load(head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="feats_dir:head.npz (shipped view)")
    parser.add_argument("--extra", required=True, help="feats_dir:head.npz (second view)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (fa, ha), (fb, hb) = view(args.base), view(args.extra)
    rng = np.random.default_rng(0)
    report = {}
    for name in SETS:
        da, db = S.load(fa, name), S.load(fb, name)
        y = np.asarray([L.HAM7.index(L.U8_TO_HAM7[u]) for u in da["y"]])
        pa = probabilities(ha, da["x"])
        pb = combine([pa, probabilities(hb, db["x"])], "minent")
        diffs = []
        for _ in range(DRAWS):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == len(np.unique(y)):
                diffs.append(bacc(y[i], pb[i]) - bacc(y[i], pa[i]))
        ra, rb = evaluate(da["y"], pa), evaluate(da["y"], pb)
        report[name] = {"n": int(len(y)), "bacc_base": bacc(y, pa), "bacc_views": bacc(y, pb),
                        "diff": bacc(y, pb) - bacc(y, pa), "ci95": np.percentile(diffs, [2.5, 97.5]).tolist(),
                        "malignant_auc": [ra["malignant_auc"], rb["malignant_auc"]]}
        print(json.dumps({name: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report[name].items()}}))
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
