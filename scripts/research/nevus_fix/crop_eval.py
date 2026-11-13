#!/usr/bin/env python3
"""Inference-only view study: the shipped v10.4 HAM7 head on squash / centre-square / zoomed-crop features.

No refit. Views are combined by averaging row-L2 features and re-normalising. Selection on milk_dev
(clinical bacc + 0.5 * dermoscopic bacc); every other set is report-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import nevus_study as S
from metrics import evaluate

SETS = ("milk_dev", "milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def same_ids(view_dirs, name):
    source = "milk" if name.startswith("milk_") else name
    ids = [np.load(Path(d) / f"{source}.npz")["image_ids"] for d in view_dirs]
    return all(len(i) == len(ids[0]) and (i == ids[0]).all() for i in ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--views", required=True, help="name=dir,name=dir,... (first is the shipped view)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    head = np.load(args.head)
    views = dict(item.split("=") for item in args.views.split(","))
    data = {}
    for name in SETS:
        per_view = {v: S.load(Path(d), name) for v, d in views.items()}
        if any(x is None for x in per_view.values()):
            continue
        if not same_ids(views.values(), name):
            raise SystemExit(f"{name}: views are not row-aligned")
        data[name] = per_view
    combos = [c for r in range(1, len(views) + 1) for c in itertools.combinations(views, r)]
    report = {}
    for combo in combos:
        tag = "+".join(combo)
        report[tag] = {}
        for name, per_view in data.items():
            ref = per_view[combo[0]]
            x = l2(np.mean([l2(per_view[v]["x"]) for v in combo], 0))
            z = x @ head["weight"].T + head["bias"]
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            r = {"all": evaluate(ref["y"], p)}
            for m in ("clin", "derm"):
                mask = ref["mod"] == m
                if 20 <= mask.sum() < len(mask):
                    r[m] = evaluate(ref["y"][mask], p[mask])
            report[tag][name] = r
        dev = report[tag].get("milk_dev")
        report[tag]["dev_score"] = (dev["clin"]["balanced_accuracy"] + 0.5 * dev["derm"]["balanced_accuracy"]
                                    if dev else None)
        print(json.dumps({"views": tag, "dev_score": report[tag]["dev_score"],
                          "bacc": {n: round(r["all"]["balanced_accuracy"], 3) for n, r in report[tag].items()
                                   if n != "dev_score"},
                          "nv": {n: round(r["all"]["recall"].get("nv", float("nan")), 2) for n, r in report[tag].items()
                                 if n != "dev_score"}}), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
