#!/usr/bin/env python3
"""Combine per-view HAM7 readers (each fit on its own view's features): single views, mean, min-entropy.

Views: name=feats_dir:head.npz. Selection on milk_dev (clinical bacc + 0.5 * dermoscopic bacc);
every other set is report-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import nevus_study as S
from metrics import evaluate

SETS = ("milk_dev", "milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")


def probabilities(head, x):
    z = x @ head["weight"].T + head["bias"]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def combine(probs, how):
    stack = np.stack(probs)
    if how == "mean":
        return stack.mean(0)
    entropy = -(stack * np.log(stack + 1e-12)).sum(2)
    return stack[entropy.argmin(0), np.arange(stack.shape[1])]


def scored(y, p, mod):
    out = {"all": evaluate(y, p)}
    for m in ("clin", "derm"):
        mask = mod == m
        if 20 <= mask.sum() < len(mask):
            out[m] = evaluate(y[mask], p[mask])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    views = {}
    for item in args.views.split(","):
        name, rest = item.split("=")
        feats, head = rest.split(":")
        views[name] = (Path(feats), np.load(head))
    report = {}
    for name in SETS:
        loads = {v: S.load(f, name) for v, (f, _) in views.items()}
        if any(d is None for d in loads.values()):
            continue
        ref = next(iter(loads.values()))
        if any(len(d["y"]) != len(ref["y"]) or (d["group"] != ref["group"]).any() for d in loads.values()):
            raise SystemExit(f"{name}: views are not row-aligned")
        probs = {v: probabilities(views[v][1], loads[v]["x"]) for v in views}
        variants = {v: p for v, p in probs.items()}
        variants["mean"] = combine(list(probs.values()), "mean")
        variants["minent"] = combine(list(probs.values()), "minent")
        report[name] = {v: scored(ref["y"], p, ref["mod"]) for v, p in variants.items()}
    summary = {}
    for v in report["milk_dev"]:
        dev = report["milk_dev"][v]
        summary[v] = {"dev_score": round(dev["clin"]["balanced_accuracy"] + 0.5 * dev["derm"]["balanced_accuracy"], 4),
                      "bacc": {n: round(report[n][v]["all"]["balanced_accuracy"], 3) for n in report},
                      "nv": {n: round(report[n][v]["all"]["recall"].get("nv", float("nan")), 2) for n in report},
                      "mel": {n: round(report[n][v]["all"]["recall"].get("mel", float("nan")), 2) for n in report}}
        print(json.dumps({"variant": v, **summary[v]}), flush=True)
    args.out.write_text(json.dumps({"summary": summary, "detail": report}, indent=1))


if __name__ == "__main__":
    main()
