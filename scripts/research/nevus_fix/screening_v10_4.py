#!/usr/bin/env python3
"""Select the melanoma-priority offset for the v10.4 HAM7 head on milk_dev (rule: v10.4/screening/RULE.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import labels as L
import nevus_study as S
from screening_select import DELTA_GRID, operating_curve, operating_metrics, screening_predictions, select_delta

TESTS = ("milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet")
SEED = 20261113


def probabilities(head, x):
    z = x @ head["weight"].T + head["bias"]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ham7(u8):
    return np.asarray([L.HAM7.index(L.U8_TO_HAM7[u]) for u in u8])


def nested(y, p, groups):
    order = np.random.default_rng(SEED).permutation(len(y))
    y, p, groups = y[order], p[order], groups[order]
    picks, preds = [], np.empty(len(y), dtype=int)
    for fit, held in GroupKFold(5).split(p, y, groups):
        curve = operating_curve(y[fit], p[fit], L.HAM7, "mel", "nv")
        delta = select_delta({d: m["macro_f1"] for d, m in curve.items()})
        picks.append(delta)
        preds[held] = screening_predictions(p[held], L.HAM7.index("mel"), delta)
    return picks, operating_metrics(y, preds, L.HAM7, "mel", "nv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    head = np.load(args.head, allow_pickle=False)
    dev = S.load(args.feats, "milk_dev")
    y, p = ham7(dev["y"]), probabilities(head, dev["x"])
    curve = operating_curve(y, p, L.HAM7, "mel", "nv")
    delta = select_delta({d: m["macro_f1"] for d, m in curve.items()})
    picks, held = nested(y, p, dev["group"])
    report = {"grid": list(DELTA_GRID), "selected_delta": delta, "dev_curve": {str(d): m for d, m in curve.items()},
              "nested": {"fold_deltas": picks, "metrics": held, "at_zero": curve[0.0]}, "tests": {}}
    for name in TESTS:
        d = S.load(args.feats, name)
        if d is None:
            continue
        yt, pt = ham7(d["y"]), probabilities(head, d["x"])
        report["tests"][name] = {
            "at_zero": operating_metrics(yt, pt.argmax(1), L.HAM7, "mel", "nv"),
            "at_delta": operating_metrics(yt, screening_predictions(pt, L.HAM7.index("mel"), delta), L.HAM7, "mel", "nv")}
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"selected_delta": delta, "fold_deltas": picks,
                      "nested_mel_recall": round(held["target_recall"], 3), "nested_macro_f1": round(held["macro_f1"], 3),
                      "at_zero_mel_recall": round(curve[0.0]["target_recall"], 3)}))


if __name__ == "__main__":
    main()
