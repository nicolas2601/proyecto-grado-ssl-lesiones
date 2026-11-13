#!/usr/bin/env python3
"""Cost of a larger nevus prior on the v10.4 HAM7 head: nevus recall vs melanoma and non-nevus called nevus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import labels as L
import nevus_study as S

OFFSETS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SETS = (("milk_dev", "clin"), ("internet", None), ("ddi", None), ("pad_test", None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    head = np.load(args.head)
    nv, mel = L.HAM7.index("nv"), L.HAM7.index("mel")
    report = {}
    for name, mod in SETS:
        d = S.load(args.feats, name)
        y = np.asarray([L.HAM7.index(L.U8_TO_HAM7[u]) for u in d["y"]])
        z = d["x"] @ head["weight"].T + head["bias"]
        if mod:
            keep = d["mod"] == mod
            y, z = y[keep], z[keep]
        rows = []
        for off in OFFSETS:
            shifted = z.copy()
            shifted[:, nv] += off
            p = shifted.argmax(1)
            rows.append({"offset": off, "nevus_ok": int((p[y == nv] == nv).sum()), "nevi": int((y == nv).sum()),
                         "non_nv_to_nv": int((p[y != nv] == nv).sum()), "non_nv": int((y != nv).sum()),
                         "mel_to_nv": int((p[y == mel] == nv).sum()), "mel": int((y == mel).sum()),
                         "bacc": float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))})
        report[f"{name}/{mod}" if mod else name] = rows
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report)[:400])


if __name__ == "__main__":
    main()
