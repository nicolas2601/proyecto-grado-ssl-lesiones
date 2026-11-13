#!/usr/bin/env python3
"""Render the nevus-collapse study JSON into a Markdown table (numbers are generated, never typed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SETS = (("milk_test", "clin", "MILK10k clinical, test lesions"), ("milk_test", "derm", "MILK10k dermoscopic, test lesions"),
        ("pad_test", "all", "PAD-UFES-20 test (smartphone)"), ("ddi", "all", "DDI (clinical, diverse skin tones)"),
        ("fitz", "all", "Fitzpatrick17k atlas subset (web)"), ("internet", "all", "Wikimedia Commons, reviewed (web)"),
        ("bosque", "all", "BOSQUE (Colombia)"), ("ham_holdout", "all", "HAM10000 holdout (dermoscopy)"))
COLUMNS = (("balanced_accuracy", "bacc"), ("non_nv_predicted_nv", "non-nevus→nevus"), ("mel_to_nv", "mel→nevus"),
           ("malignant_auc", "malignant AUC"))


def fmt(value) -> str:
    return "–" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    base, new = study["baselines"]["shipped_HAM7"], study["readers"][args.spec]
    head = "| External set | n | " + " | ".join(f"{label} v10.1 → v10.4" for _, label in COLUMNS) + " |"
    lines = [head, "|" + "---|" * (2 + len(COLUMNS))]
    for name, part, title in SETS:
        if name not in new or part not in new[name]:
            continue
        b, n = base[name][part], new[name][part]
        cells = [f"{fmt(b[key])} → {fmt(n[key])}" for key, _ in COLUMNS]
        lines.append(f"| {title} | {n['n']} | " + " | ".join(cells) + " |")
    lines += ["", f"Reader: `{args.spec}`; selection rule: {study['selection']['rule']}.",
              "Training rows: " + ", ".join(f"{k} {v}" for k, v in study["train_counts"].items()) + "."]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
