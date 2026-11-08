#!/usr/bin/env python3
"""Additional architecture-bank results, explicitly separate from matched MoCo readouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(models):
    result = []
    for name, cohorts in sorted(models.items()):
        if not name.startswith("bank-"):
            continue
        protocol = ("supervised fine-tuning bank; not matched frozen LR" if name.endswith("-finetune")
                    else "frozen architecture bank; initializer/provenance differs")
        for cohort, metrics in cohorts.items():
            if cohort not in ("PAD", "HAM"):
                raise ValueError("Only approved development cohorts are eligible")
            for label, scores in metrics["per_class"].items():
                if any(not math.isfinite(scores[key]) or not 0 <= scores[key] <= 1 for key in ("precision", "recall", "f1")):
                    raise ValueError("Per-class bank metrics must be finite probabilities")
                result.append({"model": name, "protocol": protocol, "cohort": cohort, "class": label,
                               **{key: scores[key] for key in ("precision", "recall", "f1")},
                               "macro_f1": metrics.get("macro_f1")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "resultados/clasificacion/v10")
    args = parser.parse_args()
    source = args.output / "bank-source.json"
    if args.raw:
        raw = args.raw.read_bytes()
        models = {k: v for k, v in json.loads(raw).items() if k.startswith("bank-")}
        payload = {"source_raw_sha256": hashlib.sha256(raw).hexdigest(), "models": models,
                   "scope": "Development architecture bank; no independent validation or matched-training claim"}
        with source.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    payload = json.loads(source.read_text())
    records = rows(payload["models"])
    primary = json.loads((args.output / "metrics.json").read_text())
    for row in records:
        support = {item["class"]: item for item in primary["cohorts"][row["cohort"]]["records"][0]["classes"]}
        if row["class"] not in support:
            raise ValueError("Bank ontology differs from approved development classes")
        row.update({key: support[row["class"]][key] for key in ("support_images", "support_groups")})
    columns = ["model", "protocol", "cohort", "class", "precision", "recall", "f1", "macro_f1", "support_images", "support_groups"]
    output = io.StringIO()
    writer = csv.DictWriter(output, columns)
    writer.writeheader()
    writer.writerows(records)
    (args.output / "BANK.csv").write_text(output.getvalue())
    text = ["# Additional architecture-bank comparisons", "",
            "Descriptive development OOF results on the same approved cohorts. Do not mix this ranking with the controlled frozen-reader comparison.",
            "Fine-tuning uses a distinct supervised training/epoch-selection procedure; CV repetitions are not three globally trained SSL encoders.",
            "Frozen bank initializers/architectures differ; the timm DINOv2 bank is not the provenance-bound native reference in V5.",
            "Full per-class precision/recall/F1 and support: BANK.csv. Source aggregate: bank-source.json.", "",
            "| Model | Protocol | PAD macro-F1 | HAM macro-F1 |", "|---|---|---:|---:|"]
    for name, values in sorted(payload["models"].items()):
        protocol = "supervised fine-tuning" if name.endswith("-finetune") else "frozen architecture context"
        text.append(f"| {name} | {protocol} | {values['PAD']['macro_f1']:.4f} | {values['HAM']['macro_f1']:.4f} |")
    text += ["", "No clinical winner or significance claim follows. Sealed test roles were not accessed."]
    (args.output / "BANK.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"models": len(payload["models"]), "class_rows": len(records), "source_sha256": payload["source_raw_sha256"]}))


if __name__ == "__main__":
    main()
