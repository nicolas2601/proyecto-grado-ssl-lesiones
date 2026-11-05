"""Per-class precision/recall/F1 on PAD-UFES-20 from stored out-of-fold probabilities.

Usage: python scripts/per_class_report_20261029.py OUT_DIR NAME=DIR[,DIR...] [NAME=DIR ...]
Each NAME averages over every listed evaluation dir and every CV-seed run inside its PAD readout.
Predictions are the argmax of the stored OOF probabilities; labels come from the PAD metadata.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METADATA = Path("/home/jovyan/work/pad_ufes_20/metadata.csv")


def load_labels():
    return {row["img_id"].removesuffix(".png"): row["diagnostic"] for row in csv.DictReader(open(METADATA))}


def class_metrics(truth, predicted, classes):
    """Precision, recall, F1 per class; a class never predicted has precision 0.0 and is flagged."""
    result = {}
    for index, name in enumerate(classes):
        tp = int(np.sum((truth == index) & (predicted == index)))
        predicted_count, support = int(np.sum(predicted == index)), int(np.sum(truth == index))
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {"precision": precision, "recall": recall, "f1": f1, "never_predicted": predicted_count == 0}
    return result


def readout_metrics(path, labels):
    readout = json.loads(Path(path).read_text())
    classes = readout["classes"]
    truth = np.asarray([classes.index(labels[image_id]) for image_id in readout["image_ids"]])
    return [class_metrics(truth, np.asarray(run["oof_probabilities"]).argmax(axis=1), classes)
            for run in readout["runs"]], classes


def group_metrics(directories, labels):
    runs, classes = [], None
    for directory in directories:
        metrics, classes = readout_metrics(ROOT / "results" / directory / "PAD-readout.json", labels)
        runs.extend(metrics)
    mean = {name: {key: float(np.mean([run[name][key] for run in runs])) for key in ("precision", "recall", "f1")}
            for name in classes}
    return {"runs": len(runs), "classes": mean}


def markdown(report, classes):
    head = "| Método | " + " | ".join(f"{c} P / R" for c in classes) + " | F1 macro |\n"
    head += "|---|" + "---|" * (len(classes) + 1) + "\n"
    lines = []
    for name, group in report.items():
        cells = [f"{group['classes'][c]['precision']:.2f} / {group['classes'][c]['recall']:.2f}" for c in classes]
        macro = np.mean([group["classes"][c]["f1"] for c in classes])
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {macro:.3f} |")
    return head + "\n".join(lines) + "\n"


def main(argv):
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    report = {}
    for spec in argv[2:]:
        name, directories = spec.split("=", 1)
        report[name] = group_metrics(directories.split(","), labels)
    classes = sorted(next(iter(report.values()))["classes"])
    (out / "precision-por-clase.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / "precision-por-clase.md").write_text(markdown(report, classes))
    print(markdown(report, classes))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv)
