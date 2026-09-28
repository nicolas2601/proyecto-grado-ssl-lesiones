"""Generate docs/release/MODEL_CARD.md from the committed v7 evaluation files.

Every number comes from ``resultados/clasificacion/v7[/s43]/final_eval.json``
and the committed probabilities beside them (accuracy and melanoma metrics
with the same bootstrap as TABLA_CLASIFICADOR.md). Nothing is transcribed.

    python scripts/build_model_card.py            # write
    python scripts/build_model_card.py --check    # fail if the committed card is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import per_class_metrics as pcm  # noqa: E402
from dermssl.release import package, preprocess  # noqa: E402

V7 = Path("resultados") / "clasificacion" / "v7"
FINAL = {42: V7 / "final_eval.json", 43: V7 / "s43" / "final_eval.json"}
TARGET = REPO / "docs" / "release" / "MODEL_CARD.md"
MODELS = (("A", "plain"), ("B", "proposed"))
TEMPLATE = REPO / "docs" / "release" / "model_card_template.md"
COLUMNS = ("Macro-F1", "Bal-acc", "Accuracy", "AUC (macro OvR)", "Mel sensitivity", "Mel AUC", "ECE")


def _cell(value: float, ci) -> str:
    return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def model_row(label: str, doc: dict, probas: dict, role: str, classes) -> str:
    rec, y, proba = doc["results"][role], probas["y_true"], probas[role]
    acc, acc_low, acc_high = pcm.accuracy_with_ci(y, proba)
    report = pcm.class_report(y, proba, classes)
    mel = pcm.positive_intervals(y, proba, classes)
    if abs(report["macro_f1"] - rec["macro_f1"]) > package.JSON_F1_TOLERANCE:
        raise ValueError(f"{label}: macro-F1 from the probabilities differs from final_eval.json")
    cells = [_cell(rec["macro_f1"], rec["ci95"]["macro_f1"]),
             _cell(rec["balanced_accuracy"], rec["ci95"]["balanced_accuracy"]),
             _cell(acc, (acc_low, acc_high)), _cell(rec["auc"], rec["ci95"]["auc"]),
             _cell(report["positive_sensitivity"], mel["sensitivity"]),
             _cell(report["positive_auc"], mel["auc"]), _cell(rec["ece"], rec["ci95"]["ece"])]
    return f"| {label} | " + " | ".join(cells) + " |"


def metrics_table(docs: dict, probas: dict) -> str:
    classes = pcm.HAM10000_CLASSES
    lines = ["| Model | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
    for arm, role in MODELS:
        for seed in sorted(docs):
            lines.append(model_row(f"`{arm}_s{seed}` ({docs[seed]['arms'][role]})", docs[seed],
                                   probas[seed], role, classes))
    lines.append(model_row("DINOv2 public ViT-B/14 (reference, not released)", docs[42],
                           probas[42], "public", classes))
    return "\n".join(lines)


def load() -> tuple[dict, dict]:
    docs, probas = {}, {}
    for seed, path in FINAL.items():
        docs[seed] = json.loads((REPO / path).read_text(encoding="utf-8"))
        with np.load(REPO / path.with_suffix(".probas.npz")) as npz:
            probas[seed] = {k: npz[k] for k in npz.files}
    return docs, probas


def render() -> str:
    docs, probas = load()
    values = {
        "metrics_table": metrics_table(docs, probas),
        "n_val": str(docs[42]["n_eval"]),
        "sources": ", ".join(f"`{p.as_posix()}`" for p in FINAL.values()),
        "preprocessing": json.dumps(preprocess.PREPROCESSING, indent=1),
        "forms_atol": f"{package.FORMS_ATOL:g}", "json_tol": f"{package.JSON_F1_TOLERANCE:g}",
        "fp16_agreement": f"{package.FP16_MIN_AGREEMENT:.1%}",
        "fp16_delta": f"{package.FP16_MAX_DELTA_F1:g}",
    }
    text = TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError("unfilled placeholder in the model card template")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if TARGET.read_text(encoding="utf-8") != text:
            print("MODEL_CARD.md is stale: run python scripts/build_model_card.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"-> {TARGET.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
