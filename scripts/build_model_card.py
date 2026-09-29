"""Generate docs/release/MODEL_CARD.md from the committed v7 evaluation files.

Every number comes from the ``final_eval.json`` of each registered seed
(``dermssl.v7_seeds``)
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
from dermssl import v7_seeds as seeds  # noqa: E402
from dermssl.release import package, preprocess  # noqa: E402

FINAL = {seed: seeds.final_eval_path(seed) for seed in seeds.REGISTERED_SEEDS}
TARGET = REPO / "docs" / "release" / "MODEL_CARD.md"
MODELS = (("A", "plain"), ("B", "proposed"))
TEMPLATE = REPO / "docs" / "release" / "model_card_template.md"
COLUMNS = ("Macro-F1", "Bal-acc", "Accuracy", "AUC (macro OvR)", "Mel sensitivity", "Mel AUC", "ECE")
DEPLOYED = (("macro_f1", "Macro-F1"), ("balanced_accuracy", "Bal-acc"), ("accuracy", "Accuracy"),
            ("mel_sensitivity", "Mel sensitivity"), ("mel_auc", "Mel AUC"), ("ece", "ECE"))
NO_PARITY = ("Not in this copy: the métricas de despliegue are written into the release folder's "
             "`MODEL_CARD.md` by `scripts/finalize_release.py`, from each model's `parity.json`.")


_NUMBER_EN = {2: "two", 4: "four", 6: "six", 8: "eight"}


def _models_text(arm: str) -> str:
    return ", ".join(f"`{name}`" for name in seeds.model_names((arm,)))


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
    lines.append(model_row("DINOv2 public ViT-B/14 (reference, not released)", docs[seeds.BASE_SEED],
                           probas[seeds.BASE_SEED], "public", classes))
    return "\n".join(lines)


def _deployed_row(model: str, label: str, block: dict) -> str:
    cells = [f"{block['metrics'][k]['value']:.4f} ({block['metrics'][k]['delta']:+.4f})"
             for k, _ in DEPLOYED]
    return f"| `{model}` | {label} | " + " | ".join(cells) + f" | {block['n_predictions_changed']} |"


def deployment_table(parity_root: Path | None) -> str:
    """Rows per model and ONNX precision from ``<root>/<model>/parity.json``; a note without them."""
    paths = sorted(parity_root.glob("*/parity.json")) if parity_root else []
    if not paths:
        return NO_PARITY
    lines = ["| Model | Form | " + " | ".join(label for _, label in DEPLOYED)
             + " | Predictions changed |", "|---|---|" + "---|" * (len(DEPLOYED) + 1)]
    for path in paths:
        blocks = json.loads(path.read_text(encoding="utf-8"))["deployment_metrics"]
        lines.append(_deployed_row(path.parent.name, "ONNX fp32", blocks["onnx_fp32"]))
        if "onnx_fp16" in blocks:
            state = "published" if blocks["onnx_fp16"]["gate"]["publish"] else "not published"
            lines.append(_deployed_row(path.parent.name, f"ONNX fp16 ({state})", blocks["onnx_fp16"]))
    return "\n".join(lines)


def load() -> tuple[dict, dict]:
    docs, probas = {}, {}
    for seed, path in FINAL.items():
        docs[seed] = json.loads((REPO / path).read_text(encoding="utf-8"))
        with np.load(REPO / path.with_suffix(".probas.npz")) as npz:
            probas[seed] = {k: npz[k] for k in npz.files}
    return docs, probas


def render(parity_root: Path | None = None) -> str:
    docs, probas = load()
    values = {
        "metrics_table": metrics_table(docs, probas),
        "deployment_table": deployment_table(parity_root),
        "n_val": str(docs[seeds.BASE_SEED]["n_eval"]),
        "n_models": _NUMBER_EN.get(len(MODELS) * len(FINAL), str(len(MODELS) * len(FINAL))).capitalize(),
        "a_models": _models_text("A"), "b_models": _models_text("B"),
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
