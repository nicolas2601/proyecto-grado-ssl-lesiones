"""Build resultados/clasificacion/TABLA_CLASIFICADOR.md from committed JSON/NPZ.

One table puts the v6 bank, the C1 "none" cells and the v7 arms side by side,
and a per-class section opens the v7 arms up from their saved probabilities
(confusion matrix, F1 and recall per class, melanoma sensitivity and AUC).
Nothing is transcribed: ``--check`` fails in CI if the table drifts.

    python scripts/build_classifier_table.py            # write
    python scripts/build_classifier_table.py --check    # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import final_eval as fe  # noqa: E402
from dermssl import per_class_metrics as pcm  # noqa: E402

CLS = Path("resultados") / "clasificacion"
BANK = CLS / "bank.json"
C1_CELLS = CLS / "c1" / "c1_cells.json"
FINAL = {42: CLS / "v7" / "final_eval.json", 43: CLS / "v7" / "s43" / "final_eval.json"}
PROBAS = {42: CLS / "v7" / "final_eval.probas.npz", 43: CLS / "v7" / "s43" / "final_eval.probas.npz"}
TARGET = REPO / CLS / "TABLA_CLASIFICADOR.md"
BACKBONES = ("dinov2_public_vits14", "dinov2_public_vitb14", "ssl_plain_vits14",
             "ssl_plain_vitb14", "ssl_proposed_vits14", "ssl_proposed_vitb14")
ROLES = ("plain", "proposed", "public")
V6_ARM = {"probe": "logreg", "fraction": 1.0, "seed": 0}
C1_CONDITION = "none"
NO_CI = "—"
TOLERANCE = 1e-9
LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "auc": "AUC", "ece": "ECE"}


def _num(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def _cell(value: float, ci: list[float] | None) -> str:
    return f"{_num(value)} [{_num(ci[0])}; {_num(ci[1])}]" if ci else f"{_num(value)} {NO_CI}"


def _row(version: str, arm: str, values: dict, cis: dict | None, source: str) -> str:
    cells = [_cell(values[m], cis[m] if cis else None) for m in fe.METRICS]
    return f"| {version} | {arm} | " + " | ".join(cells) + f" | {source} |"


def load_sources() -> dict:
    def read(path: Path) -> dict:
        return json.loads((REPO / path).read_text(encoding="utf-8"))

    probas = {}
    for seed, path in PROBAS.items():
        with np.load(REPO / path) as npz:
            probas[seed] = {k: npz[k] for k in npz.files}
    return {"bank": read(BANK), "c1": read(C1_CELLS),
            "docs": {seed: read(path) for seed, path in FINAL.items()}, "probas": probas}


def check_public_identical(docs: dict, probas: dict) -> None:
    """The public arm has no seed: both evaluations must hold the same numbers."""
    a, b = docs[42]["results"]["public"], docs[43]["results"]["public"]
    same_json = all(a[m] == b[m] and a["ci95"][m] == b["ci95"][m] for m in fe.METRICS)
    same_npz = np.array_equal(probas[42]["public"], probas[43]["public"])
    if not (same_json and same_npz):
        raise ValueError("el brazo DINOv2 público difiere entre s42 y s43; no se puede listar una vez")


def v6_rows(bank: dict) -> list[str]:
    rows = {r["backbone"]: r for r in bank["rows"] if all(r[k] == v for k, v in V6_ARM.items())}
    return [_row("v6", b, rows[b], None, f"`{BANK.as_posix()}`") for b in BACKBONES]


def c1_rows(c1: dict) -> list[str]:
    return [_row("C1", b, c1[f"{b}__{C1_CONDITION}"]["metrics"], None, f"`{C1_CELLS.as_posix()}`")
            for b in BACKBONES]


def v7_rows(docs: dict) -> list[str]:
    arms, both = docs[42]["arms"], f"`{FINAL[42].as_posix()}`, `{FINAL[43].as_posix()}`"
    lines = []
    for role in ("plain", "proposed"):
        for seed, doc in docs.items():
            rec = doc["results"][role]
            lines.append(_row(f"v7 s{seed}", arms[role], rec, rec["ci95"], f"`{FINAL[seed].as_posix()}`"))
        mean = {m: sum(d["results"][role][m] for d in docs.values()) / len(docs) for m in fe.METRICS}
        lines.append(_row("media s42/s43", arms[role], mean, None, both))
    pub = docs[42]["results"]["public"]
    lines.append(_row("v7 s42 = s43", arms["public"], pub, pub["ci95"], both))
    return lines


def summary_section(src: dict) -> list[str]:
    bank, c1, docs = src["bank"], src["c1"], src["docs"]
    n_c1 = sorted({cell["gate"]["n_images"] for cell in c1.values()})
    fraction = str(V6_ARM["fraction"]).replace(".", ",")
    header = "| Versión | Backbone/brazo | " + " | ".join(LABELS[m] for m in fe.METRICS) + " | Fuente |"
    return [
        "## Resumen por versión", "",
        "ECE: menor es mejor; en Macro-F1, Bal-acc y AUC, mayor es mejor. Entre corchetes, "
        f"IC 95 % bootstrap; «{NO_CI}» = la fuente no trae IC por brazo.", "",
        "Los protocolos no son idénticos y las filas no se comparan entre versiones sin leerlos: "
        f"v6 = banco con probe `{V6_ARM['probe']}`, fracción {fraction}, "
        f"semilla {V6_ARM['seed']}, val n = {bank['split']['val']}; C1 = condición `{C1_CONDITION}`, "
        f"n = {', '.join(map(str, n_c1))}; v7 = probe `{docs[42]['probe']}`, val n = "
        f"{docs[42]['n_eval']} (s42) y {docs[43]['n_eval']} (s43).", "",
        header, "|---|---|" + "---|" * len(fe.METRICS) + "---|",
        *v6_rows(bank), *c1_rows(c1), *v7_rows(docs),
    ]


def per_class_reports(src: dict) -> dict[str, dict]:
    """``{label: class_report}``; A*/B* per seed, the public arm once."""
    docs, probas, classes = src["docs"], src["probas"], src["bank"]["split"]["classes"]
    arms, out = docs[42]["arms"], {}
    jobs = [(seed, role) for role in ("plain", "proposed") for seed in docs] + [(42, "public")]
    for seed, role in jobs:
        report = pcm.class_report(probas[seed]["y_true"], probas[seed][role], classes)
        published = docs[seed]["results"][role]["macro_f1"]
        if abs(report["macro_f1"] - published) > TOLERANCE:
            raise ValueError(f"macro-F1 de {role} s{seed} desde probas ({report['macro_f1']}) "
                             f"no coincide con final_eval.json ({published})")
        label = f"{arms[role]} s42 = s43" if role == "public" else f"{arms[role]} s{seed}"
        out[label] = report
    return out


def _class_table(title: str, key: str, reports: dict, classes: list[str]) -> list[str]:
    lines = [f"### {title}", "", "| Brazo | " + " | ".join(classes) + " |",
             "|---|" + "---|" * len(classes)]
    for label, rep in reports.items():
        lines.append(f"| {label} | " + " | ".join(_num(v) for v in rep[key]) + " |")
    return lines + [""]


def _confusion(label: str, cm: np.ndarray, classes: list[str]) -> list[str]:
    lines = [f"#### {label}", "", "| real \\ predicha | " + " | ".join(classes) + " |",
             "|---|" + "---:|" * len(classes)]
    for name, row in zip(classes, cm):
        lines.append(f"| {name} | " + " | ".join(str(int(v)) for v in row) + " |")
    return lines + [""]


def per_class_section(src: dict) -> list[str]:
    classes = list(src["bank"]["split"]["classes"])
    reports = per_class_reports(src)
    lines = ["## Por clase (v7, desde las probabilidades guardadas)", "",
             "Predicción = argmax de `final_eval.probas.npz`. Orden de clases de `bank.json` "
             "(`split.classes`). El macro-F1 recalculado desde las probabilidades coincide con "
             f"`final_eval.json` (tolerancia {TOLERANCE:g}); si no, este script falla.", "",
             "| Brazo | Sensibilidad melanoma | AUC melanoma (uno contra el resto) | Macro-F1 |",
             "|---|---|---|---|"]
    for label, rep in reports.items():
        lines.append(f"| {label} | {_num(rep['positive_sensitivity'])} | "
                     f"{_num(rep['positive_auc'])} | {_num(rep['macro_f1'])} |")
    lines += ["", *_class_table("F1 por clase", "f1", reports, classes),
              *_class_table("Recall por clase", "recall", reports, classes),
              "### Matrices de confusión (filas: clase real; columnas: predicha)", ""]
    for label, rep in reports.items():
        lines += _confusion(label, rep["confusion"], classes)
    return lines


def render(src: dict) -> str:
    pcm.check_class_order(src["bank"]["split"]["classes"])
    check_public_identical(src["docs"], src["probas"])
    lines = ["# Clasificador: v6, C1 y v7 en una tabla", "",
             "Generado por `scripts/build_classifier_table.py` desde JSON y NPZ commiteados. "
             "No editar a mano: `--check` lo compara en CI.", "",
             *summary_section(src), "", *per_class_section(src)]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(load_sources())
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} esta desactualizado: corre scripts/build_classifier_table.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
