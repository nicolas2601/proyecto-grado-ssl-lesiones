"""Render the T1 table (docs/30 5) from its JSON; ``--check`` for CI.

    python scripts/build_pad_ufes_fst_table.py \
        resultados/fst/pad_ufes_t1/pad_ufes_fst_validation.json \
        --out resultados/fst/pad_ufes_t1/README.md [--check]

Every number, both verdicts and the phototype note are read from the JSON;
the kappa threshold and the conditions come from the validation script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import closing_table_rules as rules  # noqa: E402
import pad_ufes_fst_validation as fv  # noqa: E402

CONDITION_LABELS = {"whole_image": "Imagen completa", "masked": "Fondo por Otsu (máscara)"}
ROMAN = {k: r for k, r in zip(fv.FST_LABELS, ("I", "II", "III", "IV", "V", "VI"))}
NOT_AVAILABLE = "n/d"
DATASET = ("Pacheco et al., *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical "
           "images collected from smartphones*, Mendeley Data, v1, doi "
           "[10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1); artículo en *Data in Brief* 32 "
           "(2020) 106221, doi [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221). "
           "Licencia CC BY 4.0 (<http://creativecommons.org/licenses/by/4.0>). Cambios: solo se usan las "
           "imágenes con `fitspatrick` válido; no se guarda ni se modifica ninguna imagen.")
COMPARISON = ("Alipour, Burke y Courtney, *Skin Research and Technology* 32(8):e70374, 2026, doi "
              "[10.1111/srt.70374](https://doi.org/10.1111/srt.70374), PMC13459198: con parches de piel, el "
              "ITA con umbrales fijos queda por debajo de la baseline de mayoría en PAD-UFES-20 (docs/30 §5).")


def _num(x) -> str:
    return NOT_AVAILABLE if x is None else f"{x:.4f}".replace(".", ",")


def _count(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _with_ci(value, pair) -> str:
    if value is None or pair is None:
        return NOT_AVAILABLE
    return f"{_num(value)} [{_num(pair[0])}; {_num(pair[1])}]"


def _baseline(base: dict) -> str:
    if base["label"] is None:
        return NOT_AVAILABLE
    return f"{ROMAN[base['label']]} · {_num(base['accuracy'])} · {_num(base['off_by_one_accuracy'])}"


def _beats(beats: bool | None) -> str:
    return NOT_AVAILABLE if beats is None else ("sí" if beats else "no")


def _metrics_table(metrics: dict) -> list[str]:
    lines = ["| Condición | n evaluables | n sin estimación | Accuracy exacta [IC 95 %] | Off-by-one "
             "| Kappa lineal [IC 95 %] | Baseline de mayoría (fototipo · acc. · off-by-one) "
             "| ¿Supera la baseline? |", "|---|---:|---:|---|---|---|---|---|"]
    for condition in fv.CONDITIONS:
        m = metrics[condition]
        cells = (CONDITION_LABELS[condition], _count(m["n_evaluable"]), _count(m["n_unavailable"]),
                 _with_ci(m["accuracy"], m["ci95"]["accuracy"]), _num(m["off_by_one_accuracy"]),
                 _with_ci(m["linear_kappa"], m["ci95"]["linear_kappa"]), _baseline(m["majority_baseline"]),
                 _beats(m["beats_majority"]))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _confusion(condition: str, confusion: dict) -> list[str]:
    head = " | ".join(ROMAN[k] for k in fv.FST_LABELS)
    lines = [f"### {CONDITION_LABELS[condition]}", "", f"| Real \\ Predicho | {head} | Total |",
             "|---|" + "---:|" * (len(fv.FST_LABELS) + 1)]
    for actual in fv.FST_LABELS:
        row = confusion.get(str(actual), {})
        counts = [row.get(str(predicted), 0) for predicted in fv.FST_LABELS]
        lines.append(f"| {ROMAN[actual]} | " + " | ".join(_count(c) for c in counts)
                     + f" | {_count(sum(counts))} |")
    return lines


def _verdicts(verdicts: dict) -> list[str]:
    kappa = f"{fv.SLIGHT_KAPPA:.2f}".replace(".", ",")
    return [f"- **FST-P1** (predicción primaria: en ninguna condición la accuracy exacta supera la baseline "
            f"de mayoría en las mismas filas): **{verdicts['FST-P1']}**.",
            f"- **FST-P2** (predicción secundaria: el kappa lineal queda por debajo de {kappa} en las dos "
            f"condiciones): **{verdicts['FST-P2']}**."]


def render(doc: dict) -> str:
    matrices = [line for c in fv.CONDITIONS for line in ["", *_confusion(c, doc["metrics"][c]["confusion"])]]
    lines = [
        "# T1: la estimación ITA del proyecto contra el `fitspatrick` de PAD-UFES-20 (docs/30 §5)", "",
        "Generado por `scripts/build_pad_ufes_fst_table.py` desde `pad_ufes_fst_validation.json`; no editar a "
        "mano (`--check` lo compara, y `tests/test_build_pad_ufes_fst_table.py` lo corre en CI). Exploratorio. "
        f"`code_revision` {doc['code_revision']}, registro `{doc['registration_commit']}`. "
        f"{_count(doc['n_images'])} imágenes, {_count(doc['n_without_fst'])} sin fototipo válido.", "",
        rules.ITA_NOTE, "",
        "## Métricas por condición", "",
        "Las filas sin estimación (`status` distinto de `ok`) quedan fuera de las métricas. Los IC son "
        "bootstrap agrupado por paciente y son descriptivos.", "",
        *_metrics_table(doc["metrics"]), "",
        "## Veredictos de las predicciones de docs/30 §5", "", *_verdicts(doc["verdicts"]), "",
        "## Matriz de confusión (filas: fototipo de PAD; columnas: bucket ITA predicho)", *matrices, "",
        "## Nota sobre los fototipos", "", doc["fst_note"], "",
        "## Datos y referencia", "", f"- Datos: {DATASET}", f"- Comparación: {COMPARISON}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(json.loads(args.document.read_text(encoding="utf-8")))
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} no coincide con {args.document}; regenerar")
            return 1
        return 0
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
