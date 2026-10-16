"""Render the docs/30 7 addendum (A71 and seed 44 on PAD-UFES-20) from its JSON; ``--check`` for CI.

    python scripts/build_pad_ufes_addendum_table.py \
        resultados/clasificacion/pad_ufes/pad_ufes_addendum_a71_s44.json \
        --out resultados/clasificacion/pad_ufes/adenda_a71_s44.md [--check]

Every number is read from the JSON. Descriptive only: a contrast is shown as
"el IC excluye el 0", "no se distingue" or "no calculable (k/1000)", the last
one only in the phototype strata (docs/30 6, rule (b)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import pad_ufes as pu  # noqa: E402
from dermssl import pad_ufes_addendum as pa  # noqa: E402

METRICS = ("macro_f1", "balanced_accuracy", "auc", "ece")
METRIC_LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced acc.", "auc": "AUC", "ece": "ECE"}
READING_LABELS = {"primary": "Primaria descriptiva (bcc, mel, nv)",
                  "secondary": "Secundaria (akiec = ACK + SCC, bcc, mel, nv)",
                  "exploratory": "Exploratoria (akiec, bcc, bkl = SEK, mel, nv)"}
CONTRAST_LABELS = {pa.EXCLUDES_ZERO: "el IC excluye el 0", pa.INDISTINGUISHABLE: "no se distingue"}
DATASET = ("Pacheco et al., *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical "
           "images collected from smartphones*, Mendeley Data, v1, doi "
           "[10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1); artículo en *Data in Brief* 32 "
           "(2020) 106221, doi [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221). "
           "Licencia CC BY 4.0 (<http://creativecommons.org/licenses/by/4.0>). Cambios: el mapeo de etiquetas "
           "de docs/30 §2.2, la exclusión de SEK de las lecturas primaria y secundaria y el preprocesado del C1.")


def _num(x) -> str:
    return "n/d" if x is None else f"{x:.4f}".replace(".", ",")


def _k(k: int, n_resamples: int) -> str:
    return f"no calculable ({k}/{n_resamples})"


def _ci(pair, k: int, n_resamples: int) -> str:
    return _k(k, n_resamples) if k else f"[{_num(pair[0])}; {_num(pair[1])}]"


def _arm_cell(arm: dict, metric: str, n_resamples: int) -> str:
    k = arm.get("n_nonfinite", {}).get(metric, 0)
    return f"{_num(arm[metric])} {_ci(arm['ci95'][metric], k, n_resamples)}"


def _arms_table(entry: dict, n_resamples: int, outside: dict | None = None) -> list[str]:
    head = ["Modelo", *(METRIC_LABELS[m] + " [IC 95 %]" for m in METRICS)]
    if outside is not None:
        head.append("Argmax fuera de la lectura")
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---|" * (len(head) - 1)]
    for name, arm in entry["arms"].items():
        cells = [f"`{name}`", *(_arm_cell(arm, m, n_resamples) for m in METRICS)]
        if outside is not None:
            cells.append(_num(outside[name]))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _reading(row: dict, n_resamples: int) -> str:
    if row["reading"] == pa.NOT_CALCULABLE:
        return _k(row["n_nonfinite"], n_resamples)
    return CONTRAST_LABELS[row["reading"]]


def _contrasts_table(rows: list[dict], n_resamples: int) -> list[str]:
    lines = ["| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |", "|---|---|---:|---|---|"]
    for row in rows:
        ci = "n/d" if row["ci95_low"] is None else f"[{_num(row['ci95_low'])}; {_num(row['ci95_high'])}]"
        lines.append(f"| `{row['arm']}` − `{row['control']}` | {METRIC_LABELS[row['metric']]} "
                     f"| {_num(row['point_delta'])} | {ci} | {_reading(row, n_resamples)} |")
    return lines


def _readings(doc: dict, n_resamples: int) -> list[str]:
    lines = []
    for name, entry in doc["readings"].items():
        lines += ["", f"## {READING_LABELS[name]}", "",
                  f"{entry['n_images']} imágenes, {entry['n_patients']} pacientes.", "",
                  *_arms_table(entry, n_resamples, entry["outside_rate"]), "",
                  *_contrasts_table(entry["paired"], n_resamples)]
    return lines


def _strata(doc: dict, n_resamples: int) -> list[str]:
    strata = doc["fst_strata"]
    lines = ["", "## Estratos por fototipo (lectura primaria, exploratorio)", "",
             "Regla (b) de docs/30 §6: el valor puntual se reporta siempre; si alguna remuestra da un valor no "
             f"finito, el IC queda \"no calculable\" con el número exacto k/{n_resamples} de remuestras no finitas. "
             "No se descarta ninguna remuestra.", "", strata["note"]]
    for name, entry in strata["bins"].items():
        lines += ["", f"### {name}", ""]
        if not entry["n_images"]:
            lines.append(f"Sin imágenes ({entry['power']}).")
            continue
        power = "" if entry["power"] == "ok" else f" Rótulo: {entry['power']}."
        lines += [f"{entry['n_images']} imágenes, {entry['n_patients']} pacientes.{power}", "",
                  *_arms_table(entry, n_resamples), "", *_contrasts_table(entry["paired"], n_resamples)]
    return lines


from dermssl.display_nomenclature import presentation  # noqa: E402


@presentation
def render(doc: dict) -> str:
    n_resamples = doc["bootstrap"]["n_resamples"]
    protocols = [f"- `{role}`: {protocol}" for role, protocol in doc["protocols"].items()]
    lines = [
        "# PAD-UFES-20, adenda descriptiva: A71 y semilla 44 (docs/30 §7)", "",
        "Generado por `scripts/build_pad_ufes_addendum_table.py` desde `pad_ufes_addendum_a71_s44.json`; no "
        "editar a mano (`--check` lo compara y `tests/test_build_pad_ufes_addendum_table.py` lo corre en CI). "
        f"`code_revision` {doc['code_revision']}, registro `{doc['registration_commit']}`.", "",
        "**Solo descriptivo.** No hay predicción ni regla de decisión: un contraste se lee a lo sumo como "
        "\"el IC pareado excluye el 0\" o \"no se distingue\". Bootstrap pareado agrupado por paciente, semilla "
        f"{doc['bootstrap']['rng_seed']}, {n_resamples} remuestras.", "",
        "## Modelos y protocolos", "", *protocols,
        *_readings(doc, n_resamples),
        *_strata(doc, n_resamples), "",
        "## Datos", "", f"- {DATASET}", f"- {pu.THESIS_FST_SENTENCE}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    doc = json.loads(args.document.read_text(encoding="utf-8"))
    pa.assert_descriptive(doc)
    text = render(doc)
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
