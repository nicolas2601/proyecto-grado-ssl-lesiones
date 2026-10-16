"""Render the docs/30 8 WiSE-FT alpha curve on PAD-UFES-20 from its JSON; ``--check`` for CI.

    python scripts/build_pad_ufes_wiseft_table.py \
        resultados/clasificacion/pad_ufes/pad_ufes_wiseft.json \
        --out resultados/clasificacion/pad_ufes/wiseft_alpha.md [--check]

Every number is read from the JSON. PAD-W1 and PAD-W2 are shown as the JSON
decided them; every other contrast is "el IC excluye el 0", "no se distingue"
or, only in the phototype strata, "no calculable (k/1000)" (docs/30 6, rule (b)).
A ``|`` inside a cell (the v7.1 config ids) is escaped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import build_pad_ufes_addendum_table as addendum  # noqa: E402

from dermssl import pad_ufes as pu  # noqa: E402

METRICS, METRIC_LABELS = addendum.METRICS, addendum.METRIC_LABELS
READING_LABELS = {**addendum.READING_LABELS, "primary": "Primaria (bcc, mel, nv), la que decide"}
num = addendum._num  # noqa: SLF001 - one number format for every PAD table


def cell(text) -> str:
    return str(text).replace("|", "\\|")


def row(cells) -> str:
    return "| " + " | ".join(cell(c) for c in cells) + " |"


def header(names, align: str = "---") -> list[str]:
    return [row(names), "|" + "|".join(align for _ in names) + "|"]


def alpha_label(alpha) -> str:
    return f"{float(alpha):.1f}".replace(".", ",")


def _ci(low, high) -> str:
    return "n/d" if low is None else f"[{num(low)}; {num(high)}]"


def _contrast_reading(r: dict, n_resamples: int) -> str:
    if r["excludes_zero"] is None:
        return f"no calculable ({r['n_nonfinite']}/{n_resamples})"
    return "el IC excluye el 0" if r["excludes_zero"] else "no se distingue"


def _contrasts(rows: list[dict], n_resamples: int) -> list[str]:
    lines = header(["Contraste", "Métrica", "Δ puntual", "IC 95 % pareado", "Lectura"])
    for r in rows:
        lines.append(row([f"`{r['arm']}` − `{r['control']}`", METRIC_LABELS[r["metric"]], num(r["point_delta"]),
                          _ci(r["ci95_low"], r["ci95_high"]), _contrast_reading(r, n_resamples)]))
    return lines


def _arm_cell(arm: dict, metric: str, n_resamples: int) -> str:
    k = arm.get("n_nonfinite", {}).get(metric, 0)
    ci = f"no calculable ({k}/{n_resamples})" if k else _ci(*arm["ci95"][metric])
    return f"{num(arm[metric])} {ci}"


def _arms(entry: dict, n_resamples: int, outside: dict | None = None) -> list[str]:
    names = ["Pesos", *(METRIC_LABELS[m] + " [IC 95 %]" for m in METRICS)]
    lines = header(names + (["Argmax fuera de la lectura"] if outside else []))
    for name, arm in entry["arms"].items():
        extra = [num(outside[name])] if outside else []
        lines.append(row([f"`{name}`", *(_arm_cell(arm, m, n_resamples) for m in METRICS), *extra]))
    return lines


def _w1(verdict: dict) -> list[str]:
    lines = [f"**PAD-W1: {verdict['PAD-W1']}** (macro-F1 primaria, α = 0,8 − α = 1,0; Δ puntual > 0 en los "
             "tres modelos; \"mejora\" solo donde el IC pareado queda por encima de 0).", "",
             *header(["Modelo", "Δ puntual", "IC 95 % pareado", "Δ > 0", "Mejora"])]
    for model, r in verdict["by_model"].items():
        lines.append(row([model, num(r["point_delta"]), _ci(r["ci95_low"], r["ci95_high"]),
                          "sí" if r["holds"] else "no", "sí" if r["mejora"] else "no"]))
    return lines


def _w2(verdict: dict) -> list[str]:
    tol = num(verdict["tolerance"])
    lines = [f"**PAD-W2: {verdict['PAD-W2']}** (ningún paso de α a α − 0,2 baja el macro-F1 puntual en más de "
             f"{tol}, y α = 0 es el máximo o queda a ≤ {tol} del máximo, en los tres modelos).", "",
             *header(["Modelo", "Pasos que violan la tolerancia", "Distancia de α = 0 al máximo", "Cumple"])]
    for model, r in verdict["by_model"].items():
        steps = "; ".join(f"{alpha_label(v['from_alpha'])} → {alpha_label(v['to_alpha'])}: −{num(v['drop'])}"
                          for v in r["violations"]) or "ninguno"
        lines.append(row([model, steps, num(r["public_gap_to_max"]), "sí" if r["holds"] else "no"]))
    return lines


def _curve(doc: dict) -> list[str]:
    arms, ham = doc["readings"]["primary"]["arms"], doc["ham_holdout"]["curves"]
    lines = ["## Curva de α", "", "Macro-F1 de PAD (primaria, 3 clases) y, al lado, el macro-F1 del holdout de "
             f"HAM ya commiteado. {doc['ham_holdout']['scope']}", "",
             *header(["Modelo", "α", "Pesos", "PAD macro-F1 [IC 95 %]", "HAM holdout macro-F1 [IC 95 %]",
                      "Config HAM"])]
    for model, points in doc["curves"].items():
        for alpha, role in points.items():
            h = ham[model]
            ham_cells = ([_arm_cell(h["points"][alpha], "macro_f1", 0), h["points"][alpha]["config"]]
                         if h["available"] else ["sin curva", "—"])
            lines.append(row([model, alpha_label(alpha), f"`{role}`",
                              _arm_cell(arms[role], "macro_f1", doc["bootstrap"]["n_resamples"]), *ham_cells]))
    notes = [f"- {model}: {h['note']}" for model, h in ham.items() if not h["available"]]
    return lines + ([""] + notes if notes else [])


def _readings(doc: dict, n: int) -> list[str]:
    lines = []
    for name, entry in doc["readings"].items():
        lines += ["", f"## {READING_LABELS[name]}", "", f"{entry['n_images']} imágenes, {entry['n_patients']} "
                  "pacientes.", "", *_arms(entry, n, entry["outside_rate"]), "", *_contrasts(entry["paired"], n)]
    return lines


def _strata(doc: dict, n: int) -> list[str]:
    strata = doc["fst_strata"]
    lines = ["", "## Estratos por fototipo (lectura primaria, exploratorio, sin veredicto)", "",
             f"Regla (b) de docs/30 §6: el valor puntual siempre; con alguna remuestra no finita el IC queda "
             f"\"no calculable\" con k/{n}. No se descarta ninguna remuestra.", "", strata["note"]]
    for name, entry in strata["bins"].items():
        lines += ["", f"### {name}", ""]
        if not entry["n_images"]:
            lines.append(f"Sin imágenes ({entry['power']}).")
            continue
        power = "" if entry["power"] == "ok" else f" Rótulo: {entry['power']}."
        lines += [f"{entry['n_images']} imágenes, {entry['n_patients']} pacientes.{power}", "",
                  *_arms(entry, n), "", *_contrasts(entry["paired"], n)]
    return lines


def _agreement(doc: dict) -> list[str]:
    lines = ["", "## Control: puntos compartidos con la primera pasada", "",
             *header(["Pesos", "Rol en la primera pasada", "Máx. |Δ probabilidad|"])]
    for role, entry in doc["first_pass_agreement"].items():
        lines.append(row([f"`{role}`", f"`{entry['first_pass_role']}`", f"{entry['max_abs_diff']:.2e}"]))
    return lines


from dermssl.display_nomenclature import presentation  # noqa: E402


@presentation
def render(doc: dict) -> str:
    n = doc["bootstrap"]["n_resamples"]
    lines = [
        "# PAD-UFES-20: curva WiSE-FT de A* (docs/30 §8)", "",
        "Generado por `scripts/build_pad_ufes_wiseft_table.py` desde `pad_ufes_wiseft.json`; no editar a mano "
        "(`--check` lo compara y `tests/test_build_pad_ufes_wiseft_table.py` lo corre en CI). "
        f"`code_revision` {doc['code_revision']}, registro `{doc['registration_commit']}`.", "",
        f"{doc['scope']} Protocolo: {doc['protocol']}. Bootstrap pareado agrupado por paciente, semilla "
        f"{doc['bootstrap']['rng_seed']}, {n} remuestras.", "",
        "## Veredictos (registrados en docs/30 §8)", "", *_w1(doc["verdicts"]["PAD-W1"]), "",
        *_w2(doc["verdicts"]["PAD-W2"]), "", *_curve(doc), *_readings(doc, n), *_strata(doc, n),
        *_agreement(doc), "", "## Datos", "", f"- {addendum.DATASET}", f"- {pu.THESIS_FST_SENTENCE}.",
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
