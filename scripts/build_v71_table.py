"""Render the v7.1 single-evaluation table from its JSON; ``--check`` for CI.

    python scripts/build_v71_table.py resultados/clasificacion/v71/v71_final_eval.json \
        --out resultados/clasificacion/v71/README.md [--check]

The extra contrasts (``scripts/v71_extra_contrasts.py``) are rendered from the
``extra_contrasts.json`` next to the document when it exists (or ``--extra``).
The BLAS-threads limitation (``dermssl.probe_blas_limitation``) is rendered
from ``../t4`` diagnostics next to the v7.1 directory when they exist (or ``--t4``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import probe_blas_limitation as pbl  # noqa: E402
from dermssl import v71_plan as plan  # noqa: E402

LABELS = {"A71": "A* v7.1", "B71": "B* v7.1", "public_A71": "Público, protocolo de A* v7.1",
          "public_B71": "Público, protocolo de B* v7.1", "A_v7": "A* v7 (primario)",
          "B_v7": "B* v7 (primario)", "public_v7": "Público v7 (primario)",
          "public71": "Público, su mejor protocolo", "A_v7_s43": "A* v7 s43", "B_v7_s43": "B* v7 s43"}
MAIN = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "accuracy": "Acc", "auc": "AUC",
        "ece": "ECE", "mel_recall": "Sens. mel", "mel_auc": "AUC mel"}
OPERATING = {"spec80": "esp. 80 %", "spec90": "esp. 90 %", "sens95": "sens. 95 %"}
EXTRA_METRICS = {"mel_sens_at_thr_spec80": "Sens. mel (esp. 80 %)", "mel_sens_at_thr_spec90": "Sens. mel (esp. 90 %)",
                 "mel_spec_at_thr_sens95": "Esp. mel (sens. 95 %)", "mel_to_nv_rate": "Tasa mel→nv",
                 "macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc"}
CONTRAST = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "auc": "AUC", "ece": "ECE",
            "mel_recall": "Sens. mel", "mel_auc": "AUC mel"}


def _num(x) -> str:
    return "n/d" if x is None else f"{x:.4f}".replace(".", ",")


def _ci(pair) -> str:
    return f"[{_num(pair[0])}; {_num(pair[1])}]"


def _cell(result: dict, metric: str) -> str:
    return f"{_num(result[metric])} {_ci(result['ci95'][metric])}"


def _main_table(doc: dict) -> list[str]:
    lines = ["| Lectura | Configuración | " + " | ".join(MAIN.values()) + " |",
             "|---|---|" + "---|" * len(MAIN)]
    for label in plan.VAL_LABELS:
        r = doc["results"][label]
        lines.append(f"| {LABELS[label]} | `{r['config']}` | "
                     + " | ".join(_cell(r, m) for m in MAIN) + " |")
    return lines


def _operating_table(doc: dict) -> list[str]:
    head = " | ".join(f"Sens. / esp. ({v})" for v in OPERATING.values())
    lines = ["", "## Puntos de operación de melanoma (umbral fijado en el holdout)", "",
             doc["note_operating_points"], "",
             f"| Lectura | {head} |", "|---|" + "---|" * len(OPERATING)]
    for label in plan.VAL_LABELS:
        r = doc["results"][label]
        cells = [f"{_cell(r, f'mel_sens_at_thr_{k}')} / {_cell(r, f'mel_spec_at_thr_{k}')}" for k in OPERATING]
        lines.append(f"| {LABELS[label]} | " + " | ".join(cells) + " |")
    return lines


def _contrast_table(doc: dict) -> list[str]:
    rows = {(r["arm"], r["control"], r["metric"]): r for r in doc["paired"]}
    lines = ["", "## Contrastes pareados (bootstrap, semilla 20260923)", "",
             "| Contraste | " + " | ".join(f"Δ {v}" for v in CONTRAST.values()) + " |",
             "|---|" + "---|" * len(CONTRAST)]
    for arm, control in plan.VAL_CONTRASTS:
        cells = []
        for metric in CONTRAST:
            r = rows[(arm, control, metric)]
            cells.append(f"{_num(r['point_delta'])} {_ci((r['ci95_low'], r['ci95_high']))}")
        lines.append(f"| {LABELS[arm]} − {LABELS[control]} | " + " | ".join(cells) + " |")
    return lines


def _long_contrasts(rows: list[dict]) -> list[str]:
    lines = ["| Contraste | Métrica | Brazo | Control | Δ | IC 95 % | Excluye 0 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {LABELS[r['arm']]} − {LABELS[r['control']]} | {EXTRA_METRICS[r['metric']]} | "
                     f"{_num(r['point_arm'])} | {_num(r['point_control'])} | {_num(r['point_delta'])} | "
                     f"{_ci((r['ci95_low'], r['ci95_high']))} | {'sí' if r['excludes_zero'] else 'no'} |")
    return lines


def _extra_tables(extra: dict) -> list[str]:
    nv = extra["mel_to_nv"]
    entries = ["| Lectura | mel→nv | Tasa |", "|---|---|---|"]
    entries += [f"| {LABELS[k]} | {e['count']} | {_num(e['rate'])} |" for k, e in nv["entries"].items()]
    return ["", "## Contrastes adicionales (`extra_contrasts.json`)", "", extra["note_verdict"], "",
            "### Melanoma a umbral propio (comparación justa)", "", extra["fair_mel"]["note"], "",
            *_long_contrasts(extra["fair_mel"]["contrasts"]), "",
            "### Melanomas clasificados como nevus", "", nv["note"], "", *entries, "",
            *_long_contrasts(nv["contrasts"]), "",
            "### Contra los primarios v7 s42 y v7 s43", "", extra["s43"]["note"], "",
            *_long_contrasts(extra["s43"]["contrasts"])]


def _limitation(facts: dict) -> list[str]:
    return ["", f"## {pbl.TITLE}", "", *pbl.body(facts)]


from dermssl.display_nomenclature import context_presentation  # noqa: E402


@context_presentation("v7")
def render(doc: dict, extra: dict | None = None, limitation: dict | None = None) -> str:
    phase2 = doc["selection"]["phase2"]
    mel = ", ".join(f"{arm} {_num(v)}" for arm, v in phase2["mel_sens_at_spec80_holdout"].items())
    lines = [
        f"# Lectura v7.1 ({doc['split']}, docs/29)", "",
        f"n = {doc['n_eval']}, `code_revision` {doc['code_revision']}. Generado desde "
        "`v71_final_eval.json`; no editar a mano (`--check` lo compara en CI). La selección se hizo "
        "en el holdout interno y validación se leyó una sola vez. El primario v7 no cambia.", "",
        *_main_table(doc), *_operating_table(doc), *_contrast_table(doc), "",
        f"Fase 2 (docs/29 §2.7 ítem 7): sensibilidad de mel a especificidad 80 % en el holdout: {mel}. "
        f"Habilitada para proponer: {'sí' if phase2['enabled'] else 'no'}.",
        *(_extra_tables(extra) if extra else []),
        *(_limitation(limitation) if limitation else []),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra", type=Path, default=None)
    parser.add_argument("--t4", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    extra_path = args.extra or args.document.parent / "extra_contrasts.json"
    extra = json.loads(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else None
    doc = json.loads(args.document.read_text(encoding="utf-8"))
    t4_dir = args.t4 or args.document.parent.parent / "t4"
    limitation = pbl.load_facts(t4_dir, doc) if pbl.available(t4_dir) else None
    text = render(doc, extra, limitation)
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
