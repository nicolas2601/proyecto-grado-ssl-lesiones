"""CLI for dermssl.tone_stratified_eval: stratify a classifier by continuous ITA.

EVALUATION ONLY. Reads saved predictions and an ITA manifest; transforms no
image and calls neither tone_augment_lab.augment_skin_tone nor
tone_augment_lab_v2.augment_skin_tone_v2. See the module docstring for why
this exists (the G3 fairness gate is built on Fitzpatrick buckets
docs/63-instrumento-ita-alcance-y-limites.md section 4 already retired).

Inputs:

* ``--npz``: a ``.npz`` with ``y_true`` (n,) and one or more probability
  matrices (n, 7) in HAM10000_CLASSES order -- the format
  resultados/clasificacion/v7/final_eval.probas.npz already uses.
* ``--variant``: which probability matrix to read (e.g. "proposed", "plain",
  "public" for v7). Declared in the output so nobody reads a report without
  knowing which model it scored.
* ``--ita-manifest``: JSONL, one record per prediction row -- see
  dermssl.tone_stratified_eval.read_ita_manifest for the schema. Does not
  need every row to have a measurable ITA; coverage is reported, not
  required to be complete.

Usage::

    PYTHONPATH=src python scripts/eval_tone_stratified.py \\
        --npz resultados/clasificacion/v7/final_eval.probas.npz \\
        --variant proposed \\
        --ita-manifest /path/to/val_ita.jsonl

Writes a plain-text report (no emojis) plus the JSON summary under
``resultados/segmentacion/v8/tone_stratified_eval/`` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "resultados" / "segmentacion" / "v8" / "tone_stratified_eval"
sys.path.insert(0, str(REPO / "src"))

from dermssl.tone_stratified_eval import (  # noqa: E402
    build_report,
    load_npz_predictions,
    read_ita_manifest,
)


def _format_stratum_table(stratum: dict) -> list[str]:
    lines = []
    for key, entry in stratum.items():
        if entry.get("status") == "SKIPPED":
            lines.append(f"     {key}: SKIPPED ({entry['reason']})")
            continue
        lines.append(
            f"     {key}: n={entry['count']}  macro_f1={entry['macro_f1']:.4f}  "
            f"ci95=[{entry.get('macro_f1_ci95_low', float('nan')):.4f}, "
            f"{entry.get('macro_f1_ci95_high', float('nan')):.4f}]"
        )
        if entry.get("classes_absent"):
            lines.append(f"       clases ausentes (0 muestras, contribuyen F1=0): {entry['classes_absent']}")
        if entry.get("classes_below_min_n"):
            lines.append(
                f"       clases con n insuficiente (no promediar su F1 como confiable): "
                f"{entry['classes_below_min_n']}"
            )
    return lines


def _format_class_composition(composition: dict) -> list[str]:
    lines = [f"   n split completo={composition['n_full']}  n medible={composition['n_measurable']}"]
    lines.append("   clase       frac. en split    frac. en medible")
    for class_name in composition["fraction_full"]:
        full = composition["fraction_full"][class_name]
        sub = composition["fraction_measurable"][class_name]
        sub_str = f"{sub:.4f}" if sub is not None else "n/a"
        lines.append(f"     {class_name:6s}      {full:.4f}            {sub_str}")
    return lines


def render_report(report: dict) -> str:
    lines = []
    lines.append("EVALUACION ESTRATIFICADA POR ITA CONTINUO (reemplazo de G3)")
    lines.append(f"variante: {report['variant']}  clases: {','.join(report['classes'])}")
    lines.append("")

    lines.append("1. Cobertura")
    cov = report["coverage"]
    lines.append(f"   predicciones totales: {cov['n_total_predictions']}")
    lines.append(
        f"   filas con registro en el manifiesto de ITA: {cov['n_in_ita_manifest']} "
        f"({cov['manifest_coverage_fraction']:.4f})" if cov["manifest_coverage_fraction"] is not None
        else "   sin registros en el manifiesto de ITA"
    )
    lines.append(
        f"   ITA medible dentro del manifiesto: {cov['n_ita_measurable']} "
        f"({cov['ita_measurable_fraction_of_manifest']:.4f})"
        if cov["ita_measurable_fraction_of_manifest"] is not None
        else "   sin filas medibles"
    )
    lines.append(f"   por ita_status: {cov['by_ita_status']}")
    lines.append(f"   por mask_source: {cov['by_mask_source']}")
    if cov["n_in_ita_manifest"] < cov["n_total_predictions"]:
        lines.append(
            "   COBERTURA PARCIAL: todo lo que sigue es condicional al subconjunto "
            "con registro de ITA, no al total de predicciones."
        )
    lines.append("")

    lines.append("1b. Composicion de clases: split completo contra subconjunto medible")
    lines.append(
        "    Si el subconjunto medible concentra o vacia una clase frente al split, "
        "ninguna metrica de abajo se lee sin esta tabla al lado."
    )
    lines.extend(_format_class_composition(report["class_composition"]))
    lines.append("")

    lines.append("2. macro-F1 por mask_source (nunca promediado en silencio)")
    by_mask = report["macro_f1_by_mask_source"]
    if by_mask:
        lines.extend(_format_stratum_table(by_mask))
    else:
        lines.append("   sin filas medibles para estratificar")
    lines.append("")

    lines.append("3. macro-F1 por tercil DE LA COHORTE (percentiles del ITA medido, no Groh ni Corbin)")
    by_tercile = report["macro_f1_by_cohort_tercile"]
    if by_tercile:
        lines.extend(_format_stratum_table(by_tercile))
    else:
        lines.append("   sin filas medibles para estratificar")
    lines.append("")

    lines.append("4. Dependencia continua: correlacion de acierto contra grados de ITA")
    corr = report["correctness_vs_ita_spearman"]
    if corr["rho"] is not None:
        lines.append(
            f"   Spearman rho={corr['rho']:.4f}  p={corr['p_value']:.4g}  "
            f"ci95=[{corr['ci95_low']:.4f}, {corr['ci95_high']:.4f}]  n={corr['n']}"
        )
    else:
        lines.append(f"   n insuficiente para correlacion (n={corr['n']})")
    lines.append("")

    lines.append("5. Recorrido alcanzable por imagen (si el manifiesto lo trae)")
    reach = report["reachable_ita_span_percentiles"]
    if reach:
        lines.append(
            f"   mediana={reach['p50']:.3f}  p05={reach['p5']:.3f}  p25={reach['p25']:.3f}  "
            f"p75={reach['p75']:.3f}  p95={reach['p95']:.3f}  (n={reach['n']})"
        )
    else:
        lines.append("   el manifiesto no trae reachable_ita_span")
    lines.append("")

    lines.append("6. macro-F1 del split completo, solo de referencia")
    full = report["macro_f1_full_split_reference"]
    lines.append(f"   macro_f1={full['macro_f1']:.4f}  n={full['n']}")
    lines.append(f"   {full['caveat']}")
    lines.append("")

    if report["strata_with_insufficient_n"]:
        lines.append(
            f"ADVERTENCIA: estratos con n insuficiente (SKIPPED), no promediados: "
            f"{report['strata_with_insufficient_n']}"
        )
    if report["strata_with_thin_classes"]:
        lines.append("ADVERTENCIA: clases con n insuficiente DENTRO de un estrato OK (su F1 no es confiable):")
        for stratum_key, classes in report["strata_with_thin_classes"].items():
            lines.append(f"   {stratum_key}: {classes}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--ita-manifest", type=Path, required=True)
    parser.add_argument("--min-stratum-n", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    y_true, proba = load_npz_predictions(args.npz, args.variant)
    ita_rows = read_ita_manifest(args.ita_manifest)
    report = build_report(y_true, proba, ita_rows, variant=args.variant, min_n=args.min_stratum_n)

    text = render_report(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(text)
    print()
    print(f"written: {args.output_dir / 'report.txt'}")
    print(f"written: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
