"""Write resultados/clasificacion/c1/README.md from the C1 grid output.

Generated, never transcribed: the table is rendered from c1_cells.json and
c1_bootstrap.json, and ``--check`` exits non-zero when the committed file
would change -- which is what CI runs. A hand-edited number cannot survive.

The verdict column reuses ``build_results_readme._verdict``, which READS the
stored direction instead of inferring a side from the sign. That is the only
thing standing between a lower-is-better ECE and a verdict printed backwards,
so it is imported, not reimplemented.

    python scripts/build_c1_readme.py            # write
    python scripts/build_c1_readme.py --check    # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_results_readme import _ci, _fmt, _verdict  # noqa: E402

C1_DIR = REPO / "resultados" / "clasificacion" / "c1"
CONDITION_ORDER = ("none", "bbox_crop", "neutral_bg", "mask_pool")
CONDITION_LABEL = {
    "none": "(0) sin mascara",
    "bbox_crop": "(a) recorte a la caja",
    "neutral_bg": "(b) fondo gris",
    "mask_pool": "(c) pooling en mascara",
}

#: The registered prediction is evaluated in balanced accuracy, as it was
#: written; the headline table stays in macro-F1 for comparability with v6.
#: Both fixed in docs/27 §6 before any C1 number existed.
PREDICTION_METRIC = "balanced_accuracy"
PREDICTION_CONTRAST = ("dinov2_public_vitb14__mask_pool", "dinov2_public_vitb14__none")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=C1_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def cells_table(cells: dict) -> list[str]:
    lines = [
        "| Backbone | Condicion | macro-F1 | bal-acc | AUC | ECE | Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    for cell in sorted(cells.values(), key=lambda c: (c["backbone"], order.get(c["condition"], 9))):
        m = cell["metrics"]
        gate = "pasa" if cell.get("gate_passes") else "FRENA"
        lines.append(
            f"| `{cell['backbone']}` | {CONDITION_LABEL.get(cell['condition'], cell['condition'])} "
            f"| {_fmt(m.get('macro_f1'))} | {_fmt(m.get('balanced_accuracy'))} "
            f"| {_fmt(m.get('auc'))} | {_fmt(m.get('ece'))} | {gate} |"
        )
    return lines


def contrasts_table(bootstrap: dict) -> list[str]:
    rows = bootstrap.get("results", [])
    if not rows:
        return ["_Sin contrastes todavia._"]
    lines = [
        "| Brazo | Control | Metrica | Delta | IC 95 % | Veredicto |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['arm']}` | `{row['control']}` | {row['metric']} "
            f"| {_fmt(row['point_delta'], 4)} | {_ci(row)} | {_verdict(row)} |"
        )
    return lines


def prediction_verdict(bootstrap: dict) -> str:
    arm, control = PREDICTION_CONTRAST
    for row in bootstrap.get("results", []):
        if row["arm"] == arm and row["control"] == control and row["metric"] == PREDICTION_METRIC:
            return (
                f"Contraste `{arm}` contra `{control}` en {PREDICTION_METRIC}: "
                f"delta {_fmt(row['point_delta'], 4)}, IC {_ci(row)} -> {_verdict(row)}."
            )
    return "Contraste de la prediccion registrada: todavia no disponible."


def cross_sections(cross: dict | None) -> list[str]:
    """Between-backbone contrasts and the crop-retention strata, when computed."""
    if not cross:
        return []
    parts = [
        "## Contrastes entre backbones (pregunta del profe)",
        "",
        "Pareado sobre las mismas imagenes de validacion. Primero el DINOv2 publico en "
        "cada condicion contra el propuesto en (0), con (0) contra (0) como referencia "
        "que reproduce el signo de la v6; despues los dos backbones en la MISMA "
        "condicion, con segmentacion en ambos lados.",
        "",
        *contrasts_table(cross),
        "",
        f"## Estratificado por lesion conservada tras el recorte (umbral {cross['retained_threshold']})",
        "",
        "Confusor declarado en docs/27: (a) no pasa por el recorte central, asi que en "
        "lesiones cerca del borde ve mas lesion que (0), (b) y (c).",
        "",
    ]
    for stratum in cross["stratified"]:
        parts += [f"### `{stratum['stratum']}` (n = {stratum['n']})", ""]
        if stratum["cells"]:
            parts += [
                "| Celda | macro-F1 | bal-acc | AUC | ECE |",
                "|---|---|---|---|---|",
                *[
                    f"| `{name}` | {_fmt(m['macro_f1'])} | {_fmt(m['balanced_accuracy'])} "
                    f"| {_fmt(m['auc'])} | {_fmt(m['ece'])} |"
                    for name, m in stratum["cells"].items()
                ],
                "",
                *contrasts_table(stratum),
                "",
            ]
        parts += [f"- {reason}" for reason in stratum["not_contrasted"]]
        parts.append("")
    return parts


from dermssl.display_nomenclature import presentation  # noqa: E402


@presentation
def render(cells: dict, bootstrap: dict, cross: dict | None = None) -> str:
    not_contrasted = bootstrap.get("not_contrasted", [])
    parts = [
        "# C1: la segmentacion como entrada del clasificador",
        "",
        "Generado por `scripts/build_c1_readme.py` desde `c1_cells.json` y "
        "`c1_bootstrap.json`. No editar a mano: `--check` lo compara en CI.",
        "",
        "Tabla principal en macro-F1 (como la v6). La prediccion registrada se evalua "
        "en balanced accuracy, tal como se registro (docs/27 §6). Se reportan siempre "
        "las cuatro metricas. ECE: menor es mejor.",
        "",
        "## Celdas",
        "",
        *cells_table(cells),
        "",
        "## Contrastes pareados contra la condicion (0) del mismo backbone",
        "",
        f"Bootstrap pareado, {bootstrap.get('n_resamples', '?')} remuestras, "
        f"n_val = {bootstrap.get('n_val', '?')}. Una celda que no paso su gate no entra.",
        "",
        *contrasts_table(bootstrap),
        "",
        "## Prediccion registrada",
        "",
        prediction_verdict(bootstrap),
        "",
        *cross_sections(cross),
    ]
    if not_contrasted:
        parts += ["## No contrastadas", "", *[f"- {reason}" for reason in not_contrasted], ""]
    return "\n".join(parts)


def main(argv=None) -> int:
    args = parse_args(argv)
    cells = json.loads((args.dir / "c1_cells.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((args.dir / "c1_bootstrap.json").read_text(encoding="utf-8"))
    cross_path = args.dir / "c1_cross_backbone.json"
    cross = json.loads(cross_path.read_text(encoding="utf-8")) if cross_path.exists() else None
    content = render(cells, bootstrap, cross)
    target = args.dir / "README.md"
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != content:
            print(f"{target} esta desactualizado: corre scripts/build_c1_readme.py")
            return 1
        return 0
    target.write_text(content, encoding="utf-8")
    print(f"escrito {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
