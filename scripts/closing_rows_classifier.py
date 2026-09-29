"""Closing-table rows for the classifier: C1 (v6), C1-v7 and the v7 predictions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import closing_table_rules as r  # noqa: E402
from build_segmentation_table import _signed  # noqa: E402

from dermssl import v7_seeds as seeds  # noqa: E402

CLF = Path("resultados") / "clasificacion"
C1_BOOT, C1_CROSS = CLF / "c1" / "c1_bootstrap.json", CLF / "c1" / "c1_cross_backbone.json"
C1V7_STUDENT = CLF / "c1_v7" / "student_r3" / "c1_bootstrap.json"
C1V7_A2 = CLF / "c1_v7" / "a2" / "c1_bootstrap.json"
A2_VS_STUDENT = CLF / "c1_v7" / "a2_vs_student_cells.json"
FINAL = {seed: seeds.final_eval_path(seed) for seed in seeds.REGISTERED_SEEDS}
PUBLIC, PROPOSED_V6 = "dinov2_public_vitb14", "ssl_proposed_vitb14"
# C1-v7 cells were run for s42 and s43 only; the labels cover every registered seed.
BACKBONE = {PUBLIC: "público", **{f"v7_{arm.lower()}star_s{seed}": f"{arm}\\* s{seed}"
                                  for arm in ("A", "B") for seed in seeds.REGISTERED_SEEDS}}
CONDITION = {"none": "(0) sin máscara", "bbox_crop": "(a) recorte", "neutral_bg": "(b) fondo gris",
             "mask_pool": "(c) pooling"}
PREDICTIONS = {
    "P1": "P1 (primaria): el plain anclado A\\* no pierde contra DINOv2 público",
    "P2": "P2 (primaria): el término de tono (B\\* contra A\\*) mejora ECE sin perder macro-F1",
    "P3": "P3 (primaria, objetivo de la tesis): B\\* no pierde contra DINOv2 público y mejora ECE",
    "P1-NI": "P1-NI (secundaria): no inferioridad estricta de A\\* contra el público",
    "P3-IC": "P3-IC (secundaria): B\\* le gana al público en macro-F1",
}
METRIC = {"macro_f1": "macro-F1", "ece": "ECE", "balanced_accuracy": "bal-acc"}


def ci(d: dict, key: str = "point_delta") -> str:
    return f"{_signed(d[key])} [{_signed(d['ci95_low'])}; {_signed(d['ci95_high'])}]"


def _find(boot: dict, arm: str, control: str, metric: str) -> dict:
    return next(x for x in boot["results"]
                if x["arm"] == arm and x["control"] == control and x["metric"] == metric)


def _cell(arm: str) -> str:
    backbone, condition = arm.split("__")
    return f"{BACKBONE.get(backbone, f'`{backbone}`')} {CONDITION[condition]}"


def _direction(x: dict) -> str:
    return {"arm_better": r.BETTER, "control_better": r.WORSE}.get(x["direction"], r.SAME)


def _mask_counts(results: list[dict], metric: str = "macro_f1") -> tuple[list[dict], dict[str, int]]:
    rows = [x for x in results if x["metric"] == metric]
    counts = {v: sum(_direction(x) == v for x in rows) for v in (r.BETTER, r.WORSE, r.SAME)}
    return rows, counts


def _count_text(rows: list[dict], counts: dict[str, int]) -> str:
    return " · ".join(f"{v}: {counts[v]}" for v in (r.BETTER, r.WORSE, r.SAME)) + f" (de {len(rows)})"


def c1_v6(d: dict) -> tuple[str, str]:
    boot, cross = d[C1_BOOT], d[C1_CROSS]
    pred = _find(boot, f"{PUBLIC}__mask_pool", f"{PUBLIC}__none", "balanced_accuracy")
    rows, counts = _mask_counts(boot["results"])
    reach = _find(cross, f"{PUBLIC}__mask_pool", f"{PROPOSED_V6}__none", "macro_f1")
    base = _find(cross, f"{PUBLIC}__none", f"{PROPOSED_V6}__none", "macro_f1")
    text = (f"Predicción registrada (bal-acc, público (c) − (0)): Δ {ci(pred)}. "
            f"Macro-F1, cada condición contra (0) en el mismo backbone: {_count_text(rows, counts)}. "
            f"Macro-F1 público (c) − propuesto v6 (0): {ci(reach)}; referencia público (0) − propuesto "
            f"v6 (0): {ci(base)}.")
    return text, r.met(pred["direction"] == "arm_better")


def c1_v7(source: Path):
    def build(d: dict) -> tuple[str, str]:
        results = d[source]["results"]
        rows, counts = _mask_counts(results)
        worse = [f"{_cell(x['arm'])} {ci(x)}" for x in rows if _direction(x) == r.WORSE]
        text = (f"Macro-F1 contra (0) del mismo backbone: {_count_text(rows, counts)}."
                + (f" Empeoran: {'; '.join(worse)}." if worse else ""))
        return text, r.met(r.mask_helps(results))
    return build


def family_verdict(verdicts: list[str]) -> str:
    """One verdict for a family of paired contrasts; a mixed family has no
    registered reading, so it fails loudly instead of picking a side."""
    found = set(verdicts)
    if {r.BETTER, r.WORSE} <= found:
        raise ValueError("familia mixta: necesita una lectura registrada antes de resumirla")
    return r.BETTER if r.BETTER in found else r.WORSE if r.WORSE in found else r.SAME


def a2_vs_student(d: dict) -> tuple[str, str]:
    cells = [c for c in d[A2_VS_STUDENT]["results"] if not c["cell"].endswith("__none")]
    verdicts = [r.paired_verdict(c["ci95_low"], c["ci95_high"]) for c in cells]
    deltas = [c["delta_a2_minus_student"] for c in cells]
    favourable = sum(x > 0 for x in deltas)
    text = (f"Macro-F1 A2 − student_r3 en la misma celda: {verdicts.count(r.SAME)} de {len(cells)} "
            f"con IC que incluye el cero; Δ puntual entre {_signed(min(deltas))} y {_signed(max(deltas))}. "
            f"Nota descriptiva, no pre-registrada y con celdas no independientes (comparten val y "
            f"backbone): {favourable} de {len(cells)} Δ puntuales favorecen a A2.")
    return text, family_verdict(verdicts)


def _evidence(pred: dict) -> str:
    return "; ".join(f"{METRIC[m]} {ci(e)}" for m, e in pred["evidence"].items())


def v7_prediction(key: str):
    def build(d: dict) -> tuple[str, str]:
        preds = {seed: d[path]["predictions"][key] for seed, path in FINAL.items()}
        verdicts = {seed: p["verdict"] for seed, p in preds.items()}
        per_seed = " ".join(f"s{seed}: {_evidence(p)} → {p['verdict']}." for seed, p in preds.items())
        criterion = preds[seeds.BASE_SEED]["criterion"].replace("*", "\\*")
        text = f"Criterio: {criterion}. {per_seed}"
        if len(set(verdicts.values())) > 1:
            text += " Las semillas no coinciden: no robusto a la semilla."
        return text, r.seeds_verdict(verdicts)
    return build


def rows() -> list[r.RowSpec]:
    prof = "Profesor (docs/27 §3.1)"
    specs = [
        r.RowSpec(prof, "Comparativa justa: DINOv2 público + segmentación (C1, v6)",
                  "C1: las máscaras de student_r3 en la entrada de los 6 backbones v6, 4 condiciones, "
                  "probe `logreg`, bootstrap pareado (docs/27 §6, FASE 1)",
                  (C1_BOOT, C1_CROSS), c1_v6),
        r.RowSpec("Profesor (docs/27 §3.2.3) · docs/28 §5.2",
                  "La máscara ayuda al clasificador final (C1-v7 primario, máscaras de student_r3)",
                  "Misma grilla sobre A\\*, B\\* (semillas 42 y 43) y el público; regla: alguna condición "
                  "gana a (0) con IC > 0 en las dos semillas del brazo (docs/28 §5.2)",
                  (C1V7_STUDENT,), c1_v7(C1V7_STUDENT)),
        r.RowSpec("docs/28 §5.2 (secundario)", "La máscara ayuda al clasificador final con las máscaras de A2",
                  "Misma grilla y regla con las máscaras de S4-UNet A2 (docs/28 §5.2)",
                  (C1V7_A2,), c1_v7(C1V7_A2)),
        r.RowSpec("docs/28 §5.2 (secundario)", "A2 contra student_r3 como máscara del clasificador, por celda",
                  "Bootstrap pareado del macro-F1 sobre las mismas predicciones de val (docs/28 §5.2)",
                  (A2_VS_STUDENT,), a2_vs_student),
    ]
    specs += [r.RowSpec("Profesor (docs/27 §3.3) · docs/28 §5", PREDICTIONS[k],
                        "Finales A\\* y B\\* (SimSiam anclado sobre DINOv2 ViT-B), evaluación única en val "
                        f"con {seeds.count_word(len(FINAL))} semillas (docs/28 §4.1, §5.1)",
                        tuple(FINAL.values()), v7_prediction(k)) for k in PREDICTIONS]
    specs.append(r.RowSpec("Profesor (docs/27 §3.3)", "D1: generalización cross-dominio (HAM10000 → ISIC 2019/2020)",
                           "Registrado en docs/27 FASE 5; no se corrió", (), None))
    return specs
