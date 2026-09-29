"""Closing-table rows for the exploratory blocks T1-T7 (docs/27, docs/28, docs/30, docs/31).

Each verdict is the one the experiment's own JSON records: this module reads
it and never re-derives it from the numbers. A row whose JSON verdicts
disagree fails loudly instead of picking one. Every number and commit comes
from the same JSON.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_segmentation_table as seg  # noqa: E402
import closing_rows_classifier as clf  # noqa: E402
import closing_table_rules as r  # noqa: E402
import pad_ufes_fst_validation as fv  # noqa: E402
from build_segmentation_table import _count, _num, _signed  # noqa: E402
from closing_rows_segmentation import REFS, delta_ci  # noqa: E402

from dermssl import luminance_threshold as lt  # noqa: E402

RES, CLF = seg.RES, Path("resultados") / "clasificacion"
T1 = Path("resultados") / "fst" / "pad_ufes_t1" / "pad_ufes_fst_validation.json"
T2 = RES / "2026-09-28-t2-luminance-dice" / "t2_luminance_dice.json"
T3 = CLF / "c1_v7" / "t3_luminance" / "t3_c1_luminance.json"
T4 = CLF / "t4_sog" / "t4_final_eval.json"
T6 = RES / "2026-09-29-t6-luminance-threshold" / "t6_luminance_threshold.json"
T7 = RES / "2026-09-29-t7-compare" / "t7_compare.json"
T7_R0 = RES / "2026-09-29-t7-r0-check" / "t7_r0_tercile_check.json"

T1_CONDITIONS = {"whole_image": "imagen completa", "masked": "fondo por Otsu"}
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
T3_SEGMENTERS = {"student_r3": "student_r3", "a2": "A2"}
T4_WEIGHTS = {"public": "público", "A": "A\\*", "B": "B\\*"}
T4_GAP = "luminance_gap_dark_light"
TERCILE = {"dark": "oscuro", "medium": "medio", "light": "claro"}


def registered_verdict(words) -> str:
    """The single verdict a row's JSON records; mixed or unknown words fail."""
    found = set(words)
    if len(found) != 1:
        raise ValueError(f"veredictos distintos en una misma fila: {sorted(found)}")
    word = found.pop()
    if word not in r.REGISTERED:
        raise ValueError(f"veredicto fuera del vocabulario registrado: {word!r}")
    return word


def _commits(provenance: dict) -> str:
    return (f"Registro `{provenance['registration_commit'][:7]}`, "
            f"código `{provenance['code_revision'][:7]}`.")


def _mean_ci(x: float, bounds: list[float]) -> str:
    return f"{_num(x)} [{_num(bounds[0])}; {_num(bounds[1])}]"


def _t1_condition(name: str, m: dict) -> str:
    base = m["majority_baseline"]
    return (f"{T1_CONDITIONS[name]} (n = {_count(m['n_evaluable'])}): accuracy {_mean_ci(m['accuracy'], m['ci95']['accuracy'])} "
            f"contra {_num(base['accuracy'])} de la baseline de mayoría (fototipo {ROMAN[int(base['label'])]}), "
            f"kappa lineal {_signed(m['linear_kappa'])} [{_signed(m['ci95']['linear_kappa'][0])}; "
            f"{_signed(m['ci95']['linear_kappa'][1])}]")


def t1(d: dict) -> tuple[str, str]:
    doc = d[T1]
    dark = Counter(rec["actual_fst"] for rec in doc["records"] if rec["actual_fst"] >= 4)
    conditions = "; ".join(_t1_condition(name, m) for name, m in doc["metrics"].items())
    verdicts = " ".join(f"`{k}`: {v}." for k, v in doc["verdicts"].items())
    text = (f"{conditions}. {verdicts} Fototipos IV, V y VI en PAD: "
            f"{', '.join(_count(dark[k]) for k in (4, 5, 6))} imágenes (descriptivo, sin veredicto). "
            f"IC bootstrap agrupado por paciente, descriptivos. {_commits(doc)}")
    return text, registered_verdict(doc["verdicts"].values())


def t2(ref: str):
    def build(d: dict) -> tuple[str, str]:
        doc = d[T2]
        cells = {name: s["dark_minus_rest"]["dice"] for name, s in doc["datasets"][ref]["segmenters"].items()}
        parts = "; ".join(f"{name} {delta_ci(c)} (n {_count(c['n_a'])} / {_count(c['n_b'])}) → {c['verdict']}"
                          for name, c in cells.items())
        text = (f"Δ Dice tercil oscuro − resto [IC 95 %]: {parts}. Regla registrada: cumplida si el IC queda "
                f"entero bajo {_signed(doc['params']['clear_margin'], 2)}; distinguible pero no clara si queda "
                f"bajo 0. {_commits(doc['provenance'])}")
        return text, registered_verdict(c["verdict"] for c in cells.values())
    return build


def _t3_cell(doc: dict, key: dict) -> tuple[str, dict]:
    contrast = doc["segmenters"][key["segmenter"]][key["backbone"]]["dark"]["contrasts"][key["condition"]]
    label = (f"{T3_SEGMENTERS[key['segmenter']]} · {clf.BACKBONE[key['backbone']]} · "
             f"{clf.CONDITION[key['condition']]}")
    return label, contrast["macro_f1"]


def t3(d: dict) -> tuple[str, str]:
    doc = d[T3]
    keys, pred = doc["key_cells"], doc["prediction"]
    cells = [_t3_cell(doc, k) for k in keys]
    counts = " · ".join(f"{word}: {n}" for word, n in Counter(k["verdict"] for k in keys).items())
    deltas = [c["point_delta"] for _, c in cells]
    top_label, top = max(cells, key=lambda x: x[1]["ci95_low"])
    mult = doc["multiplicity"]
    text = (f"Tercil oscuro de val (n = {_count(doc['coverage']['dark'])}), {len(keys)} celdas clave contra "
            f"(0) sin máscara, Δ macro-F1 pareado: {counts}; Δ puntual entre {_signed(min(deltas))} y "
            f"{_signed(max(deltas))}. Límite inferior más alto: {top_label} {clf.ci(top)}. Celdas con "
            f"«mejora»: {len(pred['improves'])}; con «empeora»: {pred['n_worsens']}. Multiplicidad declarada: "
            f"{mult['n_contrasts']} contrastes, ~{_num(mult['expected_ci_without_zero_by_chance'], 1)} IC sin "
            f"cero esperables por azar. {_commits(doc['provenance'])}")
    return text, registered_verdict([pred["verdict"]])


def _t4_contrast(doc: dict, weight: str, metric: str) -> dict:
    return next(p for p in doc["paired"]
                if (p["arm"], p["control"], p["metric"]) == (f"{weight}_sog", f"{weight}_id", metric))


def t4(d: dict) -> tuple[str, str]:
    doc = d[T4]
    f1 = "; ".join(f"{label} {clf.ci(_t4_contrast(doc, w, 'macro_f1'))}" for w, label in T4_WEIGHTS.items())
    gap = "; ".join(f"{label} {clf.ci(_t4_contrast(doc, w, T4_GAP))}" for w, label in T4_WEIGHTS.items())
    verdicts = {k: r.met(v["holds"]) for k, v in doc["verdicts"].items()}
    text = (f"Δ SoG (p = {doc['sog_p']}) − sin SoG en el protocolo fijo, val (n = {_count(doc['n_eval'])}), IC 95 % pareado. "
            f"Macro-F1: {f1}. Brecha absoluta de macro-F1 entre los terciles de luminancia oscuro y claro "
            f"(menor es mejor): {gap}. " + " ".join(f"{k}: {v}." for k, v in verdicts.items())
            + f" {_commits(doc)}")
    return text, registered_verdict(verdicts.values())


def t6(d: dict) -> tuple[str, str]:
    ham, isic = (d[T6]["datasets"][ref] for ref in ("ham10000", "isic2018"))
    ev, v = ham["evaluation"], ham["evaluation"]["verdict"]
    thresholds = ", ".join(f"{TERCILE[k]} {_num(t['threshold'], 2)}" for k, t in ham["thresholds"].items())
    gap = ev["gap"]
    text = (f"HAM10000 val (oscuro {_count(gap['n_dark'])}, resto {_count(gap['n_rest'])}); umbrales elegidos en "
            f"ajuste: {thresholds} (fijo {_num(d[T6]['params']['fixed'], 2)}). Brecha Dice oscuro − resto: fijo "
            f"{delta_ci(gap['gap_fixed'])}, adaptado {delta_ci(gap['gap_adapted'])}; reducción "
            f"{delta_ci(gap['reduction'])} → {v['gap']}. Δ Dice global {delta_ci(ev['global'])} → {v['global']}. "
            f"ISIC 2018 (descriptivo): reducción {delta_ci(isic['evaluation']['gap']['reduction'])}, Δ Dice "
            f"global {delta_ci(isic['evaluation']['global'])}. {_commits(d[T6]['provenance'])}")
    return text, registered_verdict([v["prediction"]])


def t7(d: dict) -> tuple[str, str]:
    ham, teacher = d[T7]["datasets"]["ham10000"], d[T7_R0]["analysis"]
    v, red = ham["verdict"], ham["reduction"]
    text = (f"HAM10000 (oscuro {_count(ham['n_dark'])}, resto {_count(ham['n_rest'])}). Brecha Dice oscuro − "
            f"resto: A2 {delta_ci(ham['gap']['A2'])}, T7 {delta_ci(ham['gap']['T7'])}. R(T7) = |G(A2)| − "
            f"|G(T7)| {delta_ci(red['T7'])} → {v['gap']}; Δ Dice global T7 − A2 {delta_ci(ham['global']['T7'])} "
            f"→ {v['global']}; R(T7) − R(A2 semilla 43) {delta_ci(ham['seed_contrast'])} → {v['seed']}. "
            f"Controles: R(A2 semilla 43) {delta_ci(red['A2-s43'])}; R(solo aumento) {delta_ci(red['T7-aug'])}. "
            f"Chequeo del maestro r0 (mismas imágenes): G(r0) {delta_ci(teacher['teacher_gap_binary'])}, G(A2) "
            f"{delta_ci(teacher['a2_gap_same_images'])}; adenda requerida: "
            f"{'sí' if teacher['addendum_required'] else 'no'}. {_commits(d[T7]['provenance'])}")
    return text, registered_verdict([v["conclusion"]])


GOAL = (f"≥ {_num(lt.GAP_MARGIN, 2)} sin perder más de {_num(-lt.GLOBAL_TOLERANCE, 3)} de Dice global "
        "en HAM10000")


def rows() -> list[r.RowSpec]:
    pau = "Pau (docs/27, exploratorio)"
    specs = [
        r.RowSpec("docs/30 §5 (exploratorio)", f"T1: la {r.ITA_LABEL} contra el fototipo de PAD-UFES-20",
                  f"La {r.ITA_LABEL} por imagen (completa y con fondo por Otsu) contra la columna `fitspatrick` "
                  "de PAD-UFES-20; predicción: no supera la baseline de mayoría (`FST-P1`) y el kappa lineal "
                  f"queda bajo {_num(fv.SLIGHT_KAPPA, 2)} (`FST-P2`)", (T1,), t1,
                  (T1, T1.with_name("pad_ufes_fst_validation.fst_table.md"))),
    ]
    specs += [r.RowSpec(pau, f"T2 ({REFS[ref]}): Dice de A2 y student_r3 en el tercil oscuro de luminancia, "
                             "claramente menor que el resto",
                        "Dice por tercil de luminancia desde los `per_image.csv` ya evaluados, bootstrap no "
                        "pareado estratificado", (T2,), t2(ref)) for ref in ("ham10000", "isic2018")]
    specs += [
        r.RowSpec("Pau (docs/28, exploratorio)",
                  "T3: en el tercil oscuro de luminancia, recorte y pooling de C1-v7 tampoco mejoran a sin máscara",
                  "Probabilidades de val ya guardadas por C1-v7 (máscaras de student_r3 y de A2), Δ pareado "
                  "por tercil de luminancia", (T3,), t3),
        r.RowSpec("docs/31 (exploratorio)",
                  "T4: Shades of Gray antes del encoder no cambia el macro-F1 ni la brecha entre terciles "
                  "de luminancia",
                  "Lectura única de val con el protocolo v7.1 fijo, público, A\\* y B\\*; P1-P3 registradas en "
                  "docs/31", (T4,), t4),
        r.RowSpec(pau, "T6: un umbral de máscara por tercil de luminancia sobre A2 achica la brecha de Dice "
                       f"{GOAL}",
                  "Umbral elegido por tercil en el ajuste de HAM10000 y evaluado en HAM val, sin reentrenar "
                  "(análogo barato de arXiv 2510.06655)", (T6,), t6),
        r.RowSpec(pau, "T7: A2 reentrenado con oscurecimiento y sobremuestreo del tercil oscuro de luminancia "
                       f"achica la brecha {GOAL} y se separa del ruido de semilla",
                  "Tres brazos reentrenados (T7, A2 semilla 43, solo aumento) más el chequeo del maestro r0",
                  (T7, T7_R0), t7),
    ]
    return specs
