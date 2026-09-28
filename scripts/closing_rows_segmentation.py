"""Closing-table rows for segmentation and tone: S-diag, S4-UNet, Pau's fixes and FST."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_segmentation_table as seg  # noqa: E402
import closing_table_rules as r  # noqa: E402
from build_segmentation_table import _count, _num, _signed  # noqa: E402

RES = seg.RES
REFS = {"isic2018": "ISIC 2018", "ham10000": "HAM10000"}
STUDENT = {ref: seg.FINAL / f"evaluacion_{ref}" / "summary.json" for ref in REFS}
SDIAG = RES / "2026-09-28-sdiag"
WEIGHTS = {"public": "público", "astar": "A\\* s42", "bstar": "B\\* s42"}
METHODS = {"primary": "PCA 224 (primario)", "kmeans": "k-means (sec. 1)", "pca448": "PCA 448 (sec. 2)"}
S4 = {"A2": RES / "2026-09-28-s4-a2", "A2*": RES / "2026-09-28-s4-a2-astar", "A1": RES / "2026-09-28-s4-a1"}
A1_STATS = S4["A1"] / "pseudo_masks_stats.json"
R0 = seg.PSEUDO["r0 publicada"]
FST = RES / "2026-09-28-fst-a2"
FST_CMP, FST_ASSUMPTION = FST / "comparison_vs_published.json", FST / "assumptions.json"
# docs/27 S-diag: "alcanza o supera la cobertura de CCAM (~0,15)"; within
# 0,05 of student_r3 it is reported as a training-free alternative.
SDIAG_MIN_COVERAGE = 0.15
SDIAG_ALTERNATIVE_GAP = 0.05


def sdiag_path(ref: str, weights: str, method: str) -> Path:
    return SDIAG / f"evaluacion_{ref}_{weights}_{method}" / "summary.json"


SDIAG_ALL = tuple(sdiag_path(ref, w, m) for m in METHODS for w in WEIGHTS for ref in REFS)


def mean_ci(m: dict, digits: int = 4) -> str:
    return f"{_num(m['mean'], digits)} [{_num(m['ci95_low'], digits)}; {_num(m['ci95_high'], digits)}]"


def delta_ci(m: dict, digits: int = 4) -> str:
    return f"{_signed(m['delta'], digits)} [{_signed(m['ci95_low'], digits)}; {_signed(m['ci95_high'], digits)}]"


def _student(d: dict) -> str:
    return " / ".join(mean_ci(d[STUDENT[ref]]["metrics"]["dice"]) for ref in REFS)


def _sdiag_key(path: Path) -> tuple[str, str, str]:
    """``(ref, weights, method)`` from ``evaluacion_<ref>_<weights>_<method>``."""
    _, ref, weights, method = path.parent.name.split("_", 3)
    return ref, weights, method


def _below_student(d: dict, path: Path) -> bool:
    ref = _sdiag_key(path)[0]
    return r.below_without_overlap(d[path]["metrics"]["dice"], d[STUDENT[ref]]["metrics"]["dice"])


def sdiag_dice(d: dict) -> tuple[str, str]:
    below = {p: _below_student(d, p) for p in SDIAG_ALL}
    primary = [p for p in SDIAG_ALL if _sdiag_key(p)[2] == "primary"]
    close = sum(d[STUDENT[_sdiag_key(p)[0]]]["metrics"]["dice"]["mean"] - d[p]["metrics"]["dice"]["mean"]
                < SDIAG_ALTERNATIVE_GAP for p in primary)
    parts = "; ".join(f"{WEIGHTS[w]} " + " / ".join(mean_ci(d[sdiag_path(ref, w, 'primary')]['metrics']['dice'])
                                                   for ref in REFS) for w in WEIGHTS)
    text = (f"Dice primario [IC 95 %], ISIC 2018 / HAM10000: {parts}. student_r3: {_student(d)}. "
            f"Por debajo de student_r3 sin solape de IC: {sum(below.values())} de {len(below)} resúmenes "
            f"(IC no pareados; detalle abajo). A menos de {_num(SDIAG_ALTERNATIVE_GAP, 2)} de student_r3: "
            f"{close} de {len(primary)} (primario).")
    return text, r.met(all(below[p] for p in primary))


def sdiag_coverage(d: dict) -> tuple[str, str]:
    cov = {(w, ref): d[sdiag_path(ref, w, "primary")]["metrics"] for w in WEIGHTS for ref in REFS}
    parts = "; ".join(f"{WEIGHTS[w]} {REFS[ref]} {mean_ci(m['pred_coverage'])} (referencia "
                      f"{_num(m['ref_coverage']['mean'])})" for (w, ref), m in cov.items())
    ok = all(m["pred_coverage"]["ci95_low"] >= SDIAG_MIN_COVERAGE for m in cov.values())
    above = sum(m["pred_coverage"]["ci95_low"] > m["ref_coverage"]["ci95_high"] for m in cov.values())
    return (f"Cobertura predicha media [IC 95 %], método primario: {parts}. Umbral registrado: "
            f"≥ {_num(SDIAG_MIN_COVERAGE, 2)} (límite inferior del IC). Por encima de la cobertura de "
            f"referencia, sin solape de IC: {above} de {len(cov)}."), r.met(ok)


def _paired(d: dict, folder: Path, name: str, ref: str) -> dict:
    return d[folder / f"paired_vs_{name}_{ref}.json"]["metrics"]


def _pair_text(d: dict, folder: Path, name: str, label: str) -> str:
    parts = []
    for ref in REFS:
        m = _paired(d, folder, name, ref)
        parts += [f"Dice {REFS[ref]} {delta_ci(m['dice'])} ({r.paired_verdict(m['dice']['ci95_low'], m['dice']['ci95_high'])})",
                  f"HD95 {REFS[ref]} {delta_ci(m['hd95'], 2)} "
                  f"({r.paired_verdict(m['hd95']['ci95_low'], m['hd95']['ci95_high'], higher_is_better=False)})"]
    return f"Δ pareado contra {label}: " + "; ".join(parts) + "."


def s4_sources(key: str, pairs: tuple[str, ...]) -> tuple[Path, ...]:
    folder = S4[key]
    return (tuple(folder / f"evaluacion_{ref}" / "summary.json" for ref in REFS)
            + tuple(folder / f"paired_vs_{name}_{ref}.json" for name in pairs for ref in REFS))


def s4_row(key: str, pairs: dict[str, str]):
    folder = S4[key]

    def build(d: dict) -> tuple[str, str]:
        dice = " / ".join(mean_ci(d[folder / f"evaluacion_{ref}" / "summary.json"]["metrics"]["dice"])
                          for ref in REFS)
        pair_text = " ".join(_pair_text(d, folder, name, label) for name, label in pairs.items())
        isic = _paired(d, folder, "student", "isic2018")["dice"]
        text = (f"Dice [IC 95 %] ISIC 2018 / HAM10000: {dice}. {pair_text} Umbral registrado: límite "
                f"inferior del Δ Dice ISIC contra student_r3 ≥ −{_num(r.S4_MAX_DICE_LOSS, 2)}.")
        return text, r.met(r.s4_threshold_met(isic))
    return build


def a1_coverage(d: dict) -> tuple[str, str]:
    a1, r0 = d[A1_STATS], d[R0]
    low, high = seg.S1_COVERAGE_BAND
    cov = a1["mean_foreground_fraction"]

    def pct(x: float) -> str:
        return f"{_num(100 * x, 1)} %"
    text = (f"Cobertura media {_num(cov)} (r0 {_num(r0['mean_foreground_fraction'])}); banda de referencia "
            f"{_num(low, 2)}-{_num(high, 2)}. Excluidas {pct(a1['excluded_fraction'])} (r0 "
            f"{pct(r0['excluded_fraction'])}): por área {_count(a1['excluded_by_area'])} (r0 "
            f"{_count(r0['excluded_by_area'])}), sin región {_count(a1['excluded_no_region'])} (r0 "
            f"{_count(r0['excluded_no_region'])}). Píxeles ignorados {pct(a1['mean_ignore_fraction'])} (r0 "
            f"{pct(r0['mean_ignore_fraction'])}). Sin IC: estadístico agregado de las pseudo-máscaras.")
    return text, seg.s1_verdict(cov)


def pseudo_row(name: str):
    path, cache = seg.PSEUDO[name], seg.CACHE.get(name)

    def build(d: dict) -> tuple[str, str]:
        r0, cov = d[R0]["mean_foreground_fraction"], d[path]["mean_foreground_fraction"]
        base = (f"Cobertura media {_num(cov, 5)} (r0 {_num(r0, 5)}, Δ {_signed(cov - r0)}); sin región "
                f"{_count(d[path]['excluded_no_region'])} (r0 {_count(d[R0]['excluded_no_region'])}). Sin IC: "
                "estadístico agregado.")
        if name == "S1 unión":
            low, high = seg.S1_COVERAGE_BAND
            return base + f" Umbral: cobertura en {_num(low, 2)}-{_num(high, 2)}.", seg.s1_verdict(cov)
        gain = f" Umbral: Δ ≥ {_num(seg.MIN_COVERAGE_GAIN, 2)}."
        if cache is None or name == "S2(a) sin pelo":
            return base + gain, seg.gain_verdict(cov, r0)
        marked, painted = seg._hair(d[cache])
        sanity = seg.hair_sanity(marked, painted)
        text = (base + f" (i) marcadas {_num(100 * marked, 1)} % (umbral < {100 * seg.HAIR_MARKED_MAX:.0f} %), "
                f"fracción pintada {_num(painted)} (umbral < {_num(seg.HAIR_PAINTED_MAX, 2)}); (ii){gain}"
                + ("" if sanity else " Falla (i): (ii) no se interpreta."))
        return text, r.met(sanity and cov - r0 >= seg.MIN_COVERAGE_GAIN)
    return build


def pseudo_sources(name: str) -> tuple[Path, ...]:
    return (R0, seg.PSEUDO[name]) + ((seg.CACHE[name],) if name in seg.CACHE else ())


def fst_a2(d: dict) -> tuple[str, str]:
    pool = d[FST_CMP]["scopes"]["pool"]
    tone, fst = pool["tone_distribution"], pool["FST_distribution"]

    def top(dist: dict) -> str:
        return _num(max(abs(b["diff_prop"]) for b in dist["bins"].values()), 5)
    text = (f"Pool (n = {_count(pool['n']['b'])}): con tono asignado {_count(tone['total']['a'])} → "
            f"{_count(tone['total']['b'])}; con FST asignado {_count(fst['total']['a'])} → "
            f"{_count(fst['total']['b'])}. Mayor |Δ proporción| por bin: tono {top(tone)}, FST {top(fst)}. "
            "Sin IC: no hay contraste registrado sobre la distribución. Supuesto: "
            f"`{d[FST_ASSUMPTION]['assumption']}`.")
    return text, r.MET


def fst_flagged(d: dict) -> tuple[str, str]:
    scopes = d[FST_CMP]["scopes"]
    parts = "; ".join(f"{name} {_count(s['flagged']['b']['n'])} de {_count(s['n']['b'])} "
                      f"(con artefacto: {_count(s['flagged']['b']['n_artifact_detected'])}; confianza < 1: "
                      f"{_count(s['flagged']['b']['n_confidence_below_1'])})"
                      for name, s in scopes.items() if s["flagged"]["b"] is not None)
    published = sum(s["flagged"]["a"] is not None for s in scopes.values())
    text = (f"Registros marcados, manifiesto con máscaras de A2: {parts}. El manifiesto publicado "
            f"({d[FST_CMP]['labels']['a']}) los reporta en {published} de {len(scopes)} ámbitos.")
    return text, r.met(all(s["flagged"]["b"] is not None for s in scopes.values()))


def rows() -> list[r.RowSpec]:
    prof, pau = "Profesor (docs/27 §3.2.2)", "Pau (docs/26"
    student = tuple(STUDENT.values())
    return [
        r.RowSpec(prof, "S-diag: DINOv2 segmenta sin entrenar; predicción: Dice primario < student_r3",
                  "PCA/k-means sobre los tokens de parche de DINOv2 público, A\\* y B\\* s42, "
                  "3 métodos × 2 referencias (docs/27 FASE 1, S-diag)", SDIAG_ALL + student, sdiag_dice,
                  (SDIAG,) + student),
        r.RowSpec(prof, "S-diag: cobertura ≥ la de CCAM (~0,15)", "Mismas corridas, método primario",
                  tuple(sdiag_path(ref, w, "primary") for w in WEIGHTS for ref in REFS), sdiag_coverage,
                  (SDIAG,)),
        r.RowSpec(f"{prof} · {pau} §5.1)", "S4-UNet A2: decoder sobre DINOv2 público congelado, pseudo-máscaras r0",
                  "Decoder DPT entrenado con las pseudo-máscaras de student_r3 (docs/27, S4-UNet)",
                  s4_sources("A2", ("student",)), s4_row("A2", {"student": "student_r3"})),
        r.RowSpec(prof, "S4-UNet A2 con el encoder A\\* s42", "Mismo decoder y costo, encoder A\\* s42 (docs/27)",
                  s4_sources("A2*", ("student", "a2_public")),
                  s4_row("A2*", {"student": "student_r3", "a2_public": "A2 público"})),
        r.RowSpec(f"{prof} · {pau} §5.1)", "S4-UNet A1: la saliencia de DINOv2 reemplaza a MoCo→CCAM",
                  "Saliencia S-diag en el mismo `uncertainty_module.py`, mismo decoder (docs/27)",
                  s4_sources("A1", ("student", "a2")), s4_row("A1", {"student": "student_r3", "a2": "A2"})),
        r.RowSpec("Profesor (docs/27 §3.2.1) · Pau (docs/26 §5.2)", "S5: DINO maestro → U-Net estudiante (destilación)",
                  "No activada: docs/27 la condiciona a que el despliegue liviano sea requisito", (), None),
        r.RowSpec(f"{pau} §2.4)", "S1: unión de regiones en `select_central_region`",
                  "Pseudo-máscaras r0 regeneradas con `--region-mode union` (docs/27 FASE 2)",
                  pseudo_sources("S1 unión"), pseudo_row("S1 unión")),
        r.RowSpec(f"{pau} §3.4)", "S2(a): remoción de pelo (black-hat + TELEA) en la etapa `cache`",
                  "Caché sin pelo y saliencia con el CCAM ya entrenado (docs/27 FASE 2)",
                  pseudo_sources("S2(a) sin pelo"), pseudo_row("S2(a) sin pelo")),
        r.RowSpec(f"{pau} §3.4)", "S2': remoción de pelo con filtro por forma",
                  "Filtro de componentes alargadas sobre el black-hat (docs/27 FASE 2)",
                  pseudo_sources("S2' filtro por forma"), pseudo_row("S2' filtro por forma")),
        r.RowSpec(f"{pau} §5.3)", "Pseudo-máscaras de A1: cobertura en la banda de referencia",
                  "Stats de `make_pseudo_masks.py --source dino` contra r0 (docs/27, S4-UNet)",
                  (A1_STATS, R0), a1_coverage),
        r.RowSpec(f"{pau} §6.4, paso 1)", "S3: reportar aparte los registros con `artifact_detected` o `confidence < 1`",
                  "`fst_pseudo_manifest.py` los reporta por separado (docs/27 FASE 2, S3)", (FST_CMP,), fst_flagged),
        r.RowSpec(f"{pau} §6.4, paso 1) · docs/27", "FST con la caché sin pelo, comparado con el original",
                  "Registrado en docs/27 (FST con la caché sin pelo); sin manifiesto en el repo", (), None),
        r.RowSpec(f"{pau} §6.2) · docs/27 S4-UNet", "FST con las máscaras de A2 (segmentador candidato)",
                  "Probabilidades de A2 sobre el pool → `make_pseudo_masks.py` con las banderas de `pseudo_r2` "
                  "→ manifiesto FST (docs/27)", (FST_CMP, FST_ASSUMPTION), fst_a2),
        r.RowSpec(f"{pau} §7.3)", "C3: ablación Plain vs Proposed-oráculo",
                  "Reemplazada por el rediseño v7 (docs/28 §5.2); el oráculo queda como fase 2", (), None),
    ]


def sdiag_detail(d: dict) -> list[str]:
    lines = ["| Pesos | Método | Referencia | Dice [IC 95 %] | HD95 [IC 95 %] | Cobertura predicha | "
             "Cobertura referencia | Veredicto vs student_r3 |", "|---|---|---|---|---|---|---|---|"]
    for p in SDIAG_ALL:
        ref, weights, method = _sdiag_key(p)
        m = d[p]["metrics"]
        verdict = "por debajo, sin solape" if _below_student(d, p) else "IC solapado o por encima"
        lines.append(f"| {WEIGHTS[weights]} | {METHODS[method]} | {REFS[ref]} | {mean_ci(m['dice'])} | "
                     f"{mean_ci(m['hd95'], 2)} | {_num(m['pred_coverage']['mean'])} | "
                     f"{_num(m['ref_coverage']['mean'])} | {verdict} |")
    return lines
