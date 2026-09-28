"""Build resultados/segmentacion/TABLA_SEGMENTACION.md from committed JSON.

Two parts: the final student (student_r3) against the reference masks of
ISIC 2018 and HAM10000, and the v7 pseudo-mask variants (S1, S2(a), S2')
read against the thresholds registered in docs/27 before each run. Every
verdict is computed here from the JSON numbers; ``--check`` fails in CI if
the table drifts.

    python scripts/build_segmentation_table.py            # write
    python scripts/build_segmentation_table.py --check    # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = Path("plan-segmentacion") / "camino-b-cnn" / "resultados"
FINAL, V2, V7 = RES / "2026-09-21-final", RES / "2026-09-21-v2" / "evaluaciones", RES / "2026-09-27-v7"
TARGET = REPO / "resultados" / "segmentacion" / "TABLA_SEGMENTACION.md"
REFERENCES = {"isic2018": "ISIC 2018", "ham10000": "HAM10000"}
STUDENT = {
    "student_r3": {ref: FINAL / f"evaluacion_{ref}" / "summary.json" for ref in REFERENCES},
    "student_r3 + TTA": {ref: V2 / f"final_{ref}_tta" / "summary.json" for ref in REFERENCES},
}
PSEUDO = {
    "r0 publicada": FINAL / "pseudo_r0" / "pseudo_masks_stats.json",
    "S1 central (reproduce r0)": V7 / "s1_central_pseudo_masks_stats.json",
    "S1 unión": V7 / "s1_union_pseudo_masks_stats.json",
    "S2(a) sin pelo": V7 / "s2a_pseudo_masks_stats.json",
    "S2' filtro por forma": V7 / "s2prime_pseudo_masks_stats.json",
}
CACHE = {"S2(a) sin pelo": V7 / "s2a_cache_info.json", "S2' filtro por forma": V7 / "s2prime_cache_info.json"}
PRECISION = {"bf16": V7 / "precision_bf16_pseudo_masks_stats.json",
             "fp32": V7 / "precision_fp32_pseudo_masks_stats.json"}
DIRECTION = {"dice": "↑ mayor es mejor", "iou": "↑ mayor es mejor", "hd95": "↓ menor es mejor"}
METRIC_NAMES = {"dice": "Dice", "iou": "IoU", "hd95": "HD95 (px)"}

# Thresholds registered in docs/27-plan-v7-mejoras-ssl.md (FASE 2) before
# each run, and never tuned afterwards:
#   S1: mean_foreground_fraction rises towards 0,21-0,27.
#   S2 and S2' (ii): coverage gain against r0 of at least 0,02.
#   S2' sanity (i): under 50 % of images marked with hair AND mean painted
#   pixel fraction under 0,03; if (i) fails, S2' is not interpreted.
#   Precision control: |coverage bf16 - fp32| < 0,005 on 500 fixed images
#   makes the published r0 a valid comparison.
S1_COVERAGE_BAND = (0.21, 0.27)
MIN_COVERAGE_GAIN = 0.02
HAIR_MARKED_MAX = 0.50
HAIR_PAINTED_MAX = 0.03
PRECISION_MAX_ABS_DELTA = 0.005
MET, NOT_MET = "cumplida", "no cumplida"
NOT_INTERPRETED = "no se interpreta (falla la sanity (i))"


def s1_verdict(coverage: float) -> str:
    low, high = S1_COVERAGE_BAND
    return MET if low <= coverage <= high else NOT_MET


def gain_verdict(coverage: float, r0: float) -> str:
    return MET if coverage - r0 >= MIN_COVERAGE_GAIN else NOT_MET


def hair_sanity(marked: float, painted: float) -> bool:
    return marked < HAIR_MARKED_MAX and painted < HAIR_PAINTED_MAX


def s2prime_verdict(coverage: float, r0: float, marked: float, painted: float) -> str:
    return gain_verdict(coverage, r0) if hair_sanity(marked, painted) else NOT_INTERPRETED


def precision_valid(bf16: float, fp32: float) -> bool:
    return abs(fp32 - bf16) < PRECISION_MAX_ABS_DELTA


def _num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _signed(x: float, digits: int = 4) -> str:
    return ("−" if x < 0 else "+") + _num(abs(x), digits)


def _count(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _read(path: Path) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def load_student(paths: dict) -> dict:
    """Variants whose summary files exist; a missing file drops only its row."""
    out = {}
    for variant, refs in paths.items():
        found = {ref: _read(p) for ref, p in refs.items() if (REPO / p).is_file()}
        if found:
            out[variant] = found
    return out


def load_sources() -> dict:
    return {"student": load_student(STUDENT),
            "pseudo": {name: _read(p) for name, p in PSEUDO.items()},
            "cache": {name: _read(p) for name, p in CACHE.items()},
            "precision": {name: _read(p) for name, p in PRECISION.items()}}


def student_section(student: dict) -> list[str]:
    metas = [s["meta"] for refs in student.values() for s in refs.values()]
    checkpoints = sorted({"/".join(Path(m["checkpoint"]).parts[-2:]) for m in metas})
    lines = ["## Estudiante final contra las máscaras de referencia", "",
             f"Checkpoint `{'`, `'.join(checkpoints)}`, umbral "
             f"{', '.join(sorted({_num(m['threshold'], 1) for m in metas}))}, resolución "
             f"{', '.join(sorted({str(m['resolution']) for m in metas}))}. HD95 en píxeles a esa "
             "resolución. Las máscaras de referencia solo se leen para evaluar.", "",
             "| Variante | Referencia | Métrica | Dirección | Media [IC 95 %] | Mediana | n |",
             "|---|---|---|---|---|---|---:|"]
    for variant, refs in student.items():
        for ref, summary in refs.items():
            for key, name in METRIC_NAMES.items():
                m, d = summary["metrics"][key], 2 if key == "hd95" else 4
                lines.append(f"| {variant} | {REFERENCES[ref]} | {name} | {DIRECTION[key]} | "
                             f"{_num(m['mean'], d)} [{_num(m['ci95_low'], d)}; {_num(m['ci95_high'], d)}] | "
                             f"{_num(m['median'], d)} | {_count(m['n'])} |")
    return lines


def _hair(cache: dict) -> tuple[float, float]:
    hair = cache["hair_removal"]
    return hair["images_with_hair"] / cache["count"], hair["mean_fraction"]


def pseudo_verdicts(src: dict, r0: float) -> dict[str, tuple[str, str]]:
    """``{variant: (registered threshold, verdict)}`` computed from the JSON."""
    pseudo, cache = src["pseudo"], src["cache"]
    cov = {name: s["mean_foreground_fraction"] for name, s in pseudo.items()}
    base = pseudo["r0 publicada"]
    same = all(pseudo["S1 central (reproduce r0)"][k] == base[k]
               for k in ("mean_foreground_fraction", "excluded", "excluded_no_region", "excluded_by_area"))
    band = f"{_num(S1_COVERAGE_BAND[0], 2)}-{_num(S1_COVERAGE_BAND[1], 2)}"
    gain = f"Δ ≥ {_num(MIN_COVERAGE_GAIN, 2)}"
    return {
        "r0 publicada": ("referencia", "—"),
        "S1 central (reproduce r0)": ("igual a r0", "reproduce r0" if same else "NO reproduce r0"),
        "S1 unión": (f"cobertura en {band}", s1_verdict(cov["S1 unión"])),
        "S2(a) sin pelo": (gain, gain_verdict(cov["S2(a) sin pelo"], r0)),
        "S2' filtro por forma": (
            f"(i) marcadas < {100 * HAIR_MARKED_MAX:.0f} % y pintada < {_num(HAIR_PAINTED_MAX, 2)}; (ii) {gain}",
            s2prime_verdict(cov["S2' filtro por forma"], r0, *_hair(cache["S2' filtro por forma"]))),
    }


def pseudo_section(src: dict) -> list[str]:
    r0 = src["pseudo"]["r0 publicada"]["mean_foreground_fraction"]
    verdicts = pseudo_verdicts(src, r0)
    lines = ["## Pseudo-máscaras v7 contra los umbrales registrados (docs/27, FASE 2)", "",
             "Cobertura = `mean_foreground_fraction`. Δ contra la r0 publicada. Umbrales y "
             "veredictos calculados por el script desde los JSON.", "",
             "| Variante | Cobertura media | Δ vs r0 | Sin región | Excluidas por área | "
             "Excluidas total | Umbral registrado | Veredicto |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for name, s in src["pseudo"].items():
        threshold, verdict = verdicts[name]
        lines.append(f"| {name} | {_num(s['mean_foreground_fraction'], 5)} | "
                     f"{_signed(s['mean_foreground_fraction'] - r0)} | {_count(s['excluded_no_region'])} | "
                     f"{_count(s['excluded_by_area'])} | {_count(s['excluded'])} | {threshold} | {verdict} |")
    return lines


def precision_section(precision: dict) -> list[str]:
    bf16, fp32 = (precision[k]["mean_foreground_fraction"] for k in ("bf16", "fp32"))
    ok = precision_valid(bf16, fp32)
    return ["## Control de precisión (500 imágenes fijas)", "",
            "| Saliencia r0 | Cobertura media | Excluidas |", "|---|---:|---:|",
            f"| bf16 (publicada, GPU) | {_num(bf16, 6)} | {_count(precision['bf16']['excluded'])} |",
            f"| fp32 (recalculada, CPU) | {_num(fp32, 6)} | {_count(precision['fp32']['excluded'])} |", "",
            f"Δ fp32 − bf16 = {_signed(fp32 - bf16, 6)}; umbral |Δ| < {_num(PRECISION_MAX_ABS_DELTA, 3)}: "
            + ("la r0 publicada vale como comparación." if ok else
               "la r0 publicada NO vale como comparación; usar la r0 recalculada en fp32.")]


def hair_section(cache: dict) -> list[str]:
    lines = ["## Remoción de pelo: S2(a) contra S2'", "",
             "| Variante | Imágenes marcadas / total | % marcadas | Fracción media pintada | "
             "Inpaint masivo |", "|---|---:|---:|---:|---:|"]
    for name, info in cache.items():
        hair = info["hair_removal"]
        heavy = hair.get("heavy_inpaint_count")
        lines.append(f"| {name} | {_count(hair['images_with_hair'])} / {_count(info['count'])} | "
                     f"{_num(100 * hair['images_with_hair'] / info['count'], 1)} % | "
                     f"{_num(hair['mean_fraction'])} | {'—' if heavy is None else _count(heavy)} |")
    return lines + ["", "«—» = el `cache_info.json` de esa variante no registra el conteo."]


def render(src: dict) -> str:
    lines = ["# Segmentación: estudiante final y pseudo-máscaras v7", "",
             "Generado por `scripts/build_segmentation_table.py` desde JSON commiteados. "
             "No editar a mano: `--check` lo compara en CI.", "",
             *student_section(src["student"]), "", *pseudo_section(src), "",
             *precision_section(src["precision"]), "", *hair_section(src["cache"])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(load_sources())
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} esta desactualizado: corre scripts/build_segmentation_table.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
