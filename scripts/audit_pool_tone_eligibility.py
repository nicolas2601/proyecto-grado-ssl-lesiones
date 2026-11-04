"""Pool eligibility triage for the tone contrast arm: can this pool sustain it at all.

Built after the contrast arm died on the ISIC2019 + teacher-026 pool:
tone_augment_lab_v2's effective intervention rate there was 25.7% against the
80% floor of criterion 7 (docs/estado/clf-trainer-pareado-tono.md:140), and
restricting to the ITA-measurable subset was separately ruled out --
measurability correlates with the diagnostic class (Fisher OR 0.547,
p=0.0099; commit 9038662), so that subset is depleted in melanoma. Finding
this cost a day of hand analysis on a 33.620-row mirror that had already been
materialized. This script is that analysis, runnable BEFORE a mirror exists,
on a manifest alone.

It reports whether a pool CAN sustain the contrast; it fixes no new
threshold. Two thresholds already exist elsewhere and are only compared
against here, never redefined: MIN_B_FOR_ITA = 5.0
(dermssl.fitzpatrick) and the 80% floor of criterion 7. Nothing in this
script invents a third.

Does not import or call tone_augment_lab.augment_skin_tone or write a single
transformed pixel: this is a measurement tool over the frozen instrument
(fitzpatrick.estimate_fst) and the v2 solver's own reachability search,
nothing more.

Input: a manifest (JSONL, one object per line) with ``image_path`` and
``mask_path`` -- the same format scripts/make_tone_contact_sheet.py reads --
plus an OPTIONAL ``label`` field (any JSON-serializable value; the Fisher
test in section 4 treats it as a binary 0/1 or truthy/falsy class). Paths
resolve relative to the manifest's own directory when not absolute. Works
identically on a synthetic (e.g. S-SYNTH) or a real dermoscopy manifest --
this script does not know or care which.

Usage::

    PYTHONPATH=src python scripts/audit_pool_tone_eligibility.py \\
        --manifest /path/to/manifest.jsonl \\
        [--seed 42] [--policy-cap-degrees 25.0]

Writes a plain-text report (no emojis, paste-ready) plus a JSON summary under
``resultados/segmentacion/v8/pool_tone_eligibility/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "resultados" / "segmentacion" / "v8" / "pool_tone_eligibility"
sys.path.insert(0, str(REPO / "src"))

from dermssl.fitzpatrick import MIN_B_FOR_ITA, estimate_fst  # noqa: E402
from dermssl.tone_augment_lab_v2 import (  # noqa: E402
    MAX_POPULATION_SHIFT_FRACTION,
    _skin_lab_for_curve,
    sample_apply_draw,
    solve_ita_delta,
)

# Criterion 7, docs/estado/clf-trainer-pareado-tono.md:140-142: "La tasa
# efectiva de intervencion se mide y se publica. Si cae por debajo del 80%,
# se detiene antes de interpretar nada." Compared against here, not
# redefined.
CRITERION_7_FLOOR = 0.80


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    base = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            row["image_path"] = str(image_path if image_path.is_absolute() else base / image_path)
            row["mask_path"] = str(mask_path if mask_path.is_absolute() else base / mask_path)
            rows.append(row)
    return rows


def analyze_row(row: dict, *, seed: int, policy_cap_degrees: float | None) -> dict | None:
    """One manifest row's full eligibility profile, or ``None`` if its files are missing.

    Computes reachability INDEPENDENTLY of the 50/50 mixing draw (calling the
    solver directly, not ``augment_skin_tone_v2``) so the "without mixing"
    ceiling -- the number that actually decides feasibility -- is not
    entangled with whether this particular draw happened to select the
    image. The "with mixing" rate is then reconstructed from the same draw
    ``augment_skin_tone_v2`` would make, without running the full pipeline
    or touching a single pixel.
    """
    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])
    if not image_path.exists() or not mask_path.exists():
        return None
    img = _load_image(image_path)
    mask = _load_mask(mask_path)
    image_sha256 = _sha256_of_file(image_path)

    measurement = estimate_fst(img, mask=mask, border_mask=None)
    measurable = measurement.status == "ok"

    reachable_without_mix = False
    population_shift_ok = None
    reachable_ita_span = None
    if measurable and not measurement.fallback_used:
        skin_lab = _skin_lab_for_curve(img, mask, None)
        if skin_lab is not None:
            solved = solve_ita_delta(
                skin_lab, seed, 0, image_sha256, policy_cap_degrees=policy_cap_degrees
            )
            if solved is not None:
                reachable_ita_span = solved["reachable_ita_lo"] - solved["reachable_ita_hi"]
                population_shift_ok = solved["population_shift_fraction"] <= MAX_POPULATION_SHIFT_FRACTION
                reachable_without_mix = population_shift_ok

    mix_selected = sample_apply_draw(seed, 0, image_sha256) > 0.5
    applied_with_mix = reachable_without_mix and mix_selected

    return {
        "image_path": str(image_path),
        "lesion_fraction": float(mask.mean()),
        "b_median": measurement.b_median,
        "measurable": measurable,
        "status": measurement.status,
        "reachable_without_mix": reachable_without_mix,
        "population_shift_ok": population_shift_ok,
        "reachable_ita_span": reachable_ita_span,
        "applied_with_mix": applied_with_mix,
        "label": row.get("label"),
        "has_label": "label" in row and row["label"] is not None,
    }


def _percentiles(values: list[float], points=(5, 25, 50, 75, 95)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in points}


def _area_tercile(lesion_fraction: float, cuts: tuple[float, float]) -> int:
    if lesion_fraction <= cuts[0]:
        return 0
    if lesion_fraction <= cuts[1]:
        return 1
    return 2


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    b_values = [r["b_median"] for r in rows if np.isfinite(r["b_median"])]
    measurable = [r for r in rows if r["measurable"]]
    reachable = [r for r in rows if r["reachable_without_mix"]]
    applied = [r for r in rows if r["applied_with_mix"]]

    summary = {
        "n_total": n,
        "b_median_distribution": _percentiles(b_values) if b_values else None,
        "b_median_n_finite": len(b_values),
        "fraction_below_min_b_for_ita": (
            float(np.mean([v < MIN_B_FOR_ITA for v in b_values])) if b_values else None
        ),
        "measurable_rate": len(measurable) / n if n else None,
        "reachable_without_mix_rate": len(reachable) / n if n else None,
        "effective_intervention_rate_with_mix": len(applied) / n if n else None,
        "criterion_7_floor": CRITERION_7_FLOOR,
        "sustains_criterion_7": (len(applied) / n if n else 0.0) >= CRITERION_7_FLOOR,
        "no_reachable_among_measurable": sum(
            1 for r in measurable if not r["reachable_without_mix"] and r["population_shift_ok"] is not False
        ),
        "population_shift_rejections": sum(
            1 for r in rows if r["population_shift_ok"] is False
        ),
    }

    spans = [r["reachable_ita_span"] for r in rows if r["reachable_ita_span"] is not None]
    summary["reachable_ita_span_distribution"] = _percentiles(spans) if spans else None
    summary["reachable_ita_span_n"] = len(spans)

    labeled = [r for r in rows if r["has_label"]]
    summary["label_coverage_n"] = len(labeled)
    summary["label_coverage_fraction"] = len(labeled) / n if n else None
    if labeled:
        measurable_pos = sum(1 for r in labeled if r["measurable"] and r["label"])
        measurable_neg = sum(1 for r in labeled if r["measurable"] and not r["label"])
        unmeasurable_pos = sum(1 for r in labeled if not r["measurable"] and r["label"])
        unmeasurable_neg = sum(1 for r in labeled if not r["measurable"] and not r["label"])
        table = [[measurable_pos, measurable_neg], [unmeasurable_pos, unmeasurable_neg]]
        odds_ratio, p_value = fisher_exact(table)
        summary["measurability_vs_label"] = {
            "table_measurable_pos_neg": [measurable_pos, measurable_neg],
            "table_unmeasurable_pos_neg": [unmeasurable_pos, unmeasurable_neg],
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
            "positive_rate_measurable": measurable_pos / (measurable_pos + measurable_neg)
            if (measurable_pos + measurable_neg)
            else None,
            "positive_rate_unmeasurable": unmeasurable_pos / (unmeasurable_pos + unmeasurable_neg)
            if (unmeasurable_pos + unmeasurable_neg)
            else None,
        }

        fractions = sorted(r["lesion_fraction"] for r in labeled)
        if len(fractions) >= 3:
            cuts = (
                fractions[len(fractions) // 3],
                fractions[2 * len(fractions) // 3],
            )
            tercile_rows: dict[int, dict] = {0: {"pos_m": 0, "neg_m": 0, "pos_u": 0, "neg_u": 0}, 1: {"pos_m": 0, "neg_m": 0, "pos_u": 0, "neg_u": 0}, 2: {"pos_m": 0, "neg_m": 0, "pos_u": 0, "neg_u": 0}}
            for r in labeled:
                tercile = _area_tercile(r["lesion_fraction"], cuts)
                bucket = tercile_rows[tercile]
                key_m = "pos_m" if r["label"] else "neg_m"
                key_u = "pos_u" if r["label"] else "neg_u"
                if r["measurable"]:
                    bucket[key_m] += 1
                else:
                    bucket[key_u] += 1
            summary["measurability_vs_label_by_area_tercile"] = {
                str(t): {
                    "positive_rate_measurable": v["pos_m"] / (v["pos_m"] + v["neg_m"])
                    if (v["pos_m"] + v["neg_m"])
                    else None,
                    "positive_rate_unmeasurable": v["pos_u"] / (v["pos_u"] + v["neg_u"])
                    if (v["pos_u"] + v["neg_u"])
                    else None,
                }
                for t, v in tercile_rows.items()
            }
    return summary


def render_report(summary: dict, *, manifest_path: Path, seed: int, policy_cap_degrees) -> str:
    lines = []
    lines.append("TRIAJE DE ELEGIBILIDAD DE POOL PARA EL CONTRASTE DE TONO")
    lines.append(f"manifiesto: {manifest_path}")
    lines.append(f"n total: {summary['n_total']}  seed: {seed}  policy_cap_degrees: {policy_cap_degrees}")
    lines.append("")

    lines.append("1. Distribucion de b* de piel (fitzpatrick.estimate_fst)")
    if summary["b_median_distribution"]:
        d = summary["b_median_distribution"]
        lines.append(
            f"   mediana={d['p50']:.3f}  p05={d['p5']:.3f}  p25={d['p25']:.3f}  "
            f"p75={d['p75']:.3f}  p95={d['p95']:.3f}  (n={summary['b_median_n_finite']})"
        )
        lines.append(
            f"   fraccion bajo MIN_B_FOR_ITA={MIN_B_FOR_ITA}: {summary['fraction_below_min_b_for_ita']:.4f}"
        )
    else:
        lines.append("   sin pixeles de piel medibles en ningun caso (n=0)")
    lines.append("")

    lines.append("2. Tasa de medibilidad del ITA y techo de intervencion proyectado de v2")
    lines.append(f"   medible (status=ok): {summary['measurable_rate']:.4f}")
    lines.append(
        f"   alcanzable SIN sorteo de mezcla (techo real): {summary['reachable_without_mix_rate']:.4f}"
    )
    lines.append(
        f"   tasa efectiva proyectada CON sorteo de mezcla: {summary['effective_intervention_rate_with_mix']:.4f}"
    )
    lines.append(
        f"   rechazos por border_population_shift (MAX_POPULATION_SHIFT_FRACTION): "
        f"{summary['population_shift_rejections']}"
    )
    lines.append("")

    lines.append(f"3. Contraste contra criterio 7 (piso {CRITERION_7_FLOOR:.0%})")
    verdict = "SOSTIENE" if summary["sustains_criterion_7"] else "NO SOSTIENE"
    lines.append(
        f"   tasa efectiva con mezcla {summary['effective_intervention_rate_with_mix']:.4f} "
        f"contra piso {CRITERION_7_FLOOR:.2f}: {verdict}"
    )
    lines.append("")

    lines.append("4. Medibilidad contra etiqueta diagnostica")
    lines.append(
        f"   cobertura de etiqueta: {summary['label_coverage_n']}/{summary['n_total']} "
        f"({summary['label_coverage_fraction']:.4f})"
        if summary["label_coverage_fraction"] is not None
        else "   sin etiqueta en el manifiesto"
    )
    if summary["label_coverage_n"] < summary["n_total"]:
        lines.append(
            "   COBERTURA PARCIAL: lo que sigue es condicional al subconjunto con etiqueta, "
            "no al pool completo."
        )
    mv = summary.get("measurability_vs_label")
    if mv:
        lines.append(
            f"   tasa de positivos, medible: {mv['positive_rate_measurable']:.4f}  "
            f"no medible: {mv['positive_rate_unmeasurable']:.4f}"
        )
        lines.append(f"   Fisher exacto: odds ratio {mv['odds_ratio']:.4f}  p={mv['p_value']:.4g}")
        by_tercile = summary.get("measurability_vs_label_by_area_tercile")
        if by_tercile:
            lines.append("   por tercil de area de lesion (medible / no medible):")
            for tercile in ("0", "1", "2"):
                t = by_tercile[tercile]
                m = t["positive_rate_measurable"]
                u = t["positive_rate_unmeasurable"]
                m_str = f"{m:.4f}" if m is not None else "n/a"
                u_str = f"{u:.4f}" if u is not None else "n/a"
                lines.append(f"     tercil {tercile}: {m_str} / {u_str}")
    else:
        lines.append("   sin casos etiquetados suficientes para el test de Fisher")
    lines.append("")

    lines.append("5. Recorrido alcanzable por imagen (grados de ITA, solo entre las medibles y alcanzables)")
    if summary["reachable_ita_span_distribution"]:
        d = summary["reachable_ita_span_distribution"]
        lines.append(
            f"   mediana={d['p50']:.3f}  p05={d['p5']:.3f}  p25={d['p25']:.3f}  "
            f"p75={d['p75']:.3f}  p95={d['p95']:.3f}  (n={summary['reachable_ita_span_n']})"
        )
    else:
        lines.append("   ningun caso alcanzable; sin recorrido que reportar")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-cap-degrees", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    manifest_rows = _read_manifest(args.manifest)
    analyzed = []
    for row in manifest_rows:
        result = analyze_row(row, seed=args.seed, policy_cap_degrees=args.policy_cap_degrees)
        if result is not None:
            analyzed.append(result)

    if not analyzed:
        raise SystemExit("no readable rows in the manifest; nothing to report")

    summary = summarize(analyzed)
    report = render_report(
        summary,
        manifest_path=args.manifest,
        seed=args.seed,
        policy_cap_degrees=args.policy_cap_degrees,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report)
    print()
    print(f"written: {args.output_dir / 'report.txt'}")
    print(f"written: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
