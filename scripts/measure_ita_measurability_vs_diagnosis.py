"""Does ITA measurability depend on the diagnosis? Measured on two cohorts.

The question matters because every fairness-by-skin-tone analysis that abstains
on unmeasurable frames reports its metric on the subset that remains. If that
subset is not a random sample of the diagnoses, the metric describes a different
population than the one the reader assumes - and nothing in the usual reporting
makes that visible.

Both cohorts are read from artifacts already on disk. Nothing is recomputed from
images and no figure is copied by hand; the numbers below come out of the JSON.

Cohort A, HAM10000 (``tone_stratified_eval``): per-class counts for the full
split and for the ITA-measurable subset.

Cohort B, ISIC2019 pool (``pool_tone_eligibility``): a 2x2 table of
measurability against a binary label, already computed, on the 39.6% of the pool
that carries a label. Independent of cohort A in dataset, in masking path and in
labelling, so agreement between them is replication rather than one measurement
reported twice.

The malignant grouping is imported from ``dermssl.ddi`` rather than restated, so
there is one definition in the project and not two that can drift. Both of its
readings are reported: the primary one counts ``akiec``, which is premalignant
or in situ, and the strict one excludes it.
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import fisher_exact

from dermssl.ddi import HAM_MALIGNANT, HAM_MALIGNANT_STRICT

ROOT = Path(__file__).resolve().parents[1]
HAM = ROOT / "resultados/segmentacion/v8/tone_stratified_eval/summary.json"
ISIC = ROOT / "resultados/segmentacion/v8/pool_tone_eligibility/summary_pool_real_isic2019.json"
OUT = ROOT / "resultados/segmentacion/v8/ita_measurability_vs_diagnosis/summary.json"


def _table(positive_in: int, total_in: int, positive_out: int, total_out: int) -> dict:
    """Fisher on (measurable, unmeasurable) x (positive, negative)."""
    negative_in = total_in - positive_in
    negative_out = total_out - positive_out
    odds_ratio, p_value = fisher_exact([[positive_in, negative_in], [positive_out, negative_out]])
    return {
        "measurable_positive": positive_in,
        "measurable_negative": negative_in,
        "unmeasurable_positive": positive_out,
        "unmeasurable_negative": negative_out,
        "positive_rate_measurable": positive_in / total_in if total_in else None,
        "positive_rate_unmeasurable": positive_out / total_out if total_out else None,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
    }


def ham_cohort() -> dict:
    payload = json.loads(HAM.read_text(encoding="utf-8"))
    composition = payload["class_composition"]
    full = composition["counts_full"]
    measurable = composition["counts_measurable"]
    n_full = composition["n_full"]
    n_measurable = composition["n_measurable"]
    # The unmeasurable stratum is the complement, not a third list: every
    # prediction is in the manifest (coverage 1.0), so full = measurable + rest.
    n_unmeasurable = n_full - n_measurable

    readings = {}
    for name, grouping in (("primary_with_akiec", HAM_MALIGNANT),
                           ("strict_without_akiec", HAM_MALIGNANT_STRICT)):
        mal_measurable = sum(measurable.get(c, 0) for c in grouping)
        mal_full = sum(full.get(c, 0) for c in grouping)
        readings[name] = {
            "classes": sorted(grouping),
            **_table(mal_measurable, n_measurable, mal_full - mal_measurable, n_unmeasurable),
        }

    return {
        "cohort": "HAM10000 via tone_stratified_eval",
        "source": str(HAM.relative_to(ROOT)).replace("\\", "/"),
        "n_full": n_full,
        "n_measurable": n_measurable,
        "measurable_fraction": n_measurable / n_full,
        "nevus_share_full": full["nv"] / n_full,
        "nevus_share_measurable": measurable["nv"] / n_measurable,
        "readings": readings,
    }


def isic_cohort() -> dict:
    payload = json.loads(ISIC.read_text(encoding="utf-8"))
    block = payload["measurability_vs_label"]
    pos_in, neg_in = block["table_measurable_pos_neg"]
    pos_out, neg_out = block["table_unmeasurable_pos_neg"]
    return {
        "cohort": "ISIC2019 SSL pool via pool_tone_eligibility",
        "source": str(ISIC.relative_to(ROOT)).replace("\\", "/"),
        "n_total": payload["n_total"],
        "measurable_rate": payload["measurable_rate"],
        "label_coverage_fraction": payload["label_coverage_fraction"],
        "caveat": (
            "Condicional al 39,6 % del pool que trae etiqueta, no al pool completo. "
            "Asi lo declara el propio reporte y asi hay que citarlo."
        ),
        "reading": _table(pos_in, pos_in + neg_in, pos_out, pos_out + neg_out),
        "by_area_tercile": payload["measurability_vs_label_by_area_tercile"],
        "reported_odds_ratio": block["odds_ratio"],
        "reported_p_value": block["p_value"],
    }


def main() -> None:
    ham = ham_cohort()
    isic = isic_cohort()

    # The ISIC block was already computed upstream; recomputing it here and
    # comparing is a check that this script reads the table the same way the
    # original analysis wrote it, not an independent result.
    agreement = {
        "odds_ratio_matches_upstream": abs(
            isic["reading"]["odds_ratio"] - isic["reported_odds_ratio"]
        ) < 1e-9,
        "p_value_matches_upstream": abs(
            isic["reading"]["p_value"] - isic["reported_p_value"]
        ) < 1e-9,
    }

    payload = {
        "question": (
            "La medibilidad del ITA depende de la clase diagnostica? Si depende, toda "
            "metrica de equidad por tono que abstenga sobre lo no medible describe una "
            "poblacion distinta de la que el lector supone."
        ),
        "cohorts": [ham, isic],
        "isic_recomputation_agreement": agreement,
        "direction_replicates": (
            ham["readings"]["primary_with_akiec"]["odds_ratio"] < 1.0
            and isic["reading"]["odds_ratio"] < 1.0
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
