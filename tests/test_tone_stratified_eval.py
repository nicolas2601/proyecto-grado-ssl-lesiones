"""Tests for the continuous-ITA stratified evaluation (Fase 4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dermssl.per_class_metrics import HAM10000_CLASSES
from dermssl.tone_stratified_eval import (
    build_report,
    coverage_report,
    cohort_terciles,
    correctness_ita_correlation,
    join_predictions_with_ita,
    load_npz_predictions,
    macro_f1_by_cohort_tercile,
    macro_f1_by_mask_source,
    read_ita_manifest,
    reachable_range_percentiles,
)

N_CLASSES = len(HAM10000_CLASSES)


def _one_hot(class_name: str, confidence: float = 0.9) -> np.ndarray:
    index = HAM10000_CLASSES.index(class_name)
    proba = np.full(N_CLASSES, (1.0 - confidence) / (N_CLASSES - 1))
    proba[index] = confidence
    return proba


def test_module_never_imports_fst_bucket_tables_or_calls_the_augmenter():
    """Guardia dura pedida: nada de tabla de fototipo COMO CODIGO, nada que transforme pixeles.

    The module docstring is allowed to NAME ``FST_THRESHOLDS`` in prose (it
    explains what this replaces); what must never happen is importing or
    indexing it, or calling the augmentation functions.
    """
    source = __import__("dermssl.tone_stratified_eval", fromlist=["x"]).__file__
    text = open(source, encoding="utf-8").read()
    assert "import FST_THRESHOLDS" not in text
    assert "FST_THRESHOLDS[" not in text
    assert "fst_bucket(" not in text
    assert "augment_skin_tone(" not in text
    assert "augment_skin_tone_v2(" not in text


def test_load_npz_predictions_reads_the_named_variant(tmp_path):
    y_true = np.array([0, 1, 2])
    proba_a = np.random.default_rng(0).random((3, N_CLASSES))
    proba_b = np.random.default_rng(1).random((3, N_CLASSES))
    path = tmp_path / "preds.npz"
    np.savez(path, y_true=y_true, proposed=proba_a, plain=proba_b)

    loaded_true, loaded_proba = load_npz_predictions(path, "proposed")
    assert np.array_equal(loaded_true, y_true)
    assert np.array_equal(loaded_proba, proba_a)


def test_load_npz_predictions_rejects_missing_variant(tmp_path):
    path = tmp_path / "preds.npz"
    np.savez(path, y_true=np.array([0]), proposed=np.zeros((1, N_CLASSES)))
    with pytest.raises(ValueError, match="variant"):
        load_npz_predictions(path, "nonexistent")


def test_load_npz_predictions_rejects_wrong_class_count(tmp_path):
    path = tmp_path / "preds.npz"
    np.savez(path, y_true=np.array([0]), proposed=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="columns"):
        load_npz_predictions(path, "proposed")


def test_read_ita_manifest_reads_jsonl(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps({"row": 0, "ita": 12.5, "ita_status": "ok"}) + "\n"
        + json.dumps({"row": 1, "ita": None, "ita_status": "ita_indeterminate"}) + "\n",
        encoding="utf-8",
    )
    rows = read_ita_manifest(path)
    assert len(rows) == 2
    assert rows[0]["row"] == 0


def test_join_detects_misaligned_class_names():
    y_true = np.array([HAM10000_CLASSES.index("mel")])
    with pytest.raises(ValueError, match="not aligned"):
        join_predictions_with_ita(y_true, [{"row": 0, "class": "nv"}])


def test_join_accepts_matching_class_names():
    y_true = np.array([HAM10000_CLASSES.index("mel")])
    join_predictions_with_ita(y_true, [{"row": 0, "class": "mel"}])  # must not raise


def test_coverage_report_counts_measurable_and_declares_partial_coverage():
    ita_rows = [
        {"row": 0, "ita": 10.0, "ita_status": "ok", "mask_source": "teacher_026"},
        {"row": 1, "ita": None, "ita_status": "ita_indeterminate", "mask_source": "otsu"},
        {"row": 2, "ita": 5.0, "ita_status": "ok", "mask_source": "teacher_026"},
    ]
    report = coverage_report(ita_rows, n_total=5)  # 5 total, only 3 in manifest: partial
    assert report["n_total_predictions"] == 5
    assert report["n_in_ita_manifest"] == 3
    assert report["n_ita_measurable"] == 2
    assert report["by_ita_status"] == {"ok": 2, "ita_indeterminate": 1}
    assert report["by_mask_source"] == {"teacher_026": 2, "otsu": 1}


def _row(row, ita, class_name, mask_source="teacher_026", status="ok"):
    return {"row": row, "ita": ita, "ita_status": status, "class": class_name, "mask_source": mask_source}


def test_macro_f1_by_mask_source_separates_regimes():
    rng = np.random.default_rng(0)
    y_true, proba, ita_rows = [], [], []
    # mask A: always correct. mask B: always wrong (predicts a fixed other class).
    for i in range(40):
        y_true.append(HAM10000_CLASSES.index("mel"))
        proba.append(_one_hot("mel"))
        ita_rows.append(_row(i, rng.uniform(-10, 10), "mel", mask_source="teacher_026"))
    for i in range(40, 80):
        y_true.append(HAM10000_CLASSES.index("mel"))
        proba.append(_one_hot("nv"))  # always wrong
        ita_rows.append(_row(i, rng.uniform(-10, 10), "mel", mask_source="otsu_fallback"))

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    result = macro_f1_by_mask_source(y_true, proba, ita_rows, min_n=10)
    assert result["teacher_026"]["status"] == "OK"
    assert result["otsu_fallback"]["status"] == "OK"
    assert result["teacher_026"]["macro_f1"] > result["otsu_fallback"]["macro_f1"]


def test_macro_f1_by_mask_source_marks_thin_strata_skipped():
    y_true = np.asarray([HAM10000_CLASSES.index("mel")] * 5)
    proba = np.asarray([_one_hot("mel")] * 5)
    ita_rows = [_row(i, 1.0, "mel", mask_source="rare_source") for i in range(5)]
    result = macro_f1_by_mask_source(y_true, proba, ita_rows, min_n=30)
    assert result["rare_source"]["status"] == "SKIPPED"
    assert result["rare_source"]["count"] == 5


def test_cohort_terciles_split_by_this_cohorts_own_percentiles_not_a_fixed_cut():
    values = np.array([-40.0, -30.0, -20.0, 0.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    labels = cohort_terciles(values)
    assert set(labels) == {"cohorte_t1_mas_oscuro", "cohorte_t2_medio", "cohorte_t3_mas_claro"}
    # roughly a third each, not whatever a Groh/Corbin cut would give on this data
    counts = {label: int((labels == label).sum()) for label in set(labels)}
    assert all(2 <= c <= 4 for c in counts.values())


def test_macro_f1_by_cohort_tercile_matches_stratified_by_label_on_its_own_terciles():
    """No reinventa la estratificacion: compone cohort_terciles() con metrics.stratified_by_label.

    No se afirma un macro-F1 "perfecto": con un unico label verdadero en la
    cohorte, per_class_f1 puntua 0 las 6 clases ausentes por convencion
    documentada de dermssl.metrics ("classes with no true and no predicted
    samples score 0"), asi que el macro-F1 de una cohorte de una sola clase
    nunca llega a 1.0. Lo que se prueba es que la composicion es correcta,
    no un valor fabricado.
    """
    from dermssl import metrics

    rng = np.random.default_rng(1)
    y_true, proba, ita_rows = [], [], []
    for i in range(60):
        y_true.append(HAM10000_CLASSES.index("nv"))
        proba.append(_one_hot("nv"))
        ita_rows.append(_row(i, rng.uniform(-50, 50), "nv"))
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    result = macro_f1_by_cohort_tercile(y_true, proba, ita_rows, min_n=10)
    ita_values = np.asarray([r["ita"] for r in ita_rows])
    expected = metrics.stratified_by_label(y_true, proba, cohort_terciles(ita_values), N_CLASSES, min_n=10)
    assert result.keys() == expected.keys()
    for key in result:
        assert result[key]["status"] == expected[key]["status"]
        assert result[key]["count"] == expected[key]["count"]
        if result[key]["status"] == "OK":
            assert result[key]["macro_f1"] == pytest.approx(expected[key]["macro_f1"])


def test_correctness_ita_correlation_detects_a_real_relationship():
    y_true, proba, ita_rows = [], [], []
    for i in range(60):
        ita = -30.0 + i  # spread from -30 to 29
        correct = ita > 0  # clean split: higher ITA -> correct
        y_true.append(HAM10000_CLASSES.index("bcc"))
        proba.append(_one_hot("bcc") if correct else _one_hot("df"))
        ita_rows.append(_row(i, ita, "bcc"))
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    result = correctness_ita_correlation(y_true, proba, ita_rows)
    assert result["rho"] > 0.8
    assert result["p_value"] < 0.01
    assert result["ci95_low"] is not None and result["ci95_low"] > 0


def test_correctness_ita_correlation_handles_insufficient_n():
    y_true = np.asarray([HAM10000_CLASSES.index("mel")])
    proba = np.asarray([_one_hot("mel")])
    result = correctness_ita_correlation(y_true, proba, [_row(0, 5.0, "mel")])
    assert result["n"] == 1
    assert result["rho"] is None


def test_reachable_range_percentiles_optional_field():
    rows_without = [_row(0, 1.0, "mel")]
    assert reachable_range_percentiles(rows_without) is None

    rows_with = [{**_row(i, 1.0, "mel"), "reachable_ita_span": float(50 + i)} for i in range(10)]
    result = reachable_range_percentiles(rows_with)
    assert result is not None
    assert result["n"] == 10
    assert result["p50"] == pytest.approx(np.percentile([50 + i for i in range(10)], 50))


def test_class_composition_reports_full_and_measurable_side_by_side():
    from dermssl.tone_stratified_eval import class_composition

    y_full = np.asarray(
        [HAM10000_CLASSES.index("nv")] * 80 + [HAM10000_CLASSES.index("mel")] * 20
    )
    # measurable subset depletes mel: only 2 of the 20 mel survive
    y_measurable = np.asarray(
        [HAM10000_CLASSES.index("nv")] * 78 + [HAM10000_CLASSES.index("mel")] * 2
    )
    comp = class_composition(y_full, y_measurable)
    assert comp["n_full"] == 100
    assert comp["n_measurable"] == 80
    assert comp["fraction_full"]["mel"] == pytest.approx(0.20)
    assert comp["fraction_measurable"]["mel"] == pytest.approx(2 / 80)
    assert comp["fraction_measurable"]["mel"] < comp["fraction_full"]["mel"]


def test_macro_f1_full_reference_carries_its_own_caveat():
    from dermssl.tone_stratified_eval import NOT_COMPARABLE_CAVEAT, macro_f1_full_reference

    y_true = np.asarray([HAM10000_CLASSES.index("mel")] * 10)
    proba = np.asarray([_one_hot("mel") for _ in range(10)])
    result = macro_f1_full_reference(y_true, proba)
    assert result["n"] == 10
    assert result["caveat"] == NOT_COMPARABLE_CAVEAT
    assert "NO COMPARABLE" in result["caveat"]


def test_strata_declare_absent_and_thin_classes_without_averaging_them_away():
    # one stratum: 25 'nv' (well above min_n) plus 3 'mel' (below min_n=10), 0 of anything else.
    y_true = np.asarray(
        [HAM10000_CLASSES.index("nv")] * 25 + [HAM10000_CLASSES.index("mel")] * 3
    )
    proba = np.asarray(
        [_one_hot("nv") for _ in range(25)] + [_one_hot("mel") for _ in range(3)]
    )
    ita_rows = [_row(i, 1.0, "nv", mask_source="only_source") for i in range(25)] + [
        _row(25 + i, 1.0, "mel", mask_source="only_source") for i in range(3)
    ]
    result = macro_f1_by_mask_source(y_true, proba, ita_rows, min_n=10)
    entry = result["only_source"]
    assert entry["status"] == "OK"
    assert entry["per_class_counts"]["nv"] == 25
    assert entry["per_class_counts"]["mel"] == 3
    assert "mel" in entry["classes_below_min_n"]
    assert "nv" not in entry["classes_below_min_n"]
    assert set(entry["classes_absent"]) == set(HAM10000_CLASSES) - {"nv", "mel"}


def test_build_report_surfaces_class_composition_and_thin_class_warnings():
    y_true = np.asarray(
        [HAM10000_CLASSES.index("nv")] * 25 + [HAM10000_CLASSES.index("mel")] * 3
    )
    proba = np.asarray(
        [_one_hot("nv") for _ in range(25)] + [_one_hot("mel") for _ in range(3)]
    )
    ita_rows = [_row(i, 1.0, "nv", mask_source="only_source") for i in range(25)] + [
        _row(25 + i, 1.0, "mel", mask_source="only_source") for i in range(3)
    ]
    report = build_report(y_true, proba, ita_rows, variant="proposed", min_n=10)
    assert "class_composition" in report
    assert report["class_composition"]["n_measurable"] == 28
    assert "macro_f1_full_split_reference" in report
    assert "mask_source/only_source" in report["strata_with_thin_classes"]
    assert "mel" in report["strata_with_thin_classes"]["mask_source/only_source"]


def test_build_report_declares_partial_coverage_and_which_variant():
    rng = np.random.default_rng(2)
    n = 50
    y_true = np.full(n, HAM10000_CLASSES.index("mel"))
    proba = np.asarray([_one_hot("mel") for _ in range(n)])
    # Only half the predictions have an ITA record: partial coverage.
    ita_rows = [_row(i, rng.uniform(-20, 20), "mel") for i in range(n // 2)]

    report = build_report(y_true, proba, ita_rows, variant="proposed", min_n=5)
    assert report["variant"] == "proposed"
    assert report["coverage"]["n_total_predictions"] == n
    assert report["coverage"]["n_in_ita_manifest"] == n // 2
    assert report["coverage"]["manifest_coverage_fraction"] == pytest.approx(0.5)
