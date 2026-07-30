"""Gates are tested with the numbers from the real Kaggle run of 2026-07-29.

That run reported "G2 calidad SSL: 0.794 FALLA" when the self-supervised backbone
had actually scored 0.8005 and cleared the threshold. The gate had averaged the
SSL backbone with the supervised baseline. These tests pin that down.
"""

import numpy as np
import pytest

from dermssl import gates as g

# Verbatim from probe_grid_v5.csv of the smoke run.
REAL_GRID = [
    {"backbone": "ssl_dinov2_vits14", "probe": "knn", "fraction": 0.05, "auc": 0.7308783965},
    {"backbone": "ssl_dinov2_vits14", "probe": "prototype", "fraction": 0.05, "auc": 0.747327471},
    {"backbone": "ssl_dinov2_vits14", "probe": "logreg", "fraction": 0.05, "auc": 0.790240056},
    {"backbone": "ssl_dinov2_vits14", "probe": "knn", "fraction": 1.0, "auc": 0.8005134833},
    {"backbone": "ssl_dinov2_vits14", "probe": "prototype", "fraction": 1.0, "auc": 0.8517324215},
    {"backbone": "ssl_dinov2_vits14", "probe": "logreg", "fraction": 1.0, "auc": 0.9092452372},
    {"backbone": "sup_resnet50", "probe": "knn", "fraction": 0.05, "auc": 0.598562515},
    {"backbone": "sup_resnet50", "probe": "prototype", "fraction": 0.05, "auc": 0.6218891598},
    {"backbone": "sup_resnet50", "probe": "logreg", "fraction": 0.05, "auc": 0.6382858724},
    {"backbone": "sup_resnet50", "probe": "knn", "fraction": 1.0, "auc": 0.7874636717},
    {"backbone": "sup_resnet50", "probe": "prototype", "fraction": 1.0, "auc": 0.738453166},
    {"backbone": "sup_resnet50", "probe": "logreg", "fraction": 1.0, "auc": 0.8301518879},
]


def test_regression_gate_no_longer_averages_ssl_with_the_baseline():
    """The exact bug: 0.794 was mean(0.8005, 0.7875) across both backbones."""
    auc = g.ssl_knn_auc(REAL_GRID)
    assert auc == pytest.approx(0.8005134833)
    assert auc != pytest.approx(0.794, abs=1e-3), "regression: baseline leaked into the SSL score"


def test_ssl_knn_auc_defaults_to_the_largest_fraction():
    assert g.ssl_knn_auc(REAL_GRID) == pytest.approx(0.8005134833)
    assert g.ssl_knn_auc(REAL_GRID, fraction=0.05) == pytest.approx(0.7308783965)


def test_ssl_knn_auc_averages_across_several_ssl_backbones():
    rows = [
        {"backbone": "ssl_a", "probe": "knn", "fraction": 1.0, "auc": 0.80},
        {"backbone": "ssl_b", "probe": "knn", "fraction": 1.0, "auc": 0.90},
        {"backbone": "sup_c", "probe": "knn", "fraction": 1.0, "auc": 0.10},
    ]
    assert g.ssl_knn_auc(rows) == pytest.approx(0.85)


def test_ssl_knn_auc_ignores_nan_and_missing_rows():
    rows = [
        {"backbone": "ssl_a", "probe": "knn", "fraction": 1.0, "auc": float("nan")},
        {"backbone": "ssl_a", "probe": "knn", "fraction": 1.0, "auc": 0.75},
    ]
    assert g.ssl_knn_auc(rows) == pytest.approx(0.75)
    assert np.isnan(g.ssl_knn_auc([{"backbone": "sup_x", "probe": "knn", "fraction": 1.0, "auc": 0.9}]))
    assert np.isnan(g.ssl_knn_auc([]))


def test_architecture_control_requires_a_supervised_vit_next_to_an_ssl_vit():
    assert g.has_architecture_control(["ssl_dinov2_vits14", "sup_vit_b16"]) is True
    assert g.has_architecture_control(["ssl_dinov2_vits14", "sup_resnet50"]) is False
    # No self-supervised ViT in the run: nothing to control for.
    assert g.has_architecture_control(["sup_resnet50"]) is True


def base_kwargs(**overrides):
    kwargs = dict(
        backend="kaggle-input",
        ham_images=10015,
        lesion_overlap=0,
        rows_total=12,
        rows_ok=12,
        frozen_verified=True,
        backbones=["ssl_dinov2_vits14", "sup_resnet50"],
        grid_rows=REAL_GRID,
        smoke=True,
    )
    kwargs.update(overrides)
    return kwargs


def by_name(gates, fragment):
    return next(gate for gate in gates if fragment in gate.name)


def test_the_real_smoke_run_now_has_no_blocking_failures():
    """Same inputs as the 2026-07-29 run, which wrongly reported a failure."""
    result = g.evaluate_gates(**base_kwargs())
    assert g.all_blocking_pass(result) is True
    assert by_name(result, "calidad SSL").status == g.SKIP
    assert by_name(result, "control de arquitectura").status == g.SKIP


def test_quality_gates_are_enforced_in_a_real_run():
    result = g.evaluate_gates(**base_kwargs(smoke=False))
    quality = by_name(result, "calidad SSL")
    assert quality.status == g.PASS, "0.8005 clears the 0.80 threshold"
    # sup_resnet50 is not a ViT, so the architecture control is missing for real.
    assert by_name(result, "control de arquitectura").status == g.FAIL
    assert g.all_blocking_pass(result) is False


def test_architecture_control_passes_with_the_supervised_vit_present():
    result = g.evaluate_gates(
        **base_kwargs(smoke=False, backbones=["ssl_dinov2_vits14", "sup_vit_b16", "sup_resnet50"])
    )
    assert by_name(result, "control de arquitectura").status == g.PASS
    assert g.all_blocking_pass(result) is True


def test_quality_gate_fails_below_the_threshold_in_a_real_run():
    weak = [{"backbone": "ssl_x", "probe": "knn", "fraction": 1.0, "auc": 0.61}]
    result = g.evaluate_gates(**base_kwargs(smoke=False, grid_rows=weak, backbones=["ssl_x"]))
    assert by_name(result, "calidad SSL").status == g.FAIL
    assert g.all_blocking_pass(result) is False


def test_pipeline_gates_are_enforced_even_in_smoke():
    """A smoke run exists precisely to catch these, so they never skip."""
    leaky = g.evaluate_gates(**base_kwargs(lesion_overlap=7))
    assert by_name(leaky, "anti-leakage").status == g.FAIL
    assert g.all_blocking_pass(leaky) is False

    partial = g.evaluate_gates(**base_kwargs(rows_ok=11))
    assert by_name(partial, "end-to-end").status == g.FAIL

    thawed = g.evaluate_gates(**base_kwargs(frozen_verified=False))
    assert by_name(thawed, "backbone congelado").status == g.FAIL

    empty = g.evaluate_gates(**base_kwargs(rows_total=0, rows_ok=0))
    assert by_name(empty, "end-to-end").status == g.FAIL


def test_missing_data_fails_g0():
    assert by_name(g.evaluate_gates(**base_kwargs(ham_images=0)), "resolucion de datos").status == g.FAIL
    assert by_name(g.evaluate_gates(**base_kwargs(backend="")), "resolucion de datos").status == g.FAIL


def test_skipped_gates_do_not_block_but_are_visible():
    result = g.evaluate_gates(**base_kwargs())
    text = g.report(result)
    assert "OMITIDA" in text
    assert "TODAS LAS COMPUERTAS EXIGIBLES PASAN" in text
    assert "omitida(s) por SMOKE" in text


def test_report_names_the_number_of_failures():
    result = g.evaluate_gates(**base_kwargs(lesion_overlap=3))
    text = g.report(result)
    assert "HAY 1 COMPUERTA(S) EN FALLA" in text


def test_gate_serialisation_keeps_the_three_states():
    result = g.evaluate_gates(**base_kwargs())
    states = {gate.as_dict()["estado"] for gate in result}
    assert states <= {g.PASS, g.FAIL, g.SKIP}
    row = by_name(result, "anti-leakage").as_dict()
    assert set(row) == {"gate", "criterio", "valor", "estado", "pasa"}
    assert row["pasa"] is True


def test_blocking_is_only_true_for_failures():
    assert g.Gate("x", "c", 1, g.FAIL).blocking is True
    assert g.Gate("x", "c", 1, g.SKIP).blocking is False
    assert g.Gate("x", "c", 1, g.PASS).blocking is False
