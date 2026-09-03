"""The result tables in the READMEs are generated from the JSONs, never typed
by hand: sixteen runs times three tables is where transcription errors live."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_results_tables as brt  # noqa: E402


def _probe(bal_acc, **extra):
    base = {
        "accuracy": 0.5, "balanced_accuracy": bal_acc, "macro_f1": 0.4,
        "macro_precision": 0.4, "macro_recall": bal_acc,
        "auc_ovr_macro": 0.8, "ece": 0.1,
        "recall_por_clase": {"benign": 0.9, "malignant": 0.3},
    }
    base.update(extra)
    return base


def _result(model, knn, logreg, proto, **extra):
    data = {
        "modelo": model, "smoke": False, "dim_features": 512,
        "probes": {"knn": _probe(knn), "logreg": _probe(logreg),
                   "prototype": _probe(proto)},
    }
    data.update(extra)
    return data


def test_best_probe_is_the_one_with_the_highest_balanced_accuracy():
    assert brt.best_probe(_result("x", 0.30, 0.62, 0.51)) == "logreg"
    assert brt.best_probe(_result("x", 0.71, 0.62, 0.51)) == "knn"
    assert brt.best_probe(_result("x", 0.30, 0.42, 0.81)) == "prototype"


def test_rows_are_ranked_by_the_best_probe_descending():
    bank = {"a": _result("a", 0.3, 0.50, 0.4),
            "b": _result("b", 0.3, 0.70, 0.4),
            "c": _result("c", 0.3, 0.60, 0.4)}
    assert [r.model for r in brt.rank(bank)] == ["b", "c", "a"]


def test_row_carries_the_metrics_of_its_best_probe():
    row = brt.rank({"a": _result("a", 0.3, 0.62, 0.4)})[0]
    assert row.probe == "logreg"
    assert row.metrics["balanced_accuracy"] == 0.62
    assert row.dims == 512


def test_supervision_label_separates_ssl_from_supervised():
    assert brt.SUPERVISION["dinov2_vitb14"] == "SSL"
    assert brt.SUPERVISION["simsiam_r50"] == "SSL"
    assert brt.SUPERVISION["vit_b16"] == "Supervisado"
    assert set(brt.SUPERVISION) == set(brt.DISPLAY)


def test_metrics_table_has_one_row_per_model_and_bolds_the_leader():
    md = brt.table_metrics(brt.rank({"vit_b16": _result("vit_b16", 0.3, 0.62, 0.4),
                                     "vgg19": _result("vgg19", 0.3, 0.51, 0.4)}))
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len(lines) == 4  # header + separator + two models
    assert "**0.620**" in lines[2]
    assert "0.510" in lines[3] and "**0.510**" not in lines[3]


def test_probe_table_lists_the_three_probes():
    md = brt.table_probes(brt.rank({"vgg19": _result("vgg19", 0.31, 0.51, 0.42)}))
    assert "| kNN | LogReg | Prototipos |" in md
    assert "| 0.310 | 0.510 | 0.420 |" in md


def test_per_class_table_uses_the_classes_present_in_the_json():
    md = brt.table_per_class(brt.rank({"vgg19": _result("vgg19", 0.3, 0.5, 0.4)}),
                             field="recall_por_clase")
    assert "| benign | malignant |" in md
    assert "0.900" in md and "0.300" in md


def test_per_class_table_reports_a_field_the_old_runs_never_had(tmp_path):
    bank = {"vgg19": _result("vgg19", 0.3, 0.5, 0.4)}
    with pytest.raises(brt.MissingField):
        brt.table_per_class(brt.rank(bank), field="f1_por_clase")


def test_loading_a_bank_skips_smoke_results(tmp_path):
    (tmp_path / "results_a.json").write_text(json.dumps(_result("a", .3, .5, .4)))
    smoke = _result("b", .3, .5, .4)
    smoke["smoke"] = True
    (tmp_path / "results_b.json").write_text(json.dumps(smoke))
    bank = brt.load_bank(tmp_path)
    assert list(bank) == ["a"]


def test_loading_an_empty_bank_is_an_explicit_error(tmp_path):
    with pytest.raises(brt.MissingField):
        brt.load_bank(tmp_path)
