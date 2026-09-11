"""The result tables in the READMEs are generated from the JSONs, never typed
by hand: sixteen runs times three tables is where transcription errors live."""

import json
import re
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
        "ic95_bootstrap": {"balanced_accuracy": [bal_acc - 0.04, bal_acc + 0.04],
                           "macro_f1": [0.35, 0.45]},
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


def _with_ci(model, bal_acc, lo, hi):
    """A one-probe result whose interval we control exactly."""
    return _result(model, 0.1, bal_acc, 0.1,
                   probes={"logreg": _probe(bal_acc, ic95_bootstrap={
                       "balanced_accuracy": [lo, hi], "macro_f1": [0.3, 0.5]})})


def test_interval_table_flags_who_is_separated_from_the_leader():
    rows = brt.rank({
        "dinov2_vitb14": _with_ci("dinov2_vitb14", 0.618, 0.576, 0.660),
        "alexnet": _with_ci("alexnet", 0.589, 0.548, 0.631),
        "simsiam_r50": _with_ci("simsiam_r50", 0.494, 0.453, 0.539),
    })
    table = brt.table_intervals(rows)
    cell = {line.split("|")[1].strip(): line.split("|")[-2].strip()
            for line in table.splitlines()[2:]}
    assert cell["DINOv2 ViT-B/14"] == "-"
    assert cell["AlexNet"] == "no", "0.631 >= 0.576, the intervals touch"
    assert cell["SimSiam R50"] == "sí", "0.539 < 0.576, no overlap"
    assert "[0.576, 0.660]" in table


def test_overlap_is_computed_against_the_leader_lower_bound():
    leader = brt.rank({"a": _with_ci("a", 0.62, 0.58, 0.66)})[0]
    touching = brt.rank({"b": _with_ci("b", 0.55, 0.50, 0.58)})[0]
    clear = brt.rank({"c": _with_ci("c", 0.50, 0.45, 0.579)})[0]
    assert brt.overlaps_leader(touching, leader)
    assert not brt.overlaps_leader(clear, leader)


def test_interval_table_is_explicit_when_the_json_predates_bootstrap():
    entry = _result("resnet50", 0.3, 0.57, 0.4)
    del entry["probes"]["logreg"]["ic95_bootstrap"]
    with pytest.raises(brt.MissingField, match="bootstrap"):
        brt.table_intervals(brt.rank({"resnet50": entry}))


def _finetune_result(model, bal_acc):
    """A fine-tuning JSON: same shape, but a single probe called 'finetune'.

    There are no kNN / logreg / prototype probes because nothing is probed -
    the whole network was trained."""
    return {"modelo": model, "smoke": False, "dim_features": 2048,
            "dataset": "ham10000", "probes": {"finetune": _probe(bal_acc)}}


def _write(tmp_path, results):
    for data in results:
        (tmp_path / f"results_{data['modelo']}_ft.json").write_text(json.dumps(data))
    return tmp_path


def test_a_finetuning_bank_renders_without_crashing(tmp_path):
    """The probe table used to index knn/logreg/prototype unconditionally, so
    a bank whose only probe is 'finetune' died with KeyError mid-render."""
    _write(tmp_path, [_finetune_result("resnet50", 0.71),
                      _finetune_result("vit_b16", 0.68)])
    out = brt.render(tmp_path)
    assert "resnet50" in out.lower() or "ResNet-50" in out
    assert "0.710" in out


def test_a_single_probe_bank_skips_the_per_probe_table(tmp_path):
    """With one probe that table is just the metrics table again, column for
    column. Saying why it is absent beats printing a duplicate."""
    _write(tmp_path, [_finetune_result("resnet50", 0.71)])
    out = brt.render(tmp_path)
    assert "### Balanced accuracy por probe: no aplica" in out
    assert "| kNN |" not in out and "| LogReg |" not in out
    assert "finetune" in out.lower(), "the reader should learn which probe it was"


def test_the_frozen_banks_still_get_their_probe_table():
    rows = brt.rank(brt.load_bank(REPO / "results" / "baselines"))
    out = brt.render(REPO / "results" / "baselines")
    assert "Balanced accuracy por probe" in out
    assert all(p in brt.table_probes(rows) for p in ("kNN", "LogReg", "Prototipos"))


def test_the_coupling_artefact_is_large_at_our_noise_level():
    """The README cites a measured rate; this keeps that number honest.

    delta = fine_tuned - frozen puts the frozen score on both axes, so noise
    alone yields a strong negative frozen-vs-delta correlation. If this rate
    were small the warning would be over-cautious; it is not."""
    sys.path.insert(0, str(REPO / "scripts"))
    import coupling_check

    rate = coupling_check.spurious_rate(models=8, noise=0.02, threshold=-0.6,
                                        trials=20000, seed=42)
    assert 0.55 < rate < 0.70, rate
    # Spanish typography puts a space before the sign ("61.6 %"), so compare
    # with the separator collapsed rather than demanding one spelling.
    readme = re.sub(r"\s+%", "%", (REPO / "README.md").read_text())
    assert f"{rate:.1%}" in readme, (
        f"the README must cite the measured rate, {rate:.1%}")


def test_no_real_effect_means_no_correlation_in_the_valid_comparison():
    """Control: the same simulation, compared the way the README says to.

    Frozen against fine-tuned has the frozen score on one axis only, so with
    zero real effect it must show no systematic sign - which is exactly why
    that is the comparison the conclusion has to rest on."""
    sys.path.insert(0, str(REPO / "scripts"))
    import random

    import coupling_check

    rng = random.Random(7)
    negatives = 0
    trials = 4000
    for _ in range(trials):
        frozen = [0.55 + rng.gauss(0, 0.02) for _ in range(8)]
        tuned = [0.72 + rng.gauss(0, 0.02) for _ in range(8)]
        if coupling_check.spearman(frozen, tuned) < -0.6:
            negatives += 1
    assert negatives / trials < 0.10, (
        f"the uncoupled comparison should not lean negative: {negatives / trials:.1%}")


def test_results_dir_alone_is_enough(tmp_path, capsys):
    """cluster/finetune/README documents this exact command.

    It used to abort with "give --bank <banco> or --weights" because --bank was
    required even when --results-dir already said where to look - so a command
    printed in a guide for someone with no way to ask us simply failed."""
    _write(tmp_path, [_finetune_result("resnet50", 0.71)])
    brt.main(["--results-dir", str(tmp_path)])
    assert "0.710" in capsys.readouterr().out
