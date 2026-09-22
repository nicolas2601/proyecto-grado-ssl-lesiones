"""The report's tables must come from the run, not from someone's memory.

A transcribed number survives a rerun that changed it. These tests pin the
generator that rewrites the tables from bank.json, and the --check mode that
fails when the document and the data disagree.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_results_readme as gen  # noqa: E402


def _bank(**overrides):
    rows = [
        {"backbone": "dinov2_public_vits14", "probe": "logreg", "fraction": 1.0,
         "seed": 0, "macro_f1": 0.58, "error": None},
        {"backbone": "dinov2_public_vits14", "probe": "logreg", "fraction": 1.0,
         "seed": 1, "macro_f1": 0.60, "error": None},
        {"backbone": "ssl_proposed_vits14", "probe": "logreg", "fraction": 1.0,
         "seed": 0, "macro_f1": 0.62, "error": None},
        {"backbone": "ssl_proposed_vits14", "probe": "logreg", "fraction": 0.1,
         "seed": 0, "macro_f1": 0.45, "error": None},
        {"backbone": "ssl_proposed_vits14", "probe": "logreg_balanced", "fraction": 1.0,
         "seed": 0, "macro_f1": 0.55, "error": None},
    ]
    payload = {"mode": "corrida real", "rows": rows}
    payload.update(overrides)
    return payload


def _readme(tmp_path, body="cuerpo previo\n"):
    path = tmp_path / "README.md"
    path.write_text(f"{body}{gen.BEGIN}\nviejo\n{gen.END}\ncola\n", encoding="utf-8")
    return path


def test_the_table_averages_the_seeds_of_each_backbone():
    lines = gen.classification_table(_bank())
    joined = "\n".join(lines)
    assert "0,590" in joined, "0.58 and 0.60 average to 0.59"
    assert "`ssl_proposed_vits14`" in joined


def test_the_backbones_are_ordered_by_the_headline_metric():
    lines = gen.classification_table(_bank())
    body = [line for line in lines if line.startswith("| `")]
    assert body[0].startswith("| `ssl_proposed_vits14`"), "0.62 comes before 0.59"


def test_a_failed_row_is_left_out_of_the_average():
    bank = _bank()
    bank["rows"].append({"backbone": "ssl_proposed_vits14", "probe": "logreg",
                         "fraction": 1.0, "seed": 2, "macro_f1": 0.0,
                         "error": "ValueError: boom"})
    joined = "\n".join(gen.classification_table(bank))
    assert "0,620" in joined, "the errored row must not drag the mean down"


def test_a_missing_regime_shows_a_dash_not_a_zero():
    joined = "\n".join(gen.classification_table(_bank()))
    public_row = [l for l in joined.splitlines() if "dinov2_public_vits14" in l][0]
    assert "| - |" in public_row


def test_an_empty_bank_says_so_instead_of_rendering_an_empty_table():
    lines = gen.classification_table({"rows": []})
    assert "Sin filas" in lines[0]


def test_the_strata_table_translates_the_verdict():
    strata = {
        "a": {"gap_macro_f1": 0.10, "worst_label": "dark", "gap_established": True},
        "b": {"gap_macro_f1": 0.09, "worst_label": "light", "gap_established": False},
        "c": {"gap_macro_f1": 0.08, "worst_label": "dark", "gap_established": None},
    }
    joined = "\n".join(gen.strata_table(strata))
    assert "| `a` | 0,100 | dark | si |" in joined
    assert "| `b` | 0,090 | light | no |" in joined
    assert "| `c` | 0,080 | dark | sin datos |" in joined
    assert "no es lo mismo que ausencia de brecha" in joined


def test_writing_replaces_only_the_block_between_the_markers(tmp_path):
    readme = _readme(tmp_path)
    updated = gen.replace_block(readme.read_text(encoding="utf-8"),
                                gen.render(_bank(), {}))
    assert updated.startswith("cuerpo previo")
    assert updated.rstrip().endswith("cola")
    assert "viejo" not in updated


def test_a_document_without_markers_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="marcadores"):
        gen.replace_block("sin marcadores", gen.render(_bank(), {}))


def test_check_mode_fails_when_the_document_is_stale(tmp_path, monkeypatch, capsys):
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "bank.json").write_text(json.dumps(_bank()), encoding="utf-8")
    readme = _readme(tmp_path)

    monkeypatch.setattr(sys, "argv", [
        "build_results_readme.py", "--bank", str(bank_dir),
        "--readme", str(readme), "--check",
    ])
    with pytest.raises(SystemExit) as excinfo:
        gen.main()
    assert excinfo.value.code == 1


def test_check_mode_passes_once_the_document_was_generated(tmp_path, monkeypatch):
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "bank.json").write_text(json.dumps(_bank()), encoding="utf-8")
    readme = _readme(tmp_path)

    monkeypatch.setattr(sys, "argv", [
        "build_results_readme.py", "--bank", str(bank_dir), "--readme", str(readme),
    ])
    gen.main()

    monkeypatch.setattr(sys, "argv", [
        "build_results_readme.py", "--bank", str(bank_dir),
        "--readme", str(readme), "--check",
    ])
    gen.main()  # no SystemExit


# ------------------------------------------------- the equal-compute picture
def _arm_run(mode="two_view_ssl", steps=6286, samples=804608, budget=12.0):
    return {
        "adaptation_mode": mode, "budget_gpu_hours": budget,
        "execution": {"steps_executed": steps, "samples_processed": samples},
    }


def _tone(tmp_path, arms):
    tone = tmp_path / "tone"
    for name, run in arms.items():
        (tone / name).mkdir(parents=True)
        (tone / name / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return tone


def test_the_training_table_shows_budget_and_samples_together(tmp_path):
    runs = {"ssl_plain_vits14": _arm_run()}
    row = [l for l in gen.training_table(runs) if l.startswith("| `")][0]
    assert "12,0" in row, "el presupuesto en horas-GPU"
    assert "804.608" in row, "y las muestras, en la misma fila"
    assert "6.286" in row


def test_the_paired_arm_shows_it_processed_fewer_samples(tmp_path):
    runs = {
        "ssl_plain_vits14": _arm_run(steps=6286, samples=804608),
        "ssl_proposed_vits14": _arm_run(
            mode="original_transformed_consistency", steps=3100, samples=396800
        ),
    }
    joined = "\n".join(gen.training_table(runs))
    assert "396.800" in joined
    assert "menos muestras que su" in joined, "la asimetria debe quedar explicada"


def test_the_samples_per_step_is_derived_not_transcribed():
    runs = {"x": _arm_run(steps=100, samples=12800)}
    row = [l for l in gen.training_table(runs) if l.startswith("| `")][0]
    assert "128,0" in row


def test_an_arm_without_execution_data_renders_dashes():
    row = [l for l in gen.training_table({"x": {"adaptation_mode": "two_view_ssl"}})
           if l.startswith("| `")][0]
    assert row.count("| - |") >= 1


def test_no_runs_yet_says_pending():
    assert "Pendiente" in gen.training_table({})[0]


def test_only_arms_with_a_manifest_are_read(tmp_path):
    tone = _tone(tmp_path, {"ssl_plain_vits14": _arm_run()})
    (tone / "ssl_proposed_vits14").mkdir()          # started, not finished
    assert set(gen.read_runs(tone)) == {"ssl_plain_vits14"}


# ------------------------------------------------- the full metric panel
def _metric_row(backbone, probe="logreg", fraction=1.0, seed=0, **metrics):
    row = {
        "backbone": backbone, "probe": probe, "fraction": fraction, "seed": seed,
        "error": None, "accuracy": 0.747, "balanced_accuracy": 0.618,
        "macro_f1": 0.595, "macro_precision": 0.598, "macro_recall": 0.618,
        "auc": 0.907, "ece": 0.162,
    }
    row.update(metrics)
    return row


def test_the_panel_reports_the_seven_metrics():
    bank = {"rows": [_metric_row("dinov2_public_vitb14")]}
    row = [l for l in gen.metrics_table(bank) if l.startswith("| `")][0]
    for value in ("0,747", "0,618", "0,595", "0,598", "0,907", "0,162"):
        assert value in row, f"falta {value}"


def test_the_panel_says_that_lower_ece_is_better():
    bank = {"rows": [_metric_row("x")]}
    joined = "\n".join(gen.metrics_table(bank))
    assert "En ECE menor es mejor" in joined, (
        "sin la direccion, un ECE alto se lee como si fuera bueno"
    )


def test_the_panel_uses_one_probe_for_every_backbone():
    """Best-probe-per-model inflates each row by a max over noisy estimates."""
    bank = {"rows": [
        _metric_row("a", probe="logreg", macro_f1=0.50),
        _metric_row("a", probe="knn", macro_f1=0.90),   # mejor, pero no es el fijado
    ]}
    row = [l for l in gen.metrics_table(bank) if l.startswith("| `")][0]
    assert "0,500" in row and "0,900" not in row


def test_the_panel_averages_the_seeds():
    bank = {"rows": [
        _metric_row("a", seed=0, accuracy=0.70),
        _metric_row("a", seed=1, accuracy=0.80),
    ]}
    row = [l for l in gen.metrics_table(bank) if l.startswith("| `")][0]
    assert "0,750" in row


def test_the_panel_drops_errored_rows():
    bank = {"rows": [
        _metric_row("a", accuracy=0.70),
        {**_metric_row("a", seed=1, accuracy=0.0), "error": "ValueError: boom"},
    ]}
    row = [l for l in gen.metrics_table(bank) if l.startswith("| `")][0]
    assert "0,700" in row


def test_a_nan_metric_becomes_a_dash_not_a_zero():
    bank = {"rows": [_metric_row("a", auc=float("nan"))]}
    row = [l for l in gen.metrics_table(bank) if l.startswith("| `")][0]
    assert "| - |" in row


def test_an_empty_bank_says_so():
    assert "Sin filas" in gen.metrics_table({"rows": []})[0]
