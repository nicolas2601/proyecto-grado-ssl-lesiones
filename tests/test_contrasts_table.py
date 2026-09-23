"""The confidence bands must be read from bootstrap.json, never typed.

A point estimate without an interval invites reading noise as a result. The
contrasts table is the interval, and it is generated from the file so the
document cannot claim a win the resampling did not support. These tests pin the
verdict to the data: includes zero -> noise, excludes zero -> the named side.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_results_readme as gen  # noqa: E402


def _row(arm, control, metric, delta, low, high, direction=None, **extra):
    excludes = not (low <= 0 <= high)
    if direction is None:
        direction = "indistinguishable" if not excludes else (
            "arm_better" if delta > 0 else "control_better")
    row = {
        "arm": arm, "control": control, "metric": metric,
        "point_delta": delta, "ci95_low": low, "ci95_high": high,
        "excludes_zero": excludes, "direction": direction,
    }
    row.update(extra)
    return row


def test_an_interval_that_includes_zero_is_noise_not_a_win():
    row = _row("ssl_proposed_vitb14", "ssl_plain_vitb14", "auc", -0.0013, -0.0070, 0.0054)
    assert gen._verdict(row) == "no se distingue del ruido"


def test_the_verdict_reads_direction_not_the_sign_of_the_delta():
    # ECE lower-is-better: a NEGATIVE delta is the arm winning, and the file
    # says so via direction. Inferring from the sign would flip it.
    row = _row("ssl_plain_vitb14", "dinov2_public_vitb14", "ece",
               -0.1146, -0.1239, -0.0882, direction="arm_better")
    assert gen._verdict(row) == "gana `ssl_plain_vitb14`"


def test_control_better_names_the_control():
    row = _row("ssl_proposed_vitb14", "ssl_plain_vitb14", "macro_f1",
               -0.0619, -0.1131, -0.0109, direction="control_better")
    assert gen._verdict(row) == "gana `ssl_plain_vitb14`"


def test_a_positive_excluding_interval_names_the_arm():
    row = _row("ssl_plain_vitb14", "dinov2_public_vitb14", "auc",
               0.0190, 0.0074, 0.0320, direction="arm_better")
    assert gen._verdict(row) == "gana `ssl_plain_vitb14`"


def test_the_table_renders_every_contrast_the_file_carries():
    boot = {"rng_seed": 20260923, "n_resamples": 1000, "results": [
        _row("ssl_proposed_vitb14", "ssl_plain_vitb14", "auc", -0.0013, -0.0070, 0.0054),
        _row("ssl_proposed_vitb14", "ssl_plain_vitb14", "macro_f1",
             -0.0619, -0.1131, -0.0109, direction="control_better"),
    ]}
    table = "\n".join(gen.contrasts_table(boot))
    assert "auc" in table and "macro_f1" in table
    assert "no se distingue del ruido" in table
    assert "gana `ssl_plain_vitb14`" in table
    assert "semilla 20260923" in table
    assert "1.000 resamples" in table


def test_the_container_key_can_be_rows_or_a_bare_list():
    rows = [_row("a", "b", "auc", 0.02, 0.01, 0.03)]
    assert len(gen._contrast_rows({"rows": rows})) == 1
    assert len(gen._contrast_rows(rows)) == 1
    assert gen._contrast_rows({}) == []


def test_a_missing_bootstrap_leaves_a_pending_note_not_a_crash():
    assert "Pendiente" in gen.contrasts_table({})[0]


def test_the_footnote_states_the_zero_rule_so_nobody_cites_a_point_alone():
    boot = {"contrasts": [_row("a", "b", "auc", 0.02, 0.01, 0.03)]}
    table = "\n".join(gen.contrasts_table(boot))
    assert "incluye el cero" in table
    assert "excluya el cero" in table
