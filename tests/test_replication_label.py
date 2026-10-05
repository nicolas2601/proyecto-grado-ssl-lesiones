"""Three seeds that never differ are one measurement, and the table must say so.

The bank runs every backbone at three seeds. The seed controls the label
subsample and nothing else, so at ``fraction=1.0`` - every label used - there is
nothing to subsample and the three runs return the same numbers to the last
decimal. The committee reading "promedio de semillas" over that row would read
a replication that was never performed, and would reasonably take the spread as
evidence the difference between backbones is real.

This is the rotulo failure this repo has already had twice: a number that is
correct next to a label that is not. These tests pin the label to the data.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_results_readme import (  # noqa: E402
    metrics_table,
    replication,
    replication_note,
)

PROBE = "logreg_balanced"


def _row(backbone: str, seed: int, macro_f1: float, **rest) -> dict:
    row = {
        "backbone": backbone,
        "probe": PROBE,
        "fraction": 1.0,
        "seed": seed,
        "macro_f1": macro_f1,
        "accuracy": 0.7,
        "balanced_accuracy": 0.63,
        "macro_precision": 0.6,
        "macro_recall": 0.61,
        "auc": 0.92,
        "ece": 0.03,
    }
    row.update(rest)
    return row


def _identical_seeds() -> list[dict]:
    """What the bank actually produced at 100 %: sd = 0.0000 in every row."""
    return [_row("ssl_proposed_vitb14", seed, 0.5368) for seed in (0, 1, 2)]


def test_three_identical_seeds_are_not_reported_as_a_mean():
    n_seeds, varies = replication(_identical_seeds(), PROBE, 1.0)
    assert n_seeds == 3
    assert varies is False


def test_the_note_says_the_effective_n_is_one():
    note = replication_note(3, False, 1.0)
    assert "n efectivo es 1" in note
    assert "bootstrap" in note


def test_seeds_that_do_differ_are_reported_as_an_average():
    rows = [
        _row("ssl_plain_vits14", seed, f1, fraction=0.1)
        for seed, f1 in ((0, 0.50), (1, 0.53), (2, 0.56))
    ]
    n_seeds, varies = replication(rows, PROBE, 0.1)
    assert (n_seeds, varies) == (3, True)
    assert "difieren" in replication_note(n_seeds, varies, 0.1)


def test_a_single_seed_claims_nothing():
    n_seeds, varies = replication([_row("dinov2_public_vitb14", 0, 0.61)], PROBE, 1.0)
    assert n_seeds == 1
    assert "no hay replicacion" in replication_note(n_seeds, varies, 1.0)


def test_the_published_table_carries_the_warning_not_the_old_wording():
    """The failure is silent: the table renders fine either way."""
    rendered = "\n".join(metrics_table({"rows": _identical_seeds()}, PROBE, 1.0))
    assert "promedio de semillas" not in rendered
    assert "n efectivo es 1" in rendered


def test_one_metric_moving_is_enough_to_count_as_variation():
    """Variation anywhere refutes 'the same run repeated'."""
    rows = _identical_seeds()
    rows[2] = dict(rows[2], ece=0.0361)
    _, varies = replication(rows, PROBE, 1.0)
    assert varies is True


def test_rows_from_another_probe_do_not_leak_into_the_verdict():
    rows = _identical_seeds() + [
        dict(_row("ssl_proposed_vitb14", seed, f1), probe="knn")
        for seed, f1 in ((0, 0.40), (1, 0.44), (2, 0.48))
    ]
    _, varies = replication(rows, PROBE, 1.0)
    assert varies is False, "el veredicto se contamino con filas de otro probe"


def test_failed_rows_are_not_counted_as_replication():
    rows = _identical_seeds() + [dict(_row("ssl_proposed_vitb14", 7, 0.9), error="OOM")]
    n_seeds, varies = replication(rows, PROBE, 1.0)
    assert n_seeds == 3 and varies is False
