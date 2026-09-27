"""The grid runner's pure parts, tested without a GPU.

What is worth testing here is not the extraction -- that is torch, and it has
its own gate against the bank. It is the logic around it that decides which
numbers reach the table: which cells get contrasted, against what, and what
happens to the ones that did not pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import c1_run_grid as grid  # noqa: E402


def _proba(n_rows: int, n_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n_rows, n_classes))
    return raw / raw.sum(axis=1, keepdims=True)


def _cell(passes: bool, seed: int, n_rows: int = 40, n_classes: int = 3) -> dict:
    return {
        "proba": _proba(n_rows, n_classes, seed),
        "gate": {"gate_passes": passes},
        "metrics": {"macro_f1": 0.5},
        "passes": passes,
    }


INDEX = {"c1_val_ids_sha256": "a" * 64}


class TestWhichCellsGetContrasted:
    def test_every_condition_is_contrasted_against_zero(self):
        results = {
            ("b", "none"): _cell(True, 1),
            ("b", "mask_pool"): _cell(True, 2),
            ("b", "neutral_bg"): _cell(True, 3),
        }

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        arms = {r["arm"] for r in document["results"]}
        assert arms == {"b__mask_pool", "b__neutral_bg"}
        assert {r["control"] for r in document["results"]} == {"b__none"}

    def test_the_baseline_is_not_contrasted_with_itself(self):
        results = {("b", "none"): _cell(True, 1)}

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        assert document["results"] == []

    def test_all_four_metrics_are_recorded_per_contrast(self):
        """The metric is fixed before the data, so the table cannot be
        filtered afterwards to whichever one favours a conclusion."""
        results = {("b", "none"): _cell(True, 1), ("b", "mask_pool"): _cell(True, 2)}

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        assert {r["metric"] for r in document["results"]} == set(grid.METRICS)

    def test_a_backbone_without_its_baseline_is_reported_not_silently_dropped(self):
        results = {("b", "mask_pool"): _cell(True, 2)}

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        assert document["results"] == []
        assert any("no hay celda (0)" in reason for reason in document["not_contrasted"])

    def test_backbones_are_never_crossed(self):
        """Each condition is contrasted against ITS OWN backbone's (0).

        Checked on the DATA, not on the label. A mutant that picks the first
        available (0) still writes `control: "b2__none"` in the record, so a
        test comparing names passes while the numbers come from another
        backbone -- a contrast that is correctly labelled and wrong, which is
        the worst shape a result can have.

        So the recorded point_control is compared against the metric computed
        on the baseline that SHOULD have been used.
        """
        from dermssl import paired_bootstrap as pb

        y_true = np.arange(40) % 3
        results = {
            ("b1", "none"): _cell(True, 1),
            ("b2", "none"): _cell(True, 2),
            ("b2", "mask_pool"): _cell(True, 3),
        }

        document = grid.build_contrasts(results, y_true, INDEX)
        rows = [r for r in document["results"] if r["metric"] == "macro_f1"]
        assert rows, "el escenario no produjo el contraste que se quiere verificar"

        expected = pb.metric_macro_f1(y_true, results[("b2", "none")]["proba"])
        wrong = pb.metric_macro_f1(y_true, results[("b1", "none")]["proba"])
        assert expected != wrong, "las dos lineas base son iguales: el test no discrimina"

        for record in rows:
            assert record["arm"].split("__")[0] == record["control"].split("__")[0]
            assert record["point_control"] == pytest.approx(expected), (
                "el contraste uso los datos de otro backbone como linea base"
            )


class TestAFailedGateStaysOutOfTheTable:
    def test_a_degraded_arm_is_not_contrasted(self):
        """A mask_pool cell that fell back to CLS is partly the baseline, so
        contrasting it against the baseline reports a tie that means nothing
        and reads like a finding."""
        results = {("b", "none"): _cell(True, 1), ("b", "mask_pool"): _cell(False, 2)}

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        assert document["results"] == []
        assert any("no paso su gate" in reason for reason in document["not_contrasted"])

    def test_a_degraded_baseline_disqualifies_its_whole_column(self):
        results = {
            ("b", "none"): _cell(False, 1),
            ("b", "mask_pool"): _cell(True, 2),
            ("b", "neutral_bg"): _cell(True, 3),
        }

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        assert document["results"] == []
        assert len(document["not_contrasted"]) == 2

    def test_what_was_not_contrasted_is_written_down(self):
        """An absent row would read as a question nobody asked, rather than
        one that could not be answered."""
        results = {("b", "none"): _cell(True, 1), ("b", "mask_pool"): _cell(False, 2)}

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        assert "not_contrasted" in document
        assert document["not_contrasted"], "no se registro por que falta el contraste"


class TestTheDocumentShape:
    def test_it_carries_the_split_digest_it_was_computed_on(self):
        results = {("b", "none"): _cell(True, 1), ("b", "mask_pool"): _cell(True, 2)}

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)

        assert document["val_order_sha256"] == INDEX["c1_val_ids_sha256"]
        assert document["n_val"] == 40
        assert document["paired"] is True

    def test_it_is_json_serialisable(self):
        import json

        results = {("b", "none"): _cell(True, 1), ("b", "mask_pool"): _cell(True, 2)}

        json.dumps(grid.build_contrasts(results, np.arange(40) % 3, INDEX))

    def test_the_ece_row_keeps_its_direction(self):
        results = {("b", "none"): _cell(True, 1), ("b", "mask_pool"): _cell(True, 2)}

        document = grid.build_contrasts(results, np.arange(40) % 3, INDEX)
        ece_rows = [r for r in document["results"] if r["metric"] == "ece"]

        assert ece_rows
        for row in ece_rows:
            assert row["direction"] in {"arm_better", "control_better", "indistinguishable"}


class TestTheRunOrder:
    def test_the_baseline_condition_comes_first(self):
        """Every contrast needs (0), so a grid that dies early must already
        have it."""
        assert grid.CONDITION_ORDER[0] == "none"

    def test_the_priority_backbones_are_the_professors_question(self):
        assert set(grid.PRIORITY_BACKBONES) == {
            "dinov2_public_vitb14",
            "ssl_proposed_vitb14",
        }


class TestLimitSubsamplesInsteadOfTruncating:
    def test_it_spans_the_whole_list(self):
        picks = grid._stride(8011, 40)

        assert len(picks) == 40
        assert picks[0] == 0 and picks[-1] == 8010

    def test_no_limit_keeps_everything(self):
        assert len(grid._stride(100, None)) == 100
