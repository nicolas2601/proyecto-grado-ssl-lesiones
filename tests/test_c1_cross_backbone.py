"""Cross-backbone C1 contrasts and the crop-retention strata, without a GPU."""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import c1_cross_backbone as cb
from dermssl import paired_bootstrap as pb

N = 120
Y = np.arange(N) % 3


def _proba(seed: int, bias: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((N, 3))
    raw[np.arange(N), Y] += bias
    return raw / raw.sum(axis=1, keepdims=True)


def _manifest(**over) -> dict:
    base = {"c1_val_ids_sha256": "v", "labels_sha256": "l", "masks_sha256": "m", "probe": "logreg"}
    return {**base, **over}


def _cells(**bias) -> dict:
    names = [a for pair in cb.CROSS_PAIRS for a in pair]
    return {name: _proba(i, bias.get(name, 0.0)) for i, name in enumerate(dict.fromkeys(names))}


class TestPairing:
    def test_same_images_and_labels_are_pairable(self):
        cb.check_pairable({"a": _manifest(), "b": _manifest()})

    @pytest.mark.parametrize("field", cb.PAIRING_FIELDS)
    def test_any_differing_field_refuses(self, field):
        with pytest.raises(ValueError, match=field):
            cb.check_pairable({"a": _manifest(), "b": _manifest(**{field: "other"})})

    def test_a_missing_field_refuses(self):
        broken = _manifest()
        del broken["labels_sha256"]
        with pytest.raises(ValueError, match="labels_sha256"):
            cb.check_pairable({"a": _manifest(), "b": broken})

    def test_no_cells_refuses(self):
        with pytest.raises(ValueError):
            cb.check_pairable({})


class TestCrossContrasts:
    def test_the_professors_question_is_the_first_pair(self):
        assert cb.CROSS_PAIRS[0] == (f"{cb.PUBLIC}__mask_pool", f"{cb.PROPOSED}__none")

    def test_every_pair_crosses_backbones_against_proposed_zero(self):
        for arm, control in cb.CROSS_PAIRS:
            assert arm.startswith(cb.PUBLIC) and control == f"{cb.PROPOSED}__none"

    def test_same_condition_pairs_hold_the_input_equal_on_both_sides(self):
        conditions = {arm.rsplit("__", 1)[1] for arm, _ in cb.SAME_CONDITION_PAIRS}
        assert conditions == {"bbox_crop", "neutral_bg", "mask_pool"}
        for arm, control in cb.SAME_CONDITION_PAIRS:
            assert arm.startswith(cb.PUBLIC) and control.startswith(cb.PROPOSED)
            assert arm.rsplit("__", 1)[1] == control.rsplit("__", 1)[1]

    def test_the_zero_vs_zero_reference_is_not_duplicated(self):
        assert len(set(cb.ALL_CROSS_PAIRS)) == len(cb.ALL_CROSS_PAIRS) == 7

    def test_all_four_metrics_per_pair_on_the_right_data(self):
        cells = _cells()
        passes = {name: True for name in cells}

        records, skipped = cb.contrast_pairs(cells, passes, Y, cb.CROSS_PAIRS)

        assert skipped == []
        assert len(records) == len(cb.CROSS_PAIRS) * 4
        for record in records:
            expected = pb.METRIC_FUNCTIONS[record["metric"]](Y, cells[record["arm"]])
            assert record["point_arm"] == pytest.approx(expected)

    def test_a_clearly_better_arm_is_reported_as_arm_better(self):
        arm, control = cb.CROSS_PAIRS[3]
        cells = {arm: _proba(1, bias=3.0), control: _proba(2)}
        records, _ = cb.contrast_pairs(cells, {arm: True, control: True}, Y, [(arm, control)])

        f1 = next(r for r in records if r["metric"] == "macro_f1")
        assert f1["direction"] == "arm_better"

    def test_a_failed_gate_is_listed_not_dropped(self):
        cells = _cells()
        passes = {name: True for name in cells}
        passes[cb.CROSS_PAIRS[0][0]] = False

        records, skipped = cb.contrast_pairs(cells, passes, Y, cb.CROSS_PAIRS)

        assert all(r["arm"] != cb.CROSS_PAIRS[0][0] for r in records)
        assert any("gate" in reason for reason in skipped)


class TestStrata:
    def test_empty_masks_are_their_own_stratum(self):
        strata = cb.retention_strata(np.array([1.0, 0.95, 0.9, 0.5, np.nan]))

        assert strata["retained_ge_0.9"].tolist() == [0, 1, 2]
        assert strata["retained_lt_0.9"].tolist() == [3]
        assert strata["empty_mask"].tolist() == [4]

    def test_a_tiny_stratum_is_reported_by_size_only(self):
        retained = np.ones(N)
        retained[:2] = np.nan
        cells = _cells()

        summary = cb.stratified_summary(cells, {n: True for n in cells}, Y, retained)
        empty = next(s for s in summary if s["stratum"] == "empty_mask")

        assert empty["n"] == 2 and empty["results"] == [] and empty["not_contrasted"]

    def test_stratum_metrics_are_computed_on_its_rows_only(self):
        # Odd rows only: a prefix or suffix would hide a slice taken by position.
        rows = np.arange(1, N, 2)
        retained = np.full(N, 0.1)
        retained[rows] = 1.0
        cells = _cells()
        name = cb.CROSS_PAIRS[0][0]

        summary = cb.stratified_summary(cells, {n: True for n in cells}, Y, retained)
        high = next(s for s in summary if s["stratum"] == "retained_ge_0.9")

        expected = pb.metric_macro_f1(Y[rows], cells[name][rows])
        assert expected != pytest.approx(pb.metric_macro_f1(Y[: len(rows)], cells[name][: len(rows)]))
        assert high["n"] == len(rows)
        assert high["cells"][name]["macro_f1"] == pytest.approx(expected)

    def test_misaligned_retention_refuses(self):
        cells = _cells()
        with pytest.raises(ValueError):
            cb.stratified_summary(cells, {n: True for n in cells}, Y, np.ones(N - 1))

    def test_within_backbone_pairs_use_their_own_zero(self):
        pairs = cb.within_backbone_pairs(["a__none", "a__mask_pool", "b__none", "b__bbox_crop"])
        assert pairs == [("a__mask_pool", "a__none"), ("b__bbox_crop", "b__none")]
