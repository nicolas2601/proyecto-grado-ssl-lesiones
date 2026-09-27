"""Per-cell gates for the 24-cell C1 grid.

Each cell is one (backbone, condition) pair. Two conditions can degrade in a
way that still produces a full table of plausible numbers:

* ``mask_pool`` falls back to the CLS token when the mask selects no patch.
  Enough fallbacks and the cell is no longer measuring mask pooling at all --
  it is measuring condition (0) with extra steps, and the comparison silently
  becomes "the baseline against itself".
* ``neutral_bg`` on an empty mask greys out the WHOLE image. The backbone then
  sees a flat rectangle, the probe still returns a prediction, and nothing in
  the pipeline raises.

So both are counted per image and gated per cell, at 1%.
"""

from __future__ import annotations

import pytest

from dermssl import c1_cell_gates as cg


class TestCellTally:
    def test_counts_are_per_image_and_start_at_zero(self):
        tally = cg.CellTally(backbone="dinov2_public_vitb14", condition="mask_pool")

        assert tally.n_images == 0
        assert tally.n_cls_fallback == 0
        assert tally.n_empty_after_geometry == 0

    def test_recording_an_image_advances_the_denominator(self):
        tally = cg.CellTally(backbone="b", condition="mask_pool")

        tally.record(used_cls_fallback=False, empty_after_geometry=False)
        tally.record(used_cls_fallback=True, empty_after_geometry=False)

        assert tally.n_images == 2
        assert tally.n_cls_fallback == 1

    def test_an_empty_mask_is_counted_separately_from_a_fallback(self):
        """They are different failures with different owners: an empty mask is
        the segmenter, a CLS fallback is the pooling."""
        tally = cg.CellTally(backbone="b", condition="neutral_bg")

        tally.record(used_cls_fallback=False, empty_after_geometry=True)

        assert tally.n_empty_after_geometry == 1
        assert tally.n_cls_fallback == 0

    def test_fractions_over_an_empty_cell_are_none_not_zero(self):
        """Zero images satisfies every threshold vacuously; 0.0 would read as
        'measured and clean'."""
        tally = cg.CellTally(backbone="b", condition="mask_pool")

        assert tally.fraction_cls_fallback is None
        assert tally.fraction_empty_after_geometry is None


class TestGatesPerCondition:
    def _tally(self, condition, n_images, *, fallbacks=0, empties=0):
        tally = cg.CellTally(backbone="b", condition=condition)
        for index in range(n_images):
            tally.record(
                used_cls_fallback=index < fallbacks,
                empty_after_geometry=empties > index >= 0 and index < empties,
            )
        return tally

    def test_a_clean_mask_pool_cell_passes(self):
        tally = self._tally("mask_pool", 1000)

        passes, reasons = cg.evaluate_cell_gates(tally)

        assert passes is True
        assert reasons == []

    def test_mask_pool_stops_above_one_percent_of_cls_fallback(self):
        tally = self._tally("mask_pool", 1000, fallbacks=11)

        passes, reasons = cg.evaluate_cell_gates(tally)

        assert passes is False
        assert any("fallback" in reason for reason in reasons)
        assert any("1,1" in reason or "1.1" in reason for reason in reasons)

    def test_exactly_one_percent_still_passes(self):
        """The gate is '> 1 %', so the boundary is not a failure. Pinned
        because '>' and '>=' are one keystroke apart and the difference only
        shows up on the day it matters."""
        tally = self._tally("mask_pool", 1000, fallbacks=10)

        passes, _ = cg.evaluate_cell_gates(tally)

        assert passes is True

    def test_neutral_bg_stops_above_one_percent_of_degenerate_images(self):
        tally = self._tally("neutral_bg", 1000, empties=11)

        passes, reasons = cg.evaluate_cell_gates(tally)

        assert passes is False
        assert any("gris" in reason or "vacia" in reason for reason in reasons)

    def test_a_cls_fallback_outside_mask_pool_is_a_contradiction(self):
        """Only mask_pool pools patch tokens, so a fallback anywhere else means
        the condition being run is not the condition being labelled."""
        tally = self._tally("none", 100, fallbacks=1)

        passes, reasons = cg.evaluate_cell_gates(tally)

        assert passes is False
        assert any("none" in reason for reason in reasons)

    def test_an_empty_cell_fails_instead_of_passing_vacuously(self):
        tally = cg.CellTally(backbone="b", condition="mask_pool")

        passes, reasons = cg.evaluate_cell_gates(tally)

        assert passes is False
        assert any("vacuo" in reason or "cero imagenes" in reason for reason in reasons)

    def test_every_failing_condition_is_reported_not_just_the_first(self):
        """Two independent failures in one cell must both surface.

        neutral_bg, because there the two ARE independent: it cannot pool
        patches (so any fallback is a mislabelled condition) and it can be
        greyed out whole (so empties are its own failure).
        """
        tally = cg.CellTally(backbone="b", condition="neutral_bg")
        for index in range(1000):
            tally.record(used_cls_fallback=index < 50, empty_after_geometry=index < 40)

        passes, reasons = cg.evaluate_cell_gates(tally)

        assert passes is False
        assert len(reasons) >= 2, "solo se reporto el primer fallo"

    def test_mask_pool_does_not_count_an_empty_mask_twice(self):
        """In mask_pool an empty mask IS a CLS fallback -- it selects no patch.

        Gating both would report one failure as two and make the cell look
        twice as broken as it is. The count is still REPORTED in the summary;
        what it is not is a second gate.
        """
        tally = cg.CellTally(backbone="b", condition="mask_pool")
        for index in range(1000):
            tally.record(used_cls_fallback=index < 40, empty_after_geometry=index < 40)

        passes, reasons = cg.evaluate_cell_gates(tally)

        assert passes is False
        assert len(reasons) == 1, f"el mismo fallo se conto dos veces: {reasons}"
        assert cg.summarise_cell(tally)["n_empty_after_geometry"] == 40

    def test_neutral_bg_does_not_gate_on_fallback_it_cannot_have(self):
        """neutral_bg reads the pooled output, so its fallback count is
        structurally zero and gating on it would be a gate that can never
        fire -- reported as not applicable instead."""
        tally = self._tally("neutral_bg", 1000)

        summary = cg.summarise_cell(tally)

        assert summary["cls_fallback"] == "not_applicable: neutral_bg no agrupa parches"

    def test_an_unknown_condition_is_refused(self):
        with pytest.raises(ValueError, match="condicion"):
            cg.CellTally(backbone="b", condition="inventada")


class TestSummaryIsReportable:
    def test_summary_carries_both_numerator_and_denominator(self):
        """A rate whose denominator nobody states is how a gate ends up
        measured against the wrong population."""
        tally = cg.CellTally(backbone="dinov2_public_vitb14", condition="mask_pool")
        for index in range(500):
            tally.record(used_cls_fallback=index < 3, empty_after_geometry=False)

        summary = cg.summarise_cell(tally)

        assert summary["n_images"] == 500
        assert summary["n_cls_fallback"] == 3
        assert summary["fraction_cls_fallback"] == pytest.approx(0.006)
        assert summary["backbone"] == "dinov2_public_vitb14"
        assert summary["condition"] == "mask_pool"

    def test_summary_is_json_serialisable_for_wandb(self):
        import json

        tally = cg.CellTally(backbone="b", condition="bbox_crop")
        tally.record(used_cls_fallback=False, empty_after_geometry=False)

        json.dumps(cg.summarise_cell(tally))


class TestEmptyAfterGeometryComesFromTheMaskItself:
    """The flag is derived, so it cannot disagree with the mask it describes."""

    def _prepared(self, mask):
        import numpy as np

        from dermssl import c1_extraction as ex

        return ex.PreparedInput(
            image=np.zeros((224, 224, 3), dtype=np.uint8),
            mask=mask,
            used_bbox_fallback=False,
        )

    def test_an_empty_mask_is_flagged(self):
        import numpy as np

        assert self._prepared(np.zeros((224, 224), dtype=bool)).empty_after_geometry is True

    def test_a_single_surviving_pixel_is_not_empty(self):
        import numpy as np

        mask = np.zeros((224, 224), dtype=bool)
        mask[100, 100] = True

        assert self._prepared(mask).empty_after_geometry is False

    def test_a_lesion_cropped_away_by_the_geometry_is_flagged(self):
        """The real path: a lesion that exists in the original grid and does
        not survive Resize+CenterCrop."""
        import numpy as np

        from dermssl import c1_extraction as ex

        original = np.zeros((450, 600), dtype=bool)
        original[0:6, 0:6] = True  # corner, outside what the centre crop keeps

        survived = ex._baseline_geometry_mask(original)

        assert not survived.any(), "el escenario no recorta la lesion"
        assert self._prepared(survived).empty_after_geometry is True


class TestFallbackCauseBreakdown:
    """Which of the two owners caused each CLS fallback.

    Reported, never gated: the gate is still the total, because the cell
    degrades the same either way. But if condition (c) performs badly we need
    to know whether to look at the segmenter or at the lesion size against the
    14x14 grid -- those are different people and different fixes.
    """

    def test_an_empty_mask_attributes_the_fallback_to_the_segmenter(self):
        tally = cg.CellTally(backbone="b", condition="mask_pool")

        tally.record(used_cls_fallback=True, empty_after_geometry=True)

        assert tally.n_fallback_from_empty_mask == 1
        assert tally.n_fallback_from_small_mask == 0

    def test_a_non_empty_mask_attributes_it_to_the_lesion_size(self):
        """The mask survived the geometry but no 14x14 patch is covered
        enough to be selected."""
        tally = cg.CellTally(backbone="b", condition="mask_pool")

        tally.record(used_cls_fallback=True, empty_after_geometry=False)

        assert tally.n_fallback_from_small_mask == 1
        assert tally.n_fallback_from_empty_mask == 0

    def test_the_two_causes_add_up_to_the_total(self):
        """They are exhaustive and mutually exclusive by construction; if they
        ever stop adding up, one of them is being counted somewhere else."""
        tally = cg.CellTally(backbone="b", condition="mask_pool")
        for index in range(100):
            tally.record(
                used_cls_fallback=index < 30,
                empty_after_geometry=index < 12,
            )

        assert tally.n_fallback_from_empty_mask == 12
        assert tally.n_fallback_from_small_mask == 18
        assert (
            tally.n_fallback_from_empty_mask + tally.n_fallback_from_small_mask
            == tally.n_cls_fallback
        )

    def test_an_empty_mask_without_a_fallback_is_a_contradiction(self):
        """In mask_pool an empty mask cannot select a patch, so it MUST fall
        back. Recording otherwise means the caller is not reporting what the
        extraction actually did."""
        tally = cg.CellTally(backbone="b", condition="mask_pool")

        with pytest.raises(ValueError, match="vacia"):
            tally.record(used_cls_fallback=False, empty_after_geometry=True)

    def test_the_breakdown_reaches_the_summary(self):
        tally = cg.CellTally(backbone="b", condition="mask_pool")
        for index in range(200):
            tally.record(used_cls_fallback=index < 6, empty_after_geometry=index < 2)

        summary = cg.summarise_cell(tally)

        assert summary["n_fallback_from_empty_mask"] == 2
        assert summary["n_fallback_from_small_mask"] == 4

    def test_conditions_that_cannot_pool_do_not_report_a_breakdown(self):
        """A zero breakdown on neutral_bg would read as a measurement that came
        out clean, for something that cannot happen there."""
        tally = cg.CellTally(backbone="b", condition="neutral_bg")
        tally.record(used_cls_fallback=False, empty_after_geometry=True)

        summary = cg.summarise_cell(tally)

        assert "n_fallback_from_empty_mask" not in summary
        assert summary["cls_fallback"].startswith("not_applicable")
