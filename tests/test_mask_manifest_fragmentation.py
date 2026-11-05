"""Tests for the fragmentation abstention rule in the mask contract.

The rule exists because repairing fragmentation was measured and refuted:
keeping only the largest connected component makes Dice worse against the
ISIC 2018 expert masks. Abstaining is what works. These tests pin the
behaviour and, above all, pin that the rule is opt-in.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl.mask_manifest import (
    DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
    MIN_SIGNIFICANT_COMPONENT_FRACTION,
    build_mask_manifest,
    build_mask_record,
    count_significant_components,
)

RECORD_KWARGS = {
    "dataset": "pad_adapt",
    "image_id": "PAT_0001",
    "source_revision": "rev0",
    "producer": "teacher-026",
    "version": "v1",
}


def _single_blob(size: int = 64) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[20:44, 20:44] = True
    return mask


def _two_blobs(size: int = 64) -> np.ndarray:
    """Two components of comparable size, far apart."""
    mask = np.zeros((size, size), dtype=bool)
    mask[6:22, 6:22] = True
    mask[40:56, 40:56] = True
    return mask


def _blob_with_speckle(size: int = 64) -> np.ndarray:
    """One real component plus a single stray pixel (far below 0,5 %)."""
    mask = _single_blob(size)
    mask[2, 2] = True
    return mask


# --- the rule is opt-in: the default path must not move ------------------


def test_default_path_is_unchanged_and_counts_nothing():
    record = build_mask_record(mask=_two_blobs(), **RECORD_KWARGS)
    assert record["eligible"] is True
    assert record["quarantine_reason"] is None
    assert "significant_component_count" not in record


def test_default_path_matches_explicit_none_byte_for_byte():
    mask = _two_blobs()
    implicit = build_mask_record(mask=mask, **RECORD_KWARGS)
    explicit = build_mask_record(
        mask=mask, max_significant_components=None, **RECORD_KWARGS
    )
    assert implicit == explicit


def test_enabling_the_rule_does_not_touch_an_intact_mask():
    mask = _single_blob()
    off = build_mask_record(mask=mask, **RECORD_KWARGS)
    on = build_mask_record(
        mask=mask,
        max_significant_components=DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
        **RECORD_KWARGS,
    )
    assert on["eligible"] is True
    assert on["quarantine_reason"] is None
    assert on["significant_component_count"] == 1
    assert {k: v for k, v in on.items() if k != "significant_component_count"} == off


# --- the rule, when switched on ------------------------------------------


def test_fragmented_mask_is_quarantined_when_the_rule_is_on():
    record = build_mask_record(
        mask=_two_blobs(),
        max_significant_components=DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
        **RECORD_KWARGS,
    )
    assert record["eligible"] is False
    assert record["quarantine_reason"] == "fragmented"
    assert record["significant_component_count"] == 2
    assert record["quality"] == "degenerate"


def test_speckle_does_not_count_as_fragmentation():
    """Despeckling measured +0,0000 on Dice, so speckle must not abstain."""
    record = build_mask_record(
        mask=_blob_with_speckle(),
        max_significant_components=DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
        **RECORD_KWARGS,
    )
    assert record["significant_component_count"] == 1
    assert record["eligible"] is True


def test_a_higher_budget_admits_more_components():
    record = build_mask_record(
        mask=_two_blobs(), max_significant_components=2, **RECORD_KWARGS
    )
    assert record["eligible"] is True
    assert record["significant_component_count"] == 2


def test_rejects_a_budget_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        build_mask_record(
            mask=_single_blob(), max_significant_components=0, **RECORD_KWARGS
        )


# --- precedence: an already-quarantined mask keeps its original reason ----


def test_low_confidence_still_wins_over_fragmentation():
    record = build_mask_record(
        mask=_two_blobs(),
        confidence=0.10,
        max_significant_components=DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
        **RECORD_KWARGS,
    )
    assert record["quarantine_reason"] == "low_confidence"


def test_degenerate_area_still_wins_over_fragmentation():
    tiny = np.zeros((64, 64), dtype=bool)
    tiny[0, 0] = True
    tiny[40, 40] = True
    record = build_mask_record(
        mask=tiny,
        max_significant_components=DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
        **RECORD_KWARGS,
    )
    assert record["eligible"] is False
    assert record["quarantine_reason"] == "degenerate_area"


# --- component counting --------------------------------------------------


def test_empty_mask_has_zero_components():
    assert count_significant_components(np.zeros((8, 8), dtype=bool)) == 0


def test_components_are_four_connected():
    """Diagonal touch is two components, not one."""
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:6, 2:6] = True
    mask[6:10, 6:10] = True
    assert count_significant_components(mask) == 2


def test_threshold_is_read_against_lesion_area_not_frame():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:10, :10] = True  # 100 px of lesion
    mask[90, 90] = True  # 1 px -> 1 % of lesion area, above the 0,5 % floor
    assert MIN_SIGNIFICANT_COMPONENT_FRACTION == 0.005
    assert count_significant_components(mask) == 2
    assert count_significant_components(mask, min_fraction=0.05) == 1


# --- the manifest must make the coverage cost visible --------------------


def test_manifest_reports_coverage_and_reasons():
    rows = [
        build_mask_record(
            mask=_single_blob(),
            max_significant_components=DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
            **{**RECORD_KWARGS, "image_id": "ok"},
        ),
        build_mask_record(
            mask=_two_blobs(),
            max_significant_components=DEFAULT_MAX_SIGNIFICANT_COMPONENTS,
            **{**RECORD_KWARGS, "image_id": "frag"},
        ),
    ]
    manifest = build_mask_manifest(rows)
    summary = manifest["summary"]
    assert summary["total"] == 2
    assert summary["eligible"] == 1
    assert summary["quarantined"] == 1
    assert summary["coverage_fraction"] == 0.5
    assert summary["quarantined_by_reason"] == {"fragmented": 1}


def test_manifest_without_quarantine_omits_the_reason_block():
    rows = [build_mask_record(mask=_single_blob(), **RECORD_KWARGS)]
    summary = build_mask_manifest(rows)["summary"]
    assert "quarantined_by_reason" not in summary
    assert summary["coverage_fraction"] == 1.0


def test_empty_manifest_has_no_coverage_fraction():
    summary = build_mask_manifest([])["summary"]
    assert summary["total"] == 0
    assert "coverage_fraction" not in summary
