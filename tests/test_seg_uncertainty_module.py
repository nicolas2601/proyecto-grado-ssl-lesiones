"""Pure tests for the uncertainty module and the central-component filter."""

import numpy as np
import pytest

import seg_paths  # noqa: F401
import cache_io
import uncertainty_module as um


def _blob(size=64, center=(32, 32), radius=10, peak=1.0):
    yy, xx = np.mgrid[:size, :size]
    distance = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    return np.clip(peak * (1 - distance / (radius * 1.6)), 0, 1).astype(np.float32)


def test_normalize_saliency_maps_to_unit_range():
    out = um.normalize_saliency(np.array([[2.0, 4.0], [6.0, 10.0]]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize_constant_map_is_all_zero():
    assert um.normalize_saliency(np.full((4, 4), 3.0)).max() == 0.0


def test_orient_flips_when_border_is_brighter_than_center():
    saliency = 1.0 - _blob()
    oriented = um.orient_saliency(saliency)
    assert oriented[32, 32] > oriented[0, 0]


def test_orient_keeps_map_when_center_is_brighter():
    saliency = _blob()
    assert np.array_equal(um.orient_saliency(saliency), saliency)


def test_central_region_prefers_the_region_near_the_center():
    binary = np.zeros((64, 64), dtype=bool)
    binary[28:38, 28:38] = True  # central, 100 px
    binary[2:22, 2:22] = True  # corner, larger, 400 px
    region = um.select_central_region(binary, min_area=20, max_center_distance=0.9)
    assert region is not None
    assert region[32, 32] and not region[10, 10]


def test_central_region_drops_components_smaller_than_min_area():
    binary = np.zeros((64, 64), dtype=bool)
    binary[31:33, 31:33] = True  # 4 px, central but tiny
    binary[4:20, 4:20] = True
    region = um.select_central_region(binary, min_area=50, max_center_distance=1.0)
    assert region is not None and region[10, 10] and not region[32, 32]


def test_central_region_rejects_when_too_far_from_center():
    binary = np.zeros((64, 64), dtype=bool)
    binary[0:8, 0:8] = True
    assert um.select_central_region(binary, min_area=10, max_center_distance=0.5) is None


def test_central_region_of_empty_map_is_none():
    assert um.select_central_region(np.zeros((8, 8), bool), 1, 1.0) is None


def test_pseudo_mask_has_three_bands_and_valid_flag():
    result = um.saliency_to_pseudo_mask(_blob(peak=1.0))
    mask = result.mask
    assert result.valid is True
    assert mask.dtype == np.uint8
    assert mask[32, 32] == cache_io.FOREGROUND
    assert mask[0, 0] == cache_io.BACKGROUND
    assert set(np.unique(mask)) == {cache_io.BACKGROUND, cache_io.FOREGROUND, cache_io.IGNORE}


def test_pseudo_mask_thresholds_follow_the_paper_bands():
    saliency = np.full((8, 8), 0.5, dtype=np.float32)
    saliency[0, 0], saliency[7, 7] = 0.0, 1.0
    saliency[3, 3] = 0.34  # below 0.35: background
    result = um.saliency_to_pseudo_mask(
        saliency, normalize=False, min_area_fraction=0.0, max_center_distance=10.0
    )
    assert result.mask[3, 3] == cache_io.BACKGROUND
    assert result.mask[4, 4] == cache_io.IGNORE  # 0.5 is in the uncertain band
    assert result.mask[7, 7] == cache_io.FOREGROUND


def test_non_selected_high_regions_are_ignored_not_background():
    saliency = _blob()
    saliency[2:14, 2:14] = 1.0  # a second high region near a corner
    mask = um.saliency_to_pseudo_mask(saliency).mask
    assert mask[8, 8] == cache_io.IGNORE


def test_sample_without_foreground_is_excluded():
    result = um.saliency_to_pseudo_mask(np.full((32, 32), 0.2, dtype=np.float32), normalize=False)
    assert result.valid is False
    assert np.all(result.mask == cache_io.IGNORE)


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        um.saliency_to_pseudo_mask(_blob(), low=0.7, high=0.3)


def test_batch_conversion_writes_masks_and_reports_stats():
    saliency = np.stack([_blob(), np.full((64, 64), 0.2, np.float32), 1.0 - _blob()])
    out = np.zeros(saliency.shape, dtype=np.uint8)
    stats = um.saliency_array_to_pseudo_masks(saliency, out, orient=True, chunk_size=2)
    assert stats["images"] == 3
    assert stats["excluded"] == 1
    assert out[0, 32, 32] == cache_io.FOREGROUND
    assert out[2, 32, 32] == cache_io.FOREGROUND  # oriented before thresholding
    assert np.all(out[1] == cache_io.IGNORE)
    assert 0.0 < stats["mean_foreground_fraction"] < 1.0


def test_batch_conversion_can_keep_probabilities_unnormalized():
    probabilities = np.full((1, 32, 32), 0.5, dtype=np.float32)
    out = np.zeros(probabilities.shape, dtype=np.uint8)
    stats = um.saliency_array_to_pseudo_masks(probabilities, out, orient=False, normalize=False)
    assert stats["excluded"] == 1  # a flat 0.5 map has no confident lesion


def _disk(size=64, radius=26):
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    return (yy - center) ** 2 + (xx - center) ** 2 <= radius**2


def test_candidates_outside_the_field_are_never_selected():
    saliency = _blob()
    saliency[0:16, 0:16] = 1.0  # a bright vignette corner outside the field
    field = _disk()
    mask = um.saliency_to_pseudo_mask(saliency, field=field, normalize=False).mask
    assert mask[32, 32] == cache_io.FOREGROUND
    assert mask[4, 4] == cache_io.BACKGROUND


def test_every_pixel_outside_the_field_is_background():
    field = _disk()
    mask = um.saliency_to_pseudo_mask(_blob(), field=field, normalize=False).mask
    assert np.all(mask[~field] == cache_io.BACKGROUND)


def test_a_region_covering_the_whole_field_invalidates_the_sample():
    field = _disk()
    saliency = np.where(field, 1.0, 0.0).astype(np.float32)
    result = um.saliency_to_pseudo_mask(saliency, field=field, normalize=False)
    assert result.valid is False
    assert result.reason == "area"
    assert np.all(result.mask == cache_io.IGNORE)


def test_the_area_cap_also_applies_without_a_field():
    saliency = np.ones((32, 32), dtype=np.float32)
    result = um.saliency_to_pseudo_mask(saliency, normalize=False)
    assert result.valid is False and result.reason == "area"


def test_a_generous_area_cap_keeps_the_sample():
    field = _disk()
    saliency = np.where(field, 1.0, 0.0).astype(np.float32)
    result = um.saliency_to_pseudo_mask(
        saliency, field=field, normalize=False, max_area_fraction=1.0
    )
    assert result.valid is True


def test_orientation_uses_the_field_edge_not_the_black_image_border():
    field = _disk()
    saliency = np.where(field, 0.8, 0.0).astype(np.float32)
    saliency[_disk(radius=8)] = 0.2  # dark lesion inside a bright field
    without_field = um.orient_saliency(saliency)
    with_field = um.orient_saliency(saliency, field=field)
    assert without_field[32, 32] < without_field[32, 20]  # the vignette hides the contrast
    assert with_field[32, 32] > with_field[32, 20]


def test_orientation_without_a_field_keeps_the_previous_behaviour():
    saliency = 1.0 - _blob()
    assert np.array_equal(um.orient_saliency(saliency, field=None), um.orient_saliency(saliency))


def test_a_field_with_a_different_shape_raises():
    with pytest.raises(ValueError):
        um.saliency_to_pseudo_mask(_blob(), field=np.ones((8, 8), bool))


def test_batch_conversion_counts_regions_excluded_by_area():
    field = _disk()
    saturated = np.where(field, 1.0, 0.0).astype(np.float32)
    saliency = np.stack([_blob(), saturated])
    fields = np.stack([field, field]).astype(np.uint8)
    out = np.zeros(saliency.shape, dtype=np.uint8)
    stats = um.saliency_array_to_pseudo_masks(
        saliency, out, orient=False, normalize=False, fields=fields
    )
    assert stats["excluded"] == 1
    assert stats["excluded_by_area"] == 1
    assert np.all(out[1] == cache_io.IGNORE)
    assert out[0, 0, 0] == cache_io.BACKGROUND


def test_the_minimum_area_is_measured_against_the_field_not_the_frame():
    """Both area limits must share one reference, or they contradict each other."""
    field = _disk(radius=10)
    saliency = np.zeros((64, 64), dtype=np.float32)
    saliency[30:34, 30:34] = 1.0  # 16 pixels, centred, small for the frame
    options = {"normalize": False, "min_area_fraction": 0.04, "max_area_fraction": 1.0}

    assert um.saliency_to_pseudo_mask(saliency, field=field, **options).valid is True
    assert um.saliency_to_pseudo_mask(saliency, **options).valid is False


def test_an_empty_field_is_reported_as_missing_not_as_an_area_problem():
    result = um.saliency_to_pseudo_mask(
        _blob(), field=np.zeros((64, 64), dtype=bool), normalize=False
    )
    assert result.valid is False
    assert result.reason == um.REASON_NO_FIELD
    assert np.all(result.mask == cache_io.IGNORE)


def test_batch_conversion_counts_samples_whose_field_is_empty():
    saliency = np.stack([_blob(), _blob()])
    fields = np.stack([_disk(), np.zeros((64, 64), dtype=bool)]).astype(np.uint8)
    out = np.zeros(saliency.shape, dtype=np.uint8)
    stats = um.saliency_array_to_pseudo_masks(
        saliency, out, orient=False, normalize=False, fields=fields
    )
    assert stats["excluded"] == 1
    assert stats["excluded_no_field"] == 1
    assert stats["excluded_by_area"] == 0
    assert stats["excluded_no_region"] == 0


def test_an_unknown_exclusion_reason_is_counted_instead_of_crashing(monkeypatch):
    def odd_result(saliency, **kwargs):
        mask = np.full(saliency.shape, cache_io.IGNORE, dtype=np.uint8)
        return um.PseudoMaskResult(mask=mask, valid=False, reason="mystery")

    monkeypatch.setattr(um, "saliency_to_pseudo_mask", odd_result)
    out = np.zeros((1, 8, 8), dtype=np.uint8)
    stats = um.saliency_array_to_pseudo_masks(
        np.zeros((1, 8, 8), dtype=np.float32), out, orient=False, normalize=False
    )
    assert stats["excluded"] == 1
    assert stats["excluded_by_area"] == 0


# ------------------------------------------------- artefact suppression
def _skin(size=64):
    """Flat skin patch: warm, unsaturated in blue and far from white."""
    return np.full((size, size, 3), (200, 150, 140), dtype=np.uint8)


def test_violet_ink_is_detected_as_an_artifact():
    rgb = _skin()
    rgb[10:20, 10:20] = (90, 60, 160)
    artifact = um.artifact_mask(rgb)
    assert artifact[15, 15] and not artifact[45, 45]


def test_a_specular_highlight_is_detected_as_an_artifact():
    rgb = _skin()
    rgb[30:34, 30:34] = (250, 250, 250)
    artifact = um.artifact_mask(rgb)
    assert artifact[32, 32] and not artifact[5, 5]


def test_the_artifact_mask_grows_two_pixels_around_the_evidence():
    rgb = _skin()
    rgb[32, 32] = (250, 250, 250)
    artifact = um.artifact_mask(rgb)
    assert artifact[32, 34] and not artifact[32, 36]


def test_plain_skin_carries_no_artifact():
    assert not um.artifact_mask(_skin()).any()


def test_the_artifact_mask_needs_three_colour_channels():
    with pytest.raises(ValueError):
        um.artifact_mask(np.zeros((8, 8), dtype=np.uint8))


def test_an_ink_mark_beside_a_lesion_is_ignored_instead_of_selected():
    saliency = _blob(center=(40, 40), radius=8)
    saliency[20:32, 20:32] = 1.0  # the ink mark wins the class-agnostic saliency
    artifact = np.zeros((64, 64), dtype=bool)
    artifact[20:32, 20:32] = True

    unprotected = um.saliency_to_pseudo_mask(saliency, normalize=False).mask
    assert unprotected[25, 25] == cache_io.FOREGROUND

    result = um.saliency_to_pseudo_mask(saliency, artifact=artifact, normalize=False)
    assert result.valid is True
    assert result.mask[40, 40] == cache_io.FOREGROUND
    assert result.mask[25, 25] == cache_io.IGNORE


def test_a_specular_spot_inside_the_lesion_is_ignored_not_foreground():
    artifact = np.zeros((64, 64), dtype=bool)
    artifact[31:34, 31:34] = True
    result = um.saliency_to_pseudo_mask(_blob(), artifact=artifact, normalize=False)
    assert result.valid is True
    assert result.mask[32, 32] == cache_io.IGNORE
    assert result.mask[32, 36] == cache_io.FOREGROUND


def test_artifact_pixels_outside_the_field_are_ignored_not_background():
    field = _disk()
    artifact = np.zeros((64, 64), dtype=bool)
    artifact[0:4, 0:4] = True
    mask = um.saliency_to_pseudo_mask(
        _blob(), field=field, artifact=artifact, normalize=False
    ).mask
    assert mask[2, 2] == cache_io.IGNORE
    assert mask[2, 20] == cache_io.BACKGROUND


def test_an_empty_artifact_map_changes_nothing():
    saliency = _blob()
    plain = um.saliency_to_pseudo_mask(saliency, normalize=False).mask
    empty = um.saliency_to_pseudo_mask(
        saliency, artifact=np.zeros((64, 64), dtype=bool), normalize=False
    ).mask
    assert np.array_equal(plain, empty)


def test_an_artifact_map_with_a_different_shape_raises():
    with pytest.raises(ValueError):
        um.saliency_to_pseudo_mask(_blob(), artifact=np.ones((8, 8), bool))


def test_batch_conversion_reads_the_artifacts_from_the_image_cache():
    saliency = _blob(center=(40, 40), radius=8)
    saliency[20:32, 20:32] = 1.0
    images = np.full((1, 64, 64, 3), (200, 150, 140), dtype=np.uint8)
    images[0, 20:32, 20:32] = (90, 60, 160)
    out = np.zeros((1, 64, 64), dtype=np.uint8)
    stats = um.saliency_array_to_pseudo_masks(
        saliency[None], out, orient=False, normalize=False, images=images
    )
    assert out[0, 40, 40] == cache_io.FOREGROUND
    assert out[0, 25, 25] == cache_io.IGNORE
    assert stats["mean_artifact_fraction"] > 0.0


def test_batch_conversion_without_images_reports_no_artifact_fraction():
    out = np.zeros((1, 64, 64), dtype=np.uint8)
    stats = um.saliency_array_to_pseudo_masks(
        _blob()[None], out, orient=False, normalize=False
    )
    assert stats["mean_artifact_fraction"] == 0.0
