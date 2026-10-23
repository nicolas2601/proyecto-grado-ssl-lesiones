import numpy as np

from dermssl.pipeline_segment_circle import (
    DermLIPCirclePipeline,
    PipelineMaskResult,
    select_bio_admissible_component,
)


def test_select_bio_admissible_component_prioritizes_central_organic_lesion():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [190, 150, 130]  # Skin background

    # Central lesion (RGB [110, 60, 40])
    img[35:65, 35:65] = [110, 60, 40]

    # Peripheral blue calibration sticker (RGB [15, 170, 240])
    img[0:30, 70:100] = [15, 170, 240]

    pred_binary = np.zeros((100, 100), dtype=bool)
    pred_binary[35:65, 35:65] = True
    pred_binary[0:30, 70:100] = True

    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[35:65, 35:65] = 0.92
    prob_map[0:30, 70:100] = 0.95

    # Naive without center prior chooses the sticker if slightly larger
    pred_binary[0:32, 68:100] = True
    naive = select_bio_admissible_component(pred_binary, prob_map, img, center_prior=False)
    assert naive[10, 80] is np.True_

    # Bio-admissible component selector chooses the central lesion
    filtered = select_bio_admissible_component(pred_binary, prob_map, img, center_prior=True)
    assert filtered[50, 50] is np.True_
    assert filtered[10, 80] is np.False_


def test_circle_shift_preserves_lesion_and_vignette_exactly():
    pipeline = DermLIPCirclePipeline()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Pale skin: RGB [210, 170, 150]
    img[:, :] = [210, 170, 150]

    # Lesion in center: RGB [90, 50, 40]
    img[30:70, 30:70] = [90, 50, 40]

    # Black vignette corner: [0:20, 0:20] = [0, 0, 0]
    img[0:20, 0:20] = [0, 0, 0]

    lesion_mask = np.zeros((100, 100), dtype=bool)
    lesion_mask[30:70, 30:70] = True

    valid_fov_mask = np.ones((100, 100), dtype=bool)
    valid_fov_mask[0:20, 0:20] = False  # Corner vignette

    healthy_skin_mask = valid_fov_mask & (~lesion_mask)

    mask_result = PipelineMaskResult(
        lesion_mask=lesion_mask,
        valid_fov_mask=valid_fov_mask,
        healthy_skin_mask=healthy_skin_mask,
        lesion_coverage=float(lesion_mask.mean()),
        healthy_skin_coverage=float(healthy_skin_mask.mean()),
        qa_passed=True,
    )

    # Shift skin darker (+2 FST buckets towards deeper phototype)
    transformed, was_shifted = pipeline.apply_circle_shift(img, mask_result, shift_buckets=2)
    assert was_shifted is True

    # 1. Healthy skin must be noticeably darker (L* shifted down)
    orig_skin_mean = img[healthy_skin_mask].mean()
    trans_skin_mean = transformed[healthy_skin_mask].mean()
    assert trans_skin_mean < orig_skin_mean

    # 2. Lesion pixels must be preserved identically (0 delta)
    np.testing.assert_array_equal(transformed[lesion_mask], img[lesion_mask])

    # 3. Vignette pixels must be preserved identically (0 delta)
    vignette_mask = ~valid_fov_mask
    np.testing.assert_array_equal(transformed[vignette_mask], img[vignette_mask])


def test_circle_shift_falls_back_to_identity_when_qa_fails():
    pipeline = DermLIPCirclePipeline()
    img = np.ones((100, 100, 3), dtype=np.uint8) * 150

    mask_result = PipelineMaskResult(
        lesion_mask=np.zeros((100, 100), dtype=bool),
        valid_fov_mask=np.ones((100, 100), dtype=bool),
        healthy_skin_mask=np.ones((100, 100), dtype=bool),
        lesion_coverage=0.0,
        healthy_skin_coverage=1.0,
        qa_passed=False,
        fallback_reason="lesion_too_small:0.0000",
    )

    transformed, was_shifted = pipeline.apply_circle_shift(img, mask_result, shift_buckets=2)
    assert was_shifted is False
    np.testing.assert_array_equal(transformed, img)
