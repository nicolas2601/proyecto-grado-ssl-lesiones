"""Pure tests for the student data helpers."""

import numpy as np
import seg_paths  # noqa: F401
import student_data


def test_indices_with_confident_pixels_drops_fully_ignored_samples():
    masks = np.full((4, 6, 6), 255, dtype=np.uint8)
    masks[1, 2, 2] = 1
    masks[3, :, :3] = 0
    kept = student_data.indices_with_confident_pixels(masks, np.arange(4), chunk_size=2)
    assert kept.tolist() == [1, 3]


def test_indices_with_confident_pixels_respects_the_candidate_subset():
    masks = np.zeros((5, 4, 4), dtype=np.uint8)
    kept = student_data.indices_with_confident_pixels(masks, np.array([0, 2, 4]), chunk_size=8)
    assert kept.tolist() == [0, 2, 4]


def test_checkpoint_selection_choices_match_the_selector_modes():
    import selection

    assert set(student_data.SELECTION_CHOICES) == set(selection.MODES)


def test_require_training_images_explains_a_collapsed_pseudo_label_set():
    import pytest

    with pytest.raises(ValueError, match="excluyo|threshold|umbral"):
        student_data.require_training_images(np.array([], dtype=np.int64))
    assert student_data.require_training_images(np.array([3, 4])).tolist() == [3, 4]
