"""S-diag: training-free DINOv2 segmentation, pure half (docs/27, S-diag method).

Every test plants a known structure in a synthetic token grid and checks that
the registered rule recovers it. No torch, no weights, no reference masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import s_diag as sd

GRID = 16
DIM = 32


def _blob(grid=GRID, top=5, left=5, side=6):
    mask = np.zeros((grid, grid), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def _tokens(mask, seed=0, noise=0.05):
    """Background tokens near one direction, blob tokens near another."""
    rng = np.random.default_rng(seed)
    background, lesion = rng.normal(size=DIM), rng.normal(size=DIM)
    tokens = np.where(mask.reshape(-1, 1), lesion, background)
    return tokens + noise * rng.normal(size=tokens.shape)


def test_central_block_is_the_central_quarter_of_the_grid():
    rows, cols = sd.central_block(16)
    assert (rows.start, rows.stop, cols.start, cols.stop) == (6, 10, 6, 10)
    rows, cols = sd.central_block(32)
    assert (rows.start, rows.stop) == (12, 20)


def test_pca_map_has_the_grid_shape_and_a_positive_centre():
    component = sd.pca_first_component_map(_tokens(_blob()), GRID)
    assert component.shape == (GRID, GRID)
    rows, cols = sd.central_block(GRID)
    assert component[rows, cols].mean() > component.mean()


def test_pca_sign_is_fixed_by_the_centre_not_by_the_svd():
    tokens = _tokens(_blob())
    straight = sd.pca_first_component_map(tokens, GRID)
    flipped = sd.pca_first_component_map(-tokens, GRID)
    np.testing.assert_allclose(straight, flipped, atol=1e-8)


def test_pca_sign_flips_when_the_centre_is_the_minority_side():
    ring = ~_blob()
    component = sd.pca_first_component_map(_tokens(ring), GRID)
    rows, cols = sd.central_block(GRID)
    assert component[rows, cols].mean() > 0


def test_tokens_are_l2_normalised_so_their_scale_does_not_matter():
    tokens = _tokens(_blob())
    scaled = tokens * np.linspace(0.5, 3.0, len(tokens))[:, None]
    np.testing.assert_allclose(
        sd.pca_first_component_map(tokens, GRID),
        sd.pca_first_component_map(scaled, GRID),
        atol=1e-8,
    )


def test_otsu_separates_a_bimodal_map():
    # skimage returns a bin centre, so noise inside the top bin of the lower
    # mode can cross it; two exact levels make the expected split unambiguous.
    values = np.where(_blob(), 1.0, -1.0)
    np.testing.assert_array_equal(sd.otsu_binary(values), _blob())


def test_otsu_of_a_constant_map_is_empty():
    assert not sd.otsu_binary(np.zeros((GRID, GRID))).any()


def test_upsample_keeps_shape_dtype_and_values():
    up = sd.upsample_nearest(_blob(), (224, 224))
    assert up.shape == (224, 224) and up.dtype == bool
    assert up.sum() == _blob().sum() * 14 * 14
    np.testing.assert_array_equal(up[::14, ::14], _blob())


def test_upsample_to_a_non_multiple_shape_samples_patch_centres():
    grid = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(
        sd.upsample_nearest(grid, (3, 3)),
        [[True, False, False], [False, True, True], [False, True, True]],
    )


def test_central_component_keeps_only_the_component_containing_the_centre():
    binary = _blob(grid=32, top=12, left=12, side=8)
    binary[0:3, 0:3] = True
    np.testing.assert_array_equal(sd.central_component(binary), _blob(grid=32, top=12, left=12, side=8))


def test_central_component_falls_back_to_the_nearest_component():
    binary = np.zeros((32, 32), dtype=bool)
    near = np.zeros_like(binary)
    near[16:19, 20:24] = True
    far = np.zeros_like(binary)
    far[0:10, 0:10] = True
    binary = near | far
    assert not binary[16, 16]
    np.testing.assert_array_equal(sd.central_component(binary), near)


def test_central_component_of_an_empty_map_is_empty():
    assert not sd.central_component(np.zeros((8, 8), dtype=bool)).any()


def test_diagonal_neighbours_belong_to_one_component():
    binary = np.zeros((8, 8), dtype=bool)
    binary[4, 4] = binary[5, 5] = binary[6, 6] = True
    assert sd.central_component(binary).sum() == 3


def test_kmeans_recovers_the_planted_blob_as_the_central_cluster():
    labels = sd.kmeans_map(_tokens(_blob()), GRID, seed=0)
    assert labels.shape == (GRID, GRID)
    lesion = labels == labels[GRID // 2, GRID // 2]
    np.testing.assert_array_equal(lesion, _blob())


def test_kmeans_is_deterministic_for_a_fixed_seed():
    tokens = _tokens(_blob(), noise=0.4)
    np.testing.assert_array_equal(sd.kmeans_map(tokens, GRID, 0), sd.kmeans_map(tokens, GRID, 0))


@pytest.mark.parametrize("method", ["pca", "kmeans"])
def test_segment_recovers_the_blob_at_the_evaluation_resolution(method):
    blob = _blob()
    blob[0, 0] = True  # a stray patch that the central-component rule must drop
    mask = sd.segment(method, _tokens(blob), GRID, (224, 224), seed=0)
    np.testing.assert_array_equal(mask, sd.upsample_nearest(_blob(), (224, 224)))


def test_segment_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown"):
        sd.segment("spectral", _tokens(_blob()), GRID, (224, 224), seed=0)


def test_tokens_must_fill_the_grid():
    with pytest.raises(ValueError, match="grid"):
        sd.pca_first_component_map(np.ones((10, DIM)), GRID)


def test_importing_s_diag_does_not_import_torch():
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    code = "import sys; import dermssl.s_diag; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.strip() == "False"
