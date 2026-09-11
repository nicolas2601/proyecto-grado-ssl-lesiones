"""FST estimation: sRGB->LAB conversion, ITA/bucket mapping, and the
mask-guided estimator. Pure numpy in production; scikit-image is a test
oracle only (D2)."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from dermssl import fitzpatrick as fp

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "dermssl" / "fitzpatrick.py"


def random_srgb(rng, shape=(16, 16, 3)):
    return rng.uniform(0.0, 1.0, size=shape)


def test_srgb_to_lab_matches_skimage_reference():
    skimage_color = pytest.importorskip("skimage.color")
    rng = np.random.default_rng(0)
    img = random_srgb(rng)

    lab = fp.srgb_to_lab(img)
    expected = skimage_color.rgb2lab(img)

    assert lab.shape == expected.shape
    np.testing.assert_allclose(lab, expected, atol=1e-4)


def test_srgb_to_lab_matches_skimage_on_boundary_colors():
    """Black/white/near-black exercise the piecewise gamma and Lab-f()
    thresholds (different branch than the mid-range random swatch above)."""
    skimage_color = pytest.importorskip("skimage.color")
    swatches = np.array(
        [
            [0.0, 0.0, 0.0],  # black: exercises the linear gamma branch
            [1.0, 1.0, 1.0],  # white: near the top of both piecewise fns
            [0.01, 0.01, 0.01],  # just below the sRGB gamma cutover (0.04045)
            [1.0, 0.0, 0.0],  # saturated red
        ]
    ).reshape(1, 4, 3)

    lab = fp.srgb_to_lab(swatches)
    expected = skimage_color.rgb2lab(swatches)

    np.testing.assert_allclose(lab, expected, atol=1e-4)
    # Black must land at the Lab origin - a concrete, non-trivial value.
    np.testing.assert_allclose(lab[0, 0], [0.0, 0.0, 0.0], atol=1e-6)


def test_ita_degrees_matches_the_groh_formula():
    """ITA = arctan((L*-50)/b*) * 180/pi (plain arctan of the ratio - the
    literature formula, NOT arctan2/atan2 of the two components, which
    would answer a different quadrant-aware question)."""
    L = np.array([70.0, 30.0])
    b = np.array([20.0, -10.0])

    ita = fp.ita_degrees(L, b)

    expected = np.degrees(np.arctan((L - 50.0) / b))
    np.testing.assert_allclose(ita, expected)


def test_ita_degrees_stays_within_arctan_bounds_for_negative_b():
    """Negative b* (a common reading on blue-ish/shadowed skin) must not
    push ITA past +/-90: arctan is bounded by construction (error table in
    design.md), unlike arctan2 which would wrap into (90, 180]."""
    ita = fp.ita_degrees(L=np.array([30.0]), b=np.array([-10.0]))
    np.testing.assert_allclose(ita, [63.43494882], atol=1e-4)


@pytest.mark.parametrize(
    "ita, expected_bucket",
    [
        (60.0, 1),  # well above the lightest threshold
        (55.0001, 1),  # just above 55
        (55.0, 2),  # exactly on the 55 boundary falls to the next bucket
        (50.0, 2),
        (41.0, 3),  # exactly on the 41 boundary
        (35.0, 3),
        (28.0, 4),  # exactly on the 28 boundary
        (24.0, 4),
        (19.0, 5),  # exactly on the 19 boundary
        (15.0, 5),
        (10.0, 6),  # exactly on the 10 boundary
        (-30.0, 6),  # well below the darkest threshold
    ],
)
def test_fst_bucket_at_groh_thresholds_and_boundaries(ita, expected_bucket):
    assert fp.fst_bucket(ita) == expected_bucket


def disk_srgb_image(size=64, radius=15, skin_rgb=(0.85, 0.65, 0.60), lesion_rgb=(0.30, 0.20, 0.10)):
    """A well-separated synthetic dermoscopy image: dark lesion disk on a
    lighter skin background, both far from the near-black border/vignette
    threshold - the "clean image" scenario from spec.md.

    Grayscale means (0.2, 0.7) are deliberately chosen to fall on the
    low/favorable side of ``lesion_guided.otsu_threshold``'s 1/256-wide
    quantization bins - flat, noise-free synthetic clusters can otherwise
    land on a bin's tied argmax edge and produce a threshold that excludes
    the very cluster it is supposed to select (verified empirically against
    the pre-existing Otsu implementation, not something this PR modifies)."""
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.empty((size, size, 3))
    img[:] = skin_rgb
    img[dist <= radius] = lesion_rgb
    return img, dist


def _independent_expected_bucket(img, true_skin):
    """Reference bucket computed from KNOWN geometry (not the production
    mask), via the skimage LAB oracle - independent of estimate_fst's own
    masking so the assertion is not circular."""
    skimage_color = pytest.importorskip("skimage.color")
    lab = skimage_color.rgb2lab(img)
    L_med = np.median(lab[..., 0][true_skin])
    b_med = np.median(lab[..., 2][true_skin])
    ita = float(fp.ita_degrees(np.array(L_med), np.array(b_med)))
    return fp.fst_bucket(ita)


def test_estimate_fst_returns_ok_bucket_for_a_clean_skin_lesion_image():
    pytest.importorskip("skimage.color")
    img, dist = disk_srgb_image()

    result = fp.estimate_fst(img)

    assert result.status == "ok"
    expected_bucket = _independent_expected_bucket(img, true_skin=dist > 15)
    assert result.bucket == expected_bucket
    assert result.skin_fraction == pytest.approx((dist > 15).mean(), abs=0.01)


def test_estimate_fst_falls_back_to_otsu_when_the_given_mask_is_degenerate():
    """A degenerate primary mask (e.g. a failed U-Net call, all-lesion) must
    fall back to the Otsu mask and still produce a real estimate (D3)."""
    pytest.importorskip("skimage.color")
    img, dist = disk_srgb_image()
    degenerate_given_mask = np.ones(img.shape[:2], dtype=bool)  # 100% "lesion"

    result = fp.estimate_fst(img, mask=degenerate_given_mask)

    assert result.status == "ok"
    expected_bucket = _independent_expected_bucket(img, true_skin=dist > 15)
    assert result.bucket == expected_bucket


def test_estimate_fst_reports_primary_mask_provenance_and_confidence():
    img, dist = disk_srgb_image()
    mask = dist <= 15

    result = fp.estimate_fst(img, mask=mask, mask_confidence=0.92)

    assert result.status == "ok"
    assert result.fallback_used is False
    assert result.artifact_detected is False
    assert result.confidence == pytest.approx(0.92)


def test_estimate_fst_low_mask_confidence_uses_fallback_and_reduces_confidence():
    img, dist = disk_srgb_image()
    mask = dist <= 15

    result = fp.estimate_fst(img, mask=mask, mask_confidence=0.10)

    assert result.status == "ok"
    assert result.bucket == _independent_expected_bucket(img, true_skin=dist > 15)
    assert result.fallback_used is True
    assert 0.0 < result.confidence < 0.92


def test_estimate_fst_flags_dark_linear_artifact_and_reduces_confidence():
    img, dist = disk_srgb_image()
    mask = dist <= 15
    clean = fp.estimate_fst(img, mask=mask, mask_confidence=0.95)
    with_hair = img.copy()
    with_hair[6:8, 5:-5] = 0.0

    result = fp.estimate_fst(with_hair, mask=mask, mask_confidence=0.95)

    assert result.status == "ok"
    assert result.bucket is not None
    assert result.artifact_detected is True
    assert result.confidence < clean.confidence


def test_estimate_fst_no_skin_when_candidate_skin_is_almost_all_border_vignette():
    """skin_fraction < 0.05 after border/vignette exclusion -> no_skin,
    bucket None (spec: explicit bucket-or-unavailable, never a silent skip)."""
    img = np.full((20, 20, 3), 0.01)  # near-black border/vignette everywhere
    mask = np.zeros((20, 20), dtype=bool)
    mask[:10, :] = True  # top half: lesion (50% area, not degenerate)
    img[:10, :] = (0.3, 0.2, 0.15)  # lesion color (irrelevant, excluded)
    img[10, 0:5] = (0.8, 0.6, 0.5)  # 5/400 = 1.25% real skin pixels

    result = fp.estimate_fst(img, mask=mask)

    assert result.status == "no_skin"
    assert result.bucket is None
    assert result.skin_fraction == pytest.approx(5 / 400)


def test_estimate_fst_degenerate_mask_when_otsu_fallback_also_fails():
    """Given mask degenerate AND the Otsu fallback degenerate (a uniform
    image has no bimodal split) -> degenerate_mask, bucket None."""
    img = np.full((20, 20, 3), 0.5)  # uniform: Otsu cannot find a split
    degenerate_given_mask = np.ones((20, 20), dtype=bool)

    result = fp.estimate_fst(img, mask=degenerate_given_mask)

    assert result.status == "degenerate_mask"
    assert result.bucket is None
    assert np.isnan(result.ita)
    assert result.fallback_used is True
    assert result.confidence == 0.0


def test_estimate_fst_degenerate_mask_reports_skin_fraction_not_lesion_fraction():
    """``skin_fraction`` must always mean the fraction of SKIN pixels
    (mask=True means lesion, so skin = ~mask & ~border), the same
    convention every other branch uses - never the raw lesion fraction.

    The given mask (100% "lesion") is degenerate, so it gets replaced by
    the Otsu fallback on this uniform image; that fallback also ends up
    degenerate (0% "lesion" - a flat image has no bimodal split), which is
    what actually reaches the ``degenerate_mask`` return. The buggy
    ``float(mask.mean())`` reports that 0% AS IF it were the skin
    fraction, when the true skin fraction (``~mask & ~border``, no
    border here) is 100%."""
    img = np.full((20, 20, 3), 0.5)  # uniform: Otsu cannot find a split
    degenerate_given_mask = np.ones((20, 20), dtype=bool)  # 100% "lesion"

    result = fp.estimate_fst(img, mask=degenerate_given_mask)

    assert result.status == "degenerate_mask"
    assert result.skin_fraction == pytest.approx(1.0)


@pytest.mark.parametrize("bad_ita", [float("nan"), 91.0, -91.0])
def test_estimate_fst_returns_ita_out_of_range_status_instead_of_crashing(monkeypatch, bad_ita):
    """An achromatic skin median at L*=50 makes ``ita_degrees``'s
    ``(L*-50)/b*`` a literal 0/0, which is NaN - and NaN comparisons are
    always False, so a bare ``assert -90 <= ita <= 90`` raises
    AssertionError instead of the documented explicit status. This must
    degrade gracefully (status="ita_out_of_range", bucket=None), the same
    "never crash a run on an atypical image" philosophy as
    ``pretrain/lesion_guided.py`` - not crash the whole batch.

    ``ita_degrees`` is monkeypatched because constructing skin pixels
    whose LAB median lands on an EXACT 0/0 (not merely near it) via real
    sRGB->LAB conversion is floating-point-fragile; patching isolates the
    behavior this fix actually changes: how ``estimate_fst`` reacts to an
    out-of-range/NaN ITA, regardless of how it got there."""
    pytest.importorskip("skimage.color")
    img, dist = disk_srgb_image()
    mask = dist <= 15

    monkeypatch.setattr(fp, "ita_degrees", lambda L, b: bad_ita)

    result = fp.estimate_fst(img, mask=mask)

    assert result.status == "ita_out_of_range"
    assert result.bucket is None
    assert np.isnan(result.ita) if np.isnan(bad_ita) else result.ita == bad_ita


def test_estimate_fst_coerces_a_non_bool_mask_to_the_same_result_as_bool():
    """A future U-Net (Phase 3) may emit an int/uint8 0/1 mask instead of
    native bool. ``~mask`` on an integer array is bitwise NOT on the
    integers (e.g. ~np.uint8(1) == 254), not logical negation - so
    ``lab[skin]`` silently does wrong fancy-indexing instead of boolean
    masking, with no error raised, just a wrong FST bucket. An int-dtype
    mask must produce the same estimate as the equivalent bool mask."""
    pytest.importorskip("skimage.color")
    img, dist = disk_srgb_image()
    mask_bool = dist <= 15
    mask_uint8 = mask_bool.astype(np.uint8)

    result_bool = fp.estimate_fst(img, mask=mask_bool)
    result_uint8 = fp.estimate_fst(img, mask=mask_uint8)

    assert result_uint8.status == result_bool.status
    assert result_uint8.bucket == result_bool.bucket
    assert result_uint8.ita == pytest.approx(result_bool.ita)
    assert result_uint8.skin_fraction == pytest.approx(result_bool.skin_fraction)


def test_module_source_never_imports_skimage():
    """Static check (D2): a module-level ``import skimage`` would break
    ``test_notebook_build.py``'s inlined ``exec`` cell and the Kaggle
    notebook, since ``__init__.py::MODULE_ORDER`` executes this file."""
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("skimage") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("skimage")


def test_importing_fitzpatrick_never_imports_skimage():
    """Dynamic check mirroring the pure-package torch guard: importing the
    module must not pull scikit-image into ``sys.modules`` as a side effect."""
    for name in list(sys.modules):
        if name == "skimage" or name.startswith("skimage."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name == "dermssl.fitzpatrick":
            del sys.modules[name]

    import dermssl.fitzpatrick  # noqa: F401

    assert "skimage" not in sys.modules
