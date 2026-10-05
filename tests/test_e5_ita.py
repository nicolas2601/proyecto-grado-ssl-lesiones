"""E5 global ITA estimators, the canonical decode and the per-dataset L* terciles."""

import numpy as np
import pytest
from PIL import Image

from dermssl import e5_ita as ita
from dermssl import fitzpatrick as fz

SKIN = {1: (0.93, 0.80, 0.70), 3: (0.70, 0.52, 0.40), 4: (0.62, 0.45, 0.33), 5: (0.55, 0.38, 0.30),
        6: (0.25, 0.15, 0.10)}


def _lesion_image(skin, size=(90, 120), lesion=None, radius=20) -> np.ndarray:
    """Skin with a darker disc; seeded noise keeps Otsu's strict ``<`` away from a flat bin edge."""
    h, w = size
    lesion = tuple(0.4 * c for c in skin) if lesion is None else lesion
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = skin
    yy, xx = np.mgrid[:h, :w]
    img[(yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius**2] = lesion
    noise = np.random.RandomState(7).normal(0.0, 0.004, img.shape)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


def _expected_ita(color) -> float:
    lab = fz.srgb_to_lab(np.array([[color]], dtype=np.float64))[0, 0]
    return float(fz.ita_degrees(lab[0], lab[2]))


@pytest.mark.parametrize(("ita_value", "label"), [
    (63.0, "I"), (40.0001, "I"), (40.0, "II"), (23.0001, "II"), (23.0, "III"), (12.0001, "III"),
    (12.0, "IV"), (0.0001, "IV"), (0.0, "V"), (-24.9999, "V"), (-25.0, "VI"), (-80.0, "VI")])
def test_bucket_edges_belong_to_the_darker_type(ita_value, label):
    assert ita.bucket_label(fz.fst_bucket(ita_value)) == label


def test_abstention_has_its_own_label_and_never_a_roman_numeral():
    assert ita.bucket_label(None) == ita.ABSTAIN == "abstain"
    assert set(ita.BUCKET_LABELS) == {"I", "II", "III", "IV", "V", "VI"}


@pytest.mark.parametrize("bucket", sorted(SKIN))
def test_otsu_estimator_recovers_the_known_ita_and_bucket_of_the_skin(bucket):
    out = ita.global_estimates(_lesion_image(SKIN[bucket]))["otsu"]
    assert out["status"] == "ok" and out["bucket"] == ita.ROMAN[bucket - 1]
    assert out["ita"] == pytest.approx(_expected_ita(SKIN[bucket]), abs=1.0)
    assert out["L_median"] is not None and out["b_median"] is not None


def test_whole_image_estimator_is_pulled_by_the_lesion_unlike_the_otsu_one():
    img = _lesion_image(SKIN[1], lesion=SKIN[6], radius=42)  # a type-VI-dark lesion over half of the frame
    both = ita.global_estimates(img)
    assert both["otsu"]["bucket"] == "I"
    assert both["whole_image"]["bucket"] == "VI"


def test_a_grey_image_has_no_bucket_but_keeps_its_L_for_the_terciles():
    out = ita.global_estimates(_lesion_image((0.6, 0.6, 0.6)))["otsu"]
    assert out["status"] == fz.ITA_INDETERMINATE and out["bucket"] is None and out["ita"] is None
    assert out["L_median"] is not None


def test_a_black_image_abstains_with_a_status_and_no_numbers():
    out = ita.global_estimates(np.zeros((60, 80, 3), dtype=np.float32))
    for name in ("otsu", "whole_image"):
        assert out[name]["bucket"] is None and out[name]["status"] != "ok"


def test_estimates_are_json_safe_floats_or_none():
    import json

    json.dumps(ita.global_estimates(_lesion_image(SKIN[3])), allow_nan=False)
    json.dumps(ita.global_estimates(np.zeros((60, 80, 3), dtype=np.float32)), allow_nan=False)


def test_terciles_are_quantiles_of_the_dataset_with_ties_going_to_the_darker_side():
    values = list(range(1, 10))  # 1..9 -> cuts at 1/3 and 2/3 of the quantile function
    labels, cuts = ita.assign_terciles([float(v) for v in values])
    assert cuts == {"low": pytest.approx(np.quantile(values, 1 / 3)), "high": pytest.approx(np.quantile(values, 2 / 3)),
                    "n": 9}
    assert labels == ["dark"] * 3 + ["medium"] * 3 + ["light"] * 3


def test_terciles_skip_images_without_a_skin_L_and_mark_them_unavailable():
    labels, cuts = ita.assign_terciles([10.0, None, 20.0, 30.0, None, 40.0, 50.0, 60.0])
    assert labels[1] == labels[4] == ita.UNAVAILABLE and cuts["n"] == 6
    assert labels[0] == "dark" and labels[-1] == "light"


def test_terciles_refuse_to_invent_cuts_from_too_few_images():
    labels, cuts = ita.assign_terciles([10.0, 20.0])
    assert labels == [ita.UNAVAILABLE] * 2 and cuts == {"low": None, "high": None, "n": 2}


def test_terciles_follow_the_convention_of_the_pool_manifest():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plan-segmentacion" / "comun" / "evaluacion"))
    import fst_flags
    import fst_pseudo_manifest as fm

    assert ita.TERCILES == fst_flags.TONE_LABELS and ita.UNAVAILABLE == fst_flags.TONE_UNAVAILABLE
    values = [float(v) for v in np.random.RandomState(0).uniform(30, 80, 301)]
    mine, cuts = ita.assign_terciles(values)
    records = [{"L_median": v, "status": "ok"} for v in values]
    for r in records:
        r["tone_tercile"] = fst_flags.TONE_UNAVAILABLE
    ref = fm.assign_tone_terciles(records)
    assert (cuts["low"], cuts["high"]) == (ref["low"], ref["high"])
    assert mine == [r["tone_tercile"] for r in records]


def _write(tmp_path, name, size, fmt="JPEG", color=(180, 130, 100)):
    path = tmp_path / name
    Image.new("RGB", size, color).save(path, fmt)
    return path


def test_decode_keeps_the_native_grid_of_a_ham_sized_image(tmp_path):
    img = ita.decode_canonical(_write(tmp_path, "a.jpg", (600, 450)))
    assert img.shape == (450, 600, 3) and img.dtype == np.float32 and 0.0 <= img.min() and img.max() <= 1.0


def test_decode_reduces_a_large_image_to_the_canonical_short_side_keeping_the_aspect(tmp_path):
    img = ita.decode_canonical(_write(tmp_path, "big.jpg", (3000, 2000)))
    assert img.shape == (ita.CANONICAL_SHORT_SIDE, 675, 3)


def test_decode_never_upsamples(tmp_path):
    assert ita.decode_canonical(_write(tmp_path, "s.png", (200, 100), "PNG")).shape == (100, 200, 3)


def test_decode_is_deterministic_and_preserves_the_colour(tmp_path):
    path = _write(tmp_path, "big.jpg", (2400, 1800))
    first, second = ita.decode_canonical(path), ita.decode_canonical(path)
    assert np.array_equal(first, second)
    assert np.allclose(first.mean(axis=(0, 1)), np.array([180, 130, 100]) / 255.0, atol=0.02)


def test_an_unreadable_file_is_a_typed_error_naming_the_file(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(ita.ImageDecodeError, match="broken.jpg"):
        ita.decode_canonical(bad)
    with pytest.raises(ita.ImageDecodeError):
        ita.decode_canonical(tmp_path / "missing.jpg")


def test_a_value_exactly_on_a_cut_goes_dark_at_the_low_cut_and_medium_at_the_high_cut():
    labels, cuts = ita.assign_terciles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # cuts land exactly on 3 and 5
    assert (cuts["low"], cuts["high"]) == (3.0, 5.0)
    assert labels == ["dark", "dark", "dark", "medium", "medium", "light", "light"]
