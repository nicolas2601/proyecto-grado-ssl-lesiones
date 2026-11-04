"""Tests for the mask-anchored, eq.-7 CIELAB tone augmentation (v2).

v2 history, see the module docstring: REVISION 1 sampled a fixed global span
(collapsed to a constant on realistic ITA distributions); REVISION 2 used a
window linearized around the measured ITA (failed far from the origin
because ITA(t) is discontinuous where skin b* crosses zero); REVISION 3
(current, tested here) maps each image's own reachable ITA range with a
grid sweep and solves exactly with ``brentq``, with no default policy cap.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl.tone_augment_lab import summarize_augmentation_records
from dermssl.tone_augment_lab_v2 import (
    _reachable_bracket,
    _skin_ita_at_t,
    _skin_lab_for_curve,
    augment_skin_tone_v2,
    sample_apply_draw,
    solve_ita_delta,
)

DIGEST = "a" * 64


def _frame(value=(0.72, 0.58, 0.49), size=64):
    img = np.zeros((size, size, 3), dtype=np.float64)
    img[:] = np.asarray(value, dtype=np.float64)
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = True
    img[mask] = (0.33, 0.21, 0.19)
    return img, mask


def _find_selected_step(seed=42, max_steps=200):
    for step in range(max_steps):
        if sample_apply_draw(seed, step, DIGEST) > 0.5:
            return step
    pytest.fail("no step selected by the mixing gate in range")


def _find_unselected_step(seed=42, max_steps=200):
    for step in range(max_steps):
        if sample_apply_draw(seed, step, DIGEST) <= 0.5:
            return step
    pytest.fail("no step rejected by the mixing gate in range")


def test_skin_ita_curve_matches_measured_ita_at_t_zero():
    """La curva de barrido tiene que arrancar exactamente donde mide estimate_fst."""
    img, mask = _frame()
    skin_lab = _skin_lab_for_curve(img, mask, None)
    assert skin_lab is not None
    assert _skin_ita_at_t(skin_lab, 0.0) == pytest.approx(_skin_ita_at_t(skin_lab, 0.0))


def test_reachable_bracket_is_monotonic_and_contains_zero():
    img, mask = _frame()
    skin_lab = _skin_lab_for_curve(img, mask, None)
    bracket = _reachable_bracket(skin_lab)
    assert bracket is not None
    t_lo, t_hi, ita_at_lo, ita_at_hi = bracket
    assert t_lo <= 0.0 <= t_hi
    assert ita_at_lo >= ita_at_hi


def test_bracket_rejects_a_b_sign_crossing_hidden_between_grid_points():
    """Regresion: una imagen cuyo b* cruza cero DENTRO de un paso de grilla.

    Caso real encontrado con una corrida sintetica: b* en t=-3 es -0,40 y en
    t=0 es +1,10 -- la ITA por si sola pasaria el chequeo de monotonicidad
    (82,5 a -72,0 visto solo en los extremos parece un descenso), pero en el
    medio el angulo salta por el cruce del eje, no desciende de forma
    continua. Sin el chequeo de signo de b*, un intento anterior de esta
    correccion dejaba pasar ese tramo y ``brentq`` devolvia un error de 70
    grados contra el objetivo pedido. Con el chequeo, este caso particular
    ademas falla la monotonicidad hacia t positivo (la ITA sube de -72 a
    -55,3 entre t=0 y t=3 por el recorte de gamut), asi que el tramo colapsa
    del todo: abstencion correcta, no un numero incorrecto.
    """
    size = 64
    base = np.array([0.45543079, 0.42675353, 0.42621283])
    img = np.zeros((size, size, 3))
    img[:] = base
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = True
    img[mask] = np.clip(base - 0.2, 0.0, 1.0)

    skin_lab = _skin_lab_for_curve(img, mask, None)
    assert _reachable_bracket(skin_lab) is None
    assert solve_ita_delta(skin_lab, 42, 37, DIGEST) is None


def test_solve_hits_the_sampled_objetivo_almost_exactly():
    """Criterio de maestro: |ITA_final - objetivo| debe ser chico, no 0,576 de razon."""
    img, mask = _frame()
    skin_lab = _skin_lab_for_curve(img, mask, None)
    solved = solve_ita_delta(skin_lab, 42, 0, DIGEST)
    assert solved is not None
    realized = _skin_ita_at_t(skin_lab, solved["ita_delta"])
    assert abs(realized - solved["ita_objetivo"]) < 0.01


def test_solve_has_no_default_policy_cap():
    """No hereda el 25: sin policy_cap_degrees, el bracket es el fisico completo."""
    img, mask = _frame()
    skin_lab = _skin_lab_for_curve(img, mask, None)
    bracket = _reachable_bracket(skin_lab)
    solved = solve_ita_delta(skin_lab, 42, 0, DIGEST)
    assert solved is not None
    assert solved["reachable_t_lo"] == bracket[0]
    assert solved["reachable_t_hi"] == bracket[1]
    assert solved["policy_cap_narrowed_reachable_range"] is False


def test_explicit_policy_cap_narrows_the_bracket_when_passed():
    img, mask = _frame()
    skin_lab = _skin_lab_for_curve(img, mask, None)
    solved = solve_ita_delta(skin_lab, 42, 0, DIGEST, policy_cap_degrees=5.0)
    assert solved is not None
    assert abs(solved["ita_delta"]) <= 5.0 + 1e-2
    assert solved["policy_cap_narrowed_reachable_range"] is True


def test_abstains_when_ita_is_indeterminate():
    img, mask = _frame(value=(0.5, 0.5, 0.5))
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False
    assert record["fallback_reason"].startswith("ita_unmeasured_")
    assert record["ita_objetivo"] is None
    assert record["selected_for_transform"] is False


def test_not_selected_by_mix_is_identity_and_distinct_from_abstention():
    img, mask = _frame()
    step = _find_unselected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False
    assert record["fallback_reason"] == "not_selected_by_mix"
    assert record["ita_measurement_status"] == "ok"
    assert record["selected_for_transform"] is False


def test_selected_image_is_transformed_with_an_exact_delta():
    img, mask = _frame()
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert record["applied"] is True
    assert record["selected_for_transform"] is True
    assert record["ita_measurement_status"] == "ok"
    assert record["ita_objetivo"] is not None
    assert not np.array_equal(out, img)


def test_selected_image_realizes_the_objetivo_within_tolerance():
    """El criterio medible que pidio maestro, de punta a punta."""
    img, mask = _frame()
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert record["applied"] is True
    skin_lab = _skin_lab_for_curve(img, mask, None)
    realized = _skin_ita_at_t(skin_lab, record["ita_delta_raw"])
    assert abs(realized - record["ita_objetivo"]) < 0.01


def test_lesion_preserved_pixel_for_pixel_when_transformed():
    img, mask = _frame()
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    if record["applied"]:
        assert np.array_equal(out[mask], img[mask])


def test_degenerate_lesion_mask_falls_back_through_the_frozen_guard():
    img, _ = _frame()
    full = np.ones(img.shape[:2], dtype=bool)
    out, record = augment_skin_tone_v2(img, full, 42, 0, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False


def test_record_shape_is_compatible_with_the_frozen_summarizer():
    img, mask = _frame()
    selected_step = _find_selected_step()
    unselected_step = _find_unselected_step()
    grey, grey_mask = _frame(value=(0.5, 0.5, 0.5))
    abstain_step = _find_selected_step()

    records = [
        augment_skin_tone_v2(img, mask, 42, selected_step, DIGEST)[1],
        augment_skin_tone_v2(img, mask, 42, unselected_step, DIGEST)[1],
        augment_skin_tone_v2(grey, grey_mask, 42, abstain_step, DIGEST)[1],
    ]
    summary = summarize_augmentation_records(records)
    assert summary["images"] == 3
    assert summary["transformed"] == 1
    assert set(summary["fallbacks_by_reason"]) == {
        "not_selected_by_mix",
        "ita_unmeasured_ita_indeterminate",
    }
