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

from dermssl.circle import lab_to_srgb
from dermssl.fitzpatrick import MIN_B_FOR_ITA, estimate_fst, ita_degrees, srgb_to_lab
from dermssl.tone_augment_lab import (
    B_PER_ITA_DEGREE,
    L_PER_ITA_DEGREE,
    augment_skin_tone,
    summarize_augmentation_records,
)
from dermssl.tone_augment_lab_v2 import (
    MAX_POPULATION_SHIFT_FRACTION,
    _population_shift_fraction,
    _reachable_bracket,
    _skin_ita_at_t,
    _skin_lab_for_curve,
    _skin_state_at_t,
    _unified_skin_fov,
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


def _textured_frame(seed, size=96, base=(0.35, 0.28, 0.20), shadow_fraction=0.08):
    """A skin patch with real pixel-to-pixel variance, not a flat colour.

    Needed to exercise the angle-of-medians-vs-median-of-angles distinction:
    on a flat frame every pixel has the same L and b, so the two aggregations
    agree trivially and the bug these tests guard against cannot show up.
    """
    rng = np.random.default_rng(seed)
    base = np.array(base)
    img = np.zeros((size, size, 3))
    noise = rng.normal(1.0, 0.15, size=(size, size, 1))
    img[:] = np.clip(base[None, None, :] * noise, 0, 1)
    shadow = rng.random((size, size)) < shadow_fraction
    img[shadow] = img[shadow] * 0.1
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = True
    img[mask] = np.clip(img[mask] - 0.04, 0.0, 1.0)
    return img, mask


def _textured_frame_sequence(seed, index, size=128, shadow_fraction=0.08):
    """Reproduces the Nth frame of a (seed, base-per-frame) generation sequence.

    Matches the exact construction used to discover the aggregation-order
    regression below: a random base is drawn per frame from the SAME rng
    stream, so reproducing frame ``index`` requires drawing (and discarding)
    the ``index`` frames before it, not drawing ``index`` directly.
    """
    rng = np.random.default_rng(seed)
    img = mask = None
    for _ in range(index + 1):
        base = rng.uniform(0.25, 0.6, size=3)
        img = np.zeros((size, size, 3))
        noise = rng.normal(1.0, 0.15, size=(size, size, 1))
        img[:] = np.clip(base[None, None, :] * noise, 0, 1)
        shadow = rng.random((size, size)) < shadow_fraction
        img[shadow] = img[shadow] * 0.1
        mask = np.zeros((size, size), dtype=bool)
        mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = True
        img[mask] = np.clip(img[mask] - 0.04, 0.0, 1.0)
    return img, mask


def test_ita_is_the_angle_of_the_medians_not_the_median_of_angles():
    """Regresion: aterrizo en el bug real que media el residuo de la via B.

    arctan es no lineal: median(arctan(L_i, b_i)) y arctan(median(L),
    median(b)) son estadisticos distintos salvo que no haya varianza entre
    pixeles. ``fitzpatrick._skin_ita`` (y por lo tanto ``estimate_fst``, y
    cualquiera que mida la imagen de salida) usa la segunda forma. En el
    diagnostico original, un caso con esta misma textura media mas de 10
    grados de diferencia entre las dos formas -- mismos pixeles, mismo t,
    sin reclasificacion de borde ni submuestreo de por medio.
    """
    img, mask = _textured_frame_sequence(21, index=4)
    skin_lab = _skin_lab_for_curve(img, mask, None)
    solved = solve_ita_delta(skin_lab, 42, 4, DIGEST)
    assert solved is not None
    t = solved["ita_delta"]

    ita_correct, b_med = _skin_state_at_t(skin_lab, t)

    shifted = skin_lab.copy()
    shifted[:, 0] = np.clip(skin_lab[:, 0] - t * L_PER_ITA_DEGREE, 0.0, 100.0)
    shifted[:, 2] = skin_lab[:, 2] + t * B_PER_ITA_DEGREE
    after_lab = srgb_to_lab(lab_to_srgb(shifted))
    median_of_angles = float(np.median(ita_degrees(after_lab[:, 0], after_lab[:, 2])))
    angle_of_medians = float(
        ita_degrees(np.median(after_lab[:, 0]), np.median(after_lab[:, 2]))
    )

    assert ita_correct == pytest.approx(angle_of_medians)
    assert abs(median_of_angles - angle_of_medians) > 1.0  # the two disagree on real texture

    out, _ = augment_skin_tone(img, mask, None, ita_delta=t, max_abs_ita_delta=100.0)
    final = estimate_fst(out, mask=mask, border_mask=None)
    assert final.status == "ok"
    # Not exact: fitzpatrick.estimate_fst also runs its own border exclusion
    # and skin detector, a residual _population_shift_fraction does not fully
    # eliminate (see that function's docstring). What matters here is the
    # order-of-magnitude fix: within ~1 degree of the angle-of-medians curve,
    # not within ~10 degrees of the median-of-angles bug.
    assert final.ita == pytest.approx(ita_correct, abs=1.0)
    assert abs(final.ita - median_of_angles) > 5.0


def test_population_shift_fraction_is_zero_without_near_black_pixels():
    img, mask = _frame()
    skin_lab = _skin_lab_for_curve(img, mask, None)
    assert _population_shift_fraction(skin_lab, 10.0) == 0.0


def test_population_shift_fraction_detects_border_crossing():
    """Regresion: hallazgo original de maestro, real pero mas chico que el bug de agregacion."""
    img, mask = _textured_frame(9, base=(0.30, 0.20, 0.10), shadow_fraction=0.15)
    skin_lab = _skin_lab_for_curve(img, mask, None)
    # a large positive t should push a meaningful fraction of the shadow
    # pixels across BORDER_LUMINANCE_THRESHOLD in this particular texture.
    fractions = [_population_shift_fraction(skin_lab, float(t)) for t in range(0, 30, 2)]
    assert max(fractions) > MAX_POPULATION_SHIFT_FRACTION


def test_augment_skin_tone_v2_abstains_on_large_population_shift():
    img, mask = _textured_frame(9, base=(0.30, 0.20, 0.10), shadow_fraction=0.15)
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    if record["fallback_reason"] == "border_population_shift_too_large":
        assert np.array_equal(out, img)
        assert record["applied"] is False
        assert record["ita_objetivo"] is not None  # still reported for diagnosis


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


def _frame_with_vignette(size=96, value=(0.6, 0.5, 0.4), vignette_width=10):
    """A skin patch with a near-black dermoscope vignette ring outside the lesion.

    Needed to exercise the two skin-population definitions diverging:
    on a plain frame (no vignette) both already agree everywhere, so a test
    without one cannot tell a unification bug from a correct unification.
    """
    img = np.zeros((size, size, 3))
    img[:] = np.asarray(value)
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    img[radius > size / 2 - vignette_width] = 0.005  # below BORDER_LUMINANCE_THRESHOLD
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = True
    img[mask] = (0.33, 0.21, 0.19)
    return img, mask


def test_unified_fov_excludes_the_vignette_the_selector_alone_would_keep():
    """Regresion de la Tarea 1A: las dos definiciones de piel, unificadas.

    `tone_normalization._skin_selector` por si sola (sin excluir casi-negro)
    incluiria el anillo de vineteado como piel transformable. El fov
    unificado no.
    """
    img, mask = _frame_with_vignette()
    from dermssl.tone_normalization import _skin_selector

    selector_only = _skin_selector(mask, None)
    unified = ~mask & _unified_skin_fov(img, None)
    vignette_ring = selector_only & ~unified
    assert vignette_ring.any()


def test_transformed_pixels_never_include_the_vignette():
    """La geometria congelada, llamada con el fov unificado, no toca el vineteado.

    ``feather_sigma=0`` aisla la unificacion de piel del suavizado del
    borde: con feather, CUALQUIER limite de mascara sangra un anillo de
    pocos pixeles por diseno (es lo que prueba
    ``test_feather_softens_the_boundary_without_touching_the_lesion`` en el
    modulo congelado), y eso incluiria el borde del vineteado tambien. Sin
    feather, si las dos definiciones de piel volvieran a divergir (alguien
    pasa el fov crudo en vez del unificado), este test falla: el vineteado
    quedaria transformado en vez de identico.
    """
    img, mask = _frame_with_vignette()
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST, feather_sigma=0.0)
    if record["applied"]:
        vignette = _skin_selector_vignette_mask(img, mask)
        assert vignette.any()
        assert np.array_equal(out[vignette], img[vignette])


def _skin_selector_vignette_mask(img, mask):
    from dermssl.tone_normalization import _skin_selector

    return _skin_selector(mask, None) & ~_unified_skin_fov(img, None)


def test_degenerate_lesion_mask_falls_back_through_the_frozen_guard():
    img, _ = _frame()
    full = np.ones(img.shape[:2], dtype=bool)
    out, record = augment_skin_tone_v2(img, full, 42, 0, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False


def test_lightening_can_lose_measurability_but_darkening_of_equal_magnitude_cannot():
    """Regresion: la perdida de medibilidad tras transformar es asimetrica.

    Hallazgo de una corrida sobre S-SYNTH real (745 casos aplicados): aclarar
    (t negativo) dejo el 24,3% de las salidas con b* bajo MIN_B_FOR_ITA,
    oscurecer (t positivo) el 0,0%, con corte limpio justo en MIN_B_FOR_ITA
    (max no-medible 4,973 contra min medible 5,035). No es un defecto de este
    modulo: es estructural a la geometria congelada. ``delta_b = t *
    B_PER_ITA_DEGREE`` y ``delta_l = -t * L_PER_ITA_DEGREE`` tienen signos
    acoplados -- aclarar (``delta_l > 0``, por lo tanto ``t < 0``) mueve b*
    HACIA cero en la misma operacion; oscurecer (``t > 0``) lo aleja. El cap
    simetrico de Corbin (+/-25) no refleja esto porque la literatura no mide
    si la salida sigue siendo medible por el mismo instrumento que fijo el
    objetivo -- ver docs/63-instrumento-ita-alcance-y-limites.md.

    Verificado aqui sobre la geometria congelada pura (``augment_skin_tone``
    de ``tone_augment_lab``, sin pasar por el solver de v2): mismo |t|, b*
    inicial moderado (~17), un signo cruza MIN_B_FOR_ITA y el otro no.
    """
    img, mask = _frame()  # base (0.72, 0.58, 0.49): b* inicial ~17, moderado
    magnitude = 25.0

    _, darken_record = augment_skin_tone(
        img, mask, None, ita_delta=magnitude, feather_sigma=0.0, max_abs_ita_delta=100.0
    )
    _, lighten_record = augment_skin_tone(
        img, mask, None, ita_delta=-magnitude, feather_sigma=0.0, max_abs_ita_delta=100.0
    )

    assert darken_record["skin_median_b_after"] >= MIN_B_FOR_ITA
    assert lighten_record["skin_median_b_after"] < MIN_B_FOR_ITA


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
