"""Tests for the two-channel CIELAB tone augmentation.

One test per binary criterion of docs/estado/clf-tono-lab-augment.md, so a
failure names the criterion it broke rather than a symptom.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl.fitzpatrick import srgb_to_lab
from dermssl.mask_manifest import mask_sha256
from dermssl.tone_augment_lab import (
    B_PER_ITA_DEGREE,
    L_PER_ITA_DEGREE,
    MAX_ABS_ITA_DELTA,
    augment_skin_tone,
    augment_skin_tone_from_mask_record,
    sample_ita_delta,
    summarize_augmentation_records,
)

DIGEST = "a" * 64


def _frame(value=(0.72, 0.58, 0.49), size=64):
    """A plain skin-coloured frame with a central lesion blob."""
    img = np.zeros((size, size, 3), dtype=np.float64)
    img[:] = np.asarray(value, dtype=np.float64)
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = True
    img[mask] = (0.33, 0.21, 0.19)
    return img, mask


def test_zero_delta_is_exact_identity():
    """Criterio 1: magnitud cero devuelve la imagen sin cambio alguno."""
    img, mask = _frame()
    out, record = augment_skin_tone(img, mask, ita_delta=0.0)
    assert np.array_equal(out, img)
    assert record["applied"] is True
    assert record["delta_l_requested"] == 0.0
    assert record["delta_b_requested"] == 0.0


@pytest.mark.parametrize("delta", [-25.0, -7.5, 3.0, 25.0])
@pytest.mark.parametrize("sigma", [0.0, 2.0, 6.0])
def test_lesion_preserved_pixel_for_pixel(delta, sigma):
    """Criterio 2: la lesion no cambia para ningun delta ni ningun sigma."""
    img, mask = _frame()
    out, _ = augment_skin_tone(img, mask, ita_delta=delta, feather_sigma=sigma)
    assert np.array_equal(out[mask], img[mask])


def test_degenerate_mask_falls_back_to_identity():
    """Criterio 3: mascara degenerada no edita el marco completo."""
    img, _ = _frame()
    empty = np.zeros(img.shape[:2], dtype=bool)
    out, record = augment_skin_tone(img, empty, ita_delta=20.0)
    assert np.array_equal(out, img)
    assert record["applied"] is False
    assert record["fallback_reason"] == "degenerate_lesion_mask"

    full = np.ones(img.shape[:2], dtype=bool)
    out, record = augment_skin_tone(img, full, ita_delta=20.0)
    assert np.array_equal(out, img)
    assert record["applied"] is False


def test_channel_ratio_matches_the_paper():
    """Criterio 4: b* se pide 0,50 por grado y L* 0,12, con signo opuesto."""
    img, mask = _frame()
    delta = 10.0
    _, record = augment_skin_tone(img, mask, ita_delta=delta, feather_sigma=0.0)
    assert record["delta_b_requested"] == pytest.approx(delta * B_PER_ITA_DEGREE)
    assert record["delta_l_requested"] == pytest.approx(-delta * L_PER_ITA_DEGREE)
    ratio = abs(record["delta_b_requested"] / record["delta_l_requested"])
    assert ratio == pytest.approx(B_PER_ITA_DEGREE / L_PER_ITA_DEGREE)


def test_realized_shift_tracks_the_request_without_clipping():
    """Criterio 8: el efecto realizado se mide en los dos canales."""
    img, mask = _frame()
    _, record = augment_skin_tone(img, mask, ita_delta=6.0, feather_sigma=0.0)
    assert record["gamut_clipped_pixels"] == 0
    assert record["delta_b_realized"] == pytest.approx(record["delta_b_requested"], abs=0.1)
    assert record["delta_l_realized"] == pytest.approx(record["delta_l_requested"], abs=0.1)


def test_cap_binds_and_is_recorded():
    """Criterio 5: por encima del tope se recorta y queda marcado."""
    img, mask = _frame()
    _, record = augment_skin_tone(img, mask, ita_delta=400.0)
    assert record["ita_delta_applied"] == MAX_ABS_ITA_DELTA
    assert record["ita_delta_capped"] is True

    _, record = augment_skin_tone(img, mask, ita_delta=5.0)
    assert record["ita_delta_capped"] is False


def test_strongly_negative_b_star_does_not_break():
    """Criterio 6, primera mitad: b* muy negativo se transforma sin romperse.

    Bluish dermoscopy is 44% of the pool, so this is the common case, not an
    edge case. b* here is about -23,6: a +12,5 displacement does NOT cross the
    axis, and the record must say so rather than claim a flip.
    """
    img, mask = _frame(value=(0.45, 0.52, 0.68))
    assert np.median(srgb_to_lab(img)[..., 2][~mask]) < -20.0
    out, record = augment_skin_tone(img, mask, ita_delta=25.0, feather_sigma=0.0)
    assert np.all(np.isfinite(out))
    assert record["applied"] is True
    assert record["b_sign_flipped_pixels"] == 0
    assert record["skin_median_b_after"] < 0.0
    assert np.array_equal(out[mask], img[mask])


def test_b_sign_crossing_is_counted_when_it_happens():
    """Criterio 6, segunda mitad: cuando el signo de b* cruza, se publica.

    b* starts near -6,1 and the displacement is +12,5, so the axis is crossed.
    This is the quantity that says how much the transform distorted images the
    ITA could never have described in the first place.
    """
    img, mask = _frame(value=(0.50, 0.52, 0.56))
    before = np.median(srgb_to_lab(img)[..., 2][~mask])
    assert -10.0 < before < 0.0
    out, record = augment_skin_tone(img, mask, ita_delta=25.0, feather_sigma=0.0)
    assert record["b_sign_flipped_pixels"] > 0
    assert record["skin_median_b_after"] > 0.0
    assert np.array_equal(out[mask], img[mask])


def test_sampling_is_deterministic_and_bounded():
    """Criterio 7: el muestreo depende solo de (semilla, paso, hash)."""
    first = sample_ita_delta(42, 7, DIGEST)
    assert first == sample_ita_delta(42, 7, DIGEST)
    assert first != sample_ita_delta(42, 8, DIGEST)
    assert first != sample_ita_delta(43, 7, DIGEST)
    assert first != sample_ita_delta(42, 7, "b" * 64)

    values = [sample_ita_delta(42, step, DIGEST) for step in range(400)]
    assert all(abs(v) <= MAX_ABS_ITA_DELTA for v in values)
    assert min(values) < 0 < max(values)


def test_sampling_does_not_consume_global_rng():
    """Criterio 7: no toca ningun estado RNG mutable compartido."""
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    sample_ita_delta(42, 1, DIGEST)
    assert np.random.random() == before


def test_sampling_rejects_bad_input():
    with pytest.raises(ValueError):
        sample_ita_delta(42, -1, DIGEST)
    with pytest.raises(ValueError):
        sample_ita_delta(42, 1, "short")
    with pytest.raises(ValueError):
        sample_ita_delta(42, 1, "A" * 64)
    with pytest.raises(ValueError):
        sample_ita_delta(42.0, 1, DIGEST)
    with pytest.raises(ValueError):
        sample_ita_delta(True, 1, DIGEST)


def test_gamut_count_is_independent_of_the_feather():
    """El conteo de gamut mide la transformacion, no lo que sobrevive la mezcla.

    Regresion del hallazgo de revision del 2026-11-03: contar sobre la imagen ya
    mezclada subcontaba la perdida real, porque en el anillo suavizado alfa < 1
    y la mezcla no cae exactamente en los bordes [0, 1]. El mismo caso daba
    3655 pixeles con sigma 0 y 1287 con sigma 4.
    """
    img, mask = _frame(value=(0.657, 0.562, 0.150))
    counts = {
        sigma: augment_skin_tone(img, mask, ita_delta=25.0, feather_sigma=sigma)[1][
            "gamut_clipped_pixels"
        ]
        for sigma in (0.0, 2.0, 4.0)
    }
    assert counts[0.0] > 0, "el caso de prueba debe saturar, si no no prueba nada"
    assert counts[0.0] == counts[2.0] == counts[4.0]


def test_gamut_is_counted_with_the_production_sigma():
    """El sigma por defecto es el que se va a usar; tiene que estar cubierto."""
    img, mask = _frame(value=(0.657, 0.562, 0.150))
    _, record = augment_skin_tone(img, mask, ita_delta=25.0)
    assert record["feather_sigma"] == pytest.approx(2.0)
    assert record["gamut_clipped_pixels"] > 0
    assert np.array_equal(
        augment_skin_tone(img, mask, ita_delta=25.0)[0][mask], img[mask]
    )


def test_b_crossing_treats_exact_zero_as_on_the_axis():
    """Cero exacto esta sobre el eje, no de un lado: moverse de ahi no es cruce."""
    img, mask = _frame()
    lab = srgb_to_lab(img).copy()
    lab[..., 2] = 0.0
    from dermssl.circle import lab_to_srgb

    neutral = np.clip(lab_to_srgb(lab), 0.0, 1.0)
    _, record = augment_skin_tone(neutral, mask, ita_delta=25.0, feather_sigma=0.0)
    assert record["b_sign_flipped_pixels"] == 0


def test_feather_softens_the_boundary_without_touching_the_lesion():
    """El suavizado de Corbin actua del lado de la piel, no dentro."""
    img, mask = _frame()
    hard, _ = augment_skin_tone(img, mask, ita_delta=20.0, feather_sigma=0.0)
    soft, _ = augment_skin_tone(img, mask, ita_delta=20.0, feather_sigma=4.0)
    assert not np.array_equal(hard, soft)
    assert np.array_equal(hard[mask], img[mask])
    assert np.array_equal(soft[mask], img[mask])

    # Deep skin is unaffected by the feather; only the ring near the lesion is.
    corner = (slice(0, 4), slice(0, 4))
    assert np.allclose(hard[corner], soft[corner], atol=1e-6)


def test_mask_record_adapter_refuses_mismatched_provenance():
    img, mask = _frame()
    good = {"mask_semantics": "true=lesion", "eligible": True, "mask_sha256": mask_sha256(mask)}
    _, record = augment_skin_tone_from_mask_record(img, mask, good, ita_delta=5.0)
    assert record["mask_sha256"] == good["mask_sha256"]

    with pytest.raises(ValueError):
        augment_skin_tone_from_mask_record(
            img, mask, {**good, "mask_sha256": "0" * 64}, ita_delta=5.0
        )
    with pytest.raises(ValueError):
        augment_skin_tone_from_mask_record(img, mask, {**good, "eligible": False}, ita_delta=5.0)
    with pytest.raises(ValueError):
        augment_skin_tone_from_mask_record(
            img, mask, {**good, "mask_semantics": "true=skin"}, ita_delta=5.0
        )


def test_summary_counts_fallbacks_without_dropping_them():
    img, mask = _frame()
    empty = np.zeros(img.shape[:2], dtype=bool)
    records = [
        augment_skin_tone(img, mask, ita_delta=5.0)[1],
        augment_skin_tone(img, mask, ita_delta=400.0)[1],
        augment_skin_tone(img, empty, ita_delta=5.0)[1],
    ]
    summary = summarize_augmentation_records(records)
    assert summary["images"] == 3
    assert summary["transformed"] == 2
    assert summary["capped"] == 1
    assert summary["fallbacks_by_reason"] == {"degenerate_lesion_mask": 1}
    assert summary["transformed_fraction"] == pytest.approx(2 / 3)


def test_signed_and_absolute_means_are_reported_under_honest_names():
    """El campo abs_mean calculaba una media CON SIGNO hasta 2026-11-04.

    En el pool real leia 0,0201 e invitaba a concluir que la transformacion no
    hace casi nada, sobre una corrida cuyas filas traen ita_delta de 19,73. La
    media con signo da cero porque los desplazamientos son simetricos por
    diseno: es evidencia de que el muestreador no esta sesgado, no de que la
    dosis sea debil. Este test fija que las dos cifras existan y que ninguna
    pueda volver a usar el nombre de la otra.
    """
    img, mask = _frame()
    records = [
        augment_skin_tone(img, mask, ita_delta=8.0)[1],
        augment_skin_tone(img, mask, ita_delta=-8.0)[1],
    ]
    summary = summarize_augmentation_records(records)
    assert summary["transformed"] == 2
    assert summary["ita_delta_applied_mean"] == pytest.approx(0.0)
    assert summary["ita_delta_applied_abs_mean"] == pytest.approx(8.0)
    assert summary["ita_delta_applied_abs_max"] == pytest.approx(8.0)
    # El punto entero: leer solo la media con signo esconde la dosis aplicada.
    assert summary["ita_delta_applied_abs_mean"] > summary["ita_delta_applied_mean"]


def test_rejects_non_finite_parameters():
    img, mask = _frame()
    with pytest.raises(ValueError):
        augment_skin_tone(img, mask, ita_delta=float("nan"))
    with pytest.raises(ValueError):
        augment_skin_tone(img, mask, ita_delta=5.0, feather_sigma=-1.0)
