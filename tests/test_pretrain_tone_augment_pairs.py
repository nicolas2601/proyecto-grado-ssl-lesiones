"""Contracts for the two-channel tone pair materializer.

Modelled on tests/test_pretrain_transformed_smoke.py, which covers the canonical
L*-only generator, so the two can be compared row by row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dermssl import mask_manifest
from dermssl.pretrain.tone_augment_pairs import (
    MATERIALIZATION_STEP,
    TRANSFORM_NAME,
    generate_tone_augmentations,
    generate_tone_augmentations_from_records,
)
from dermssl.pretrain.tone_transforms import MANIFEST_FILENAME
from dermssl.tone_augment_lab import MAX_ABS_ITA_DELTA, sample_ita_delta

REVISION = "pool-v1"


def _mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[11:21, 11:21] = True
    return mask


def _build(
    tmp_path: Path, *, eligible: bool = True, quarantine_reason: str = "degenerate_area"
) -> dict:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    frame = np.full((32, 32, 3), [190, 140, 110], dtype=np.uint8)
    mask = _mask()
    frame[mask] = [90, 55, 50]
    Image.fromarray(frame).save(images / "sample.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(masks / "sample.png")

    record = mask_manifest.build_mask_record(
        dataset="pool",
        image_id="sample",
        source_revision=REVISION,
        mask=mask,
        producer="teacher-026",
        version="1",
    )
    if not eligible:
        # El motivo real viaja en el registro, como en el pool: las 421 que
        # abstienen traen degenerate_area, y una imagen sin mascara traeria
        # missing_mask. El generador tiene que propagarlo, no aplanarlo.
        record = {
            **record,
            "eligible": False,
            "quality": "degenerate",
            "quarantine_reason": quarantine_reason,
        }
    manifest = tmp_path / "mask_manifest.json"
    manifest.write_text(
        json.dumps({"source_revision": REVISION, "masks": [record]}), encoding="utf-8"
    )
    return {"image_root": images, "mask_root": masks, "mask_manifest_path": manifest}


def _run(tmp_path: Path, *, seed: int = 42, out: str = "out") -> dict:
    inputs = _build(tmp_path / f"in-{out}")
    return generate_tone_augmentations(
        output_root=tmp_path / out,
        seed=seed,
        source_revision=REVISION,
        **inputs,
    )


def test_manifest_shape_matches_the_canonical_generator(tmp_path):
    payload = _run(tmp_path)
    assert payload["transform"] == TRANSFORM_NAME
    assert payload["schema_version"] == "1.0"
    assert set(payload["summary"]) >= {"total", "transformed", "rejected"}
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["rejected"] == 0
    assert "revision" in payload
    assert (tmp_path / "out" / MANIFEST_FILENAME).is_file()


def test_rows_carry_realized_effect_in_both_channels(tmp_path):
    row = _run(tmp_path)["transforms"][0]
    for key in (
        "delta_l_requested",
        "delta_l_realized",
        "delta_b_requested",
        "delta_b_realized",
        "gamut_clipped_pixels",
        "b_sign_flipped_pixels",
        "ita_delta",
    ):
        assert key in row, key
    # b* moves in the opposite direction to L*, four times as far.
    assert row["delta_b_requested"] * row["delta_l_requested"] < 0


def test_delta_is_reproducible_from_the_manifest_alone(tmp_path):
    """La magnitud se deriva del digest de la fuente, no de un RNG mutable."""
    payload = _run(tmp_path, seed=43)
    row = payload["transforms"][0]
    expected = sample_ita_delta(43, MATERIALIZATION_STEP, row["source_sha256"])
    assert row["ita_delta"] == pytest.approx(expected)
    assert abs(row["ita_delta"]) <= MAX_ABS_ITA_DELTA


def test_different_seeds_give_different_pairs(tmp_path):
    first = _run(tmp_path, seed=42, out="a")["transforms"][0]
    second = _run(tmp_path, seed=44, out="b")["transforms"][0]
    assert first["source_sha256"] == second["source_sha256"]
    assert first["ita_delta"] != second["ita_delta"]
    assert first["transformed_sha256"] != second["transformed_sha256"]


def test_lesion_is_preserved_in_the_written_file(tmp_path):
    """La preservacion tiene que sobrevivir el viaje a uint8 y a disco."""
    inputs = _build(tmp_path / "in")
    generate_tone_augmentations(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
    )
    original = np.asarray(Image.open(inputs["image_root"] / "sample.png").convert("RGB"))
    written = np.asarray(Image.open(tmp_path / "out" / "sample.png").convert("RGB"))
    mask = _mask()
    assert np.array_equal(written[mask], original[mask])
    assert not np.array_equal(written[~mask], original[~mask])


def test_ineligible_mask_becomes_counted_identity(tmp_path):
    """Una mascara inutilizable no transforma, no levanta, y no desaparece.

    421 de las 33.620 del pool abstienen. Levantar abortaria la
    materializacion en la primera; dejarlas caer haria que el brazo tratado
    entrene sobre un conjunto distinto al de su control, que es un segundo
    factor.
    """
    inputs = _build(tmp_path / "in", eligible=False)
    payload = generate_tone_augmentations(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
    )
    row = payload["transforms"][0]
    assert row["applied"] is False
    assert row["fallback_reason"] == "degenerate_area"
    assert row["ita_delta"] == 0.0
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["transformed"] == 0
    assert payload["summary"]["rejected"] == 0
    # El campo no puede variar, asi que tiene que venir marcado como tal.
    assert payload["summary"]["rejected_is_structurally_zero"] is True
    assert payload["summary"]["transformed_fraction"] == 0.0
    assert payload["summary"]["fallbacks_by_reason"] == {"degenerate_area": 1}

    original = np.asarray(Image.open(inputs["image_root"] / "sample.png").convert("RGB"))
    written = np.asarray(Image.open(tmp_path / "out" / "sample.png").convert("RGB"))
    assert np.array_equal(written, original)


def test_the_two_causes_of_abstention_stay_separable_in_the_manifest(tmp_path):
    """Una mascara degenerada y una imagen sin mascara no son el mismo hallazgo.

    Las 421 del pool son fallos del teacher 026 semisupervisado propio, que se
    pueden medir y mejorar. Las imagenes sin mascara nunca pasaron por el
    teacher. Las dos diluyen el efecto y las dos hay que declararlas, pero si
    el manifiesto las aplana bajo una sola etiqueta la dilucion deja de ser
    atribuible, que es el error que todo este contraste existe para evitar.
    """
    degenerate = generate_tone_augmentations(
        output_root=tmp_path / "a", seed=42, source_revision=REVISION,
        **_build(tmp_path / "in-a", eligible=False, quarantine_reason="degenerate_area"),
    )
    missing = generate_tone_augmentations(
        output_root=tmp_path / "b", seed=42, source_revision=REVISION,
        **_build(tmp_path / "in-b", eligible=False, quarantine_reason="missing_mask"),
    )
    assert degenerate["summary"]["fallbacks_by_reason"] == {"degenerate_area": 1}
    assert missing["summary"]["fallbacks_by_reason"] == {"missing_mask": 1}
    # Las dos siguen siendo identidad contada: separar la causa no cambia el trato.
    for payload in (degenerate, missing):
        assert payload["transforms"][0]["applied"] is False
        assert payload["summary"]["transformed"] == 0
        assert payload["summary"]["rejected"] == 0


def test_a_frame_with_no_mask_at_all_becomes_a_counted_identity(tmp_path):
    """Las 1.134 clinicas de pad_adapt no tienen mascara y aun asi deben entrar.

    El teacher 026 es de dermatoscopia y nunca corrio sobre fotos clinicas de
    celular, asi que no hay archivo que cargar. Tienen que estar en el brazo
    tratado igual: el control toma el 25 % de cada lote del rol clinico, asi que
    dejarlas fuera haria que los dos brazos entrenen sobre datos distintos. La
    alternativa, correr el segmentador fuera de dominio, inventaria mascaras de
    calidad desconocida dentro del brazo tratado.
    """
    inputs = _build(tmp_path / "in", eligible=False, quarantine_reason="missing_mask")
    # Nada que cargar: se borra la mascara del disco, como en pad_adapt.
    (inputs["mask_root"] / "sample.png").unlink()

    payload = generate_tone_augmentations(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
    )
    row = payload["transforms"][0]
    assert row["applied"] is False
    assert row["fallback_reason"] == "missing_mask"
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["fallbacks_by_reason"] == {"missing_mask": 1}

    original = np.asarray(Image.open(inputs["image_root"] / "sample.png").convert("RGB"))
    written = np.asarray(Image.open(tmp_path / "out" / "sample.png").convert("RGB"))
    assert np.array_equal(written, original)


def test_an_eligible_record_still_requires_its_mask(tmp_path):
    """Relajar la exigencia para las inelegibles no puede relajarla para las demas.

    Cuando la mascara SI decide que pixeles se protegen, su ausencia es un fallo
    y tiene que levantar.
    """
    inputs = _build(tmp_path / "in")
    (inputs["mask_root"] / "sample.png").unlink()
    with pytest.raises(FileNotFoundError):
        generate_tone_augmentations(
            output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
        )


def test_an_abstention_without_a_declared_reason_is_still_labelled(tmp_path):
    """Un registro sin quarantine_reason no puede producir una etiqueta vacia."""
    payload = generate_tone_augmentations(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION,
        **_build(tmp_path / "in", eligible=False, quarantine_reason=None),
    )
    assert payload["transforms"][0]["fallback_reason"] == "ineligible_mask"
    assert payload["summary"]["fallbacks_by_reason"] == {"ineligible_mask": 1}


def test_identity_and_transformed_rows_report_the_same_fields(tmp_path):
    """Las dos rutas no pueden divergir en que reportan."""
    good = generate_tone_augmentations(
        output_root=tmp_path / "a", seed=42, source_revision=REVISION,
        **_build(tmp_path / "in-a"),
    )["transforms"][0]
    bad = generate_tone_augmentations(
        output_root=tmp_path / "b", seed=42, source_revision=REVISION,
        **_build(tmp_path / "in-b", eligible=False),
    )["transforms"][0]
    assert set(good) == set(bad)


def _build_records_cache(tmp_path: Path, *, eligible: bool = True) -> dict:
    """La forma real del cache del pool: JSONL, mascaras planas, rutas absolutas."""
    images = tmp_path / "datos" / "isic2019" / "train-image" / "image"
    masks = tmp_path / "cache" / "masks" / "source_fit"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    frame = np.full((32, 32, 3), [190, 140, 110], dtype=np.uint8)
    mask = _mask()
    frame[mask] = [90, 55, 50]
    Image.fromarray(frame).save(images / "ISIC_0000002.jpg")
    Image.fromarray(mask.astype(np.uint8) * 255).save(masks / "ISIC_0000002.png")

    record = mask_manifest.build_mask_record(
        dataset="pool", image_id="ISIC_0000002", source_revision=REVISION,
        mask=mask, producer="teacher-026", version="1",
    )
    if not eligible:
        record = {**record, "eligible": False, "quality": "degenerate"}
    record = {
        **record,
        "source_image_path": str(images / "ISIC_0000002.jpg"),
        "mask_path": "masks/source_fit/ISIC_0000002.png",
    }
    jsonl = tmp_path / "cache" / "masks.source_fit.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return {
        "image_root": tmp_path / "datos",
        "records_path": jsonl,
        "records_root": tmp_path / "cache",
    }


def test_records_entry_point_consumes_the_pool_cache_layout(tmp_path):
    """El cache del pool es JSONL con mascaras planas; el camino canonico no lo lee.

    Los dos desajustes se encontraron verificando, no por una corrida fallida:
    _load_records espera un JSON con lista 'masks', y _load_mask reconstruye la
    ruta espejando subdirectorios.
    """
    inputs = _build_records_cache(tmp_path)
    payload = generate_tone_augmentations_from_records(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
    )
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["transformed"] == 1
    row = payload["transforms"][0]
    # La ruta relativa preserva la cohorte, que es como collect_paired_images
    # empareja original con transformada bajo dos raices.
    assert row["relative_path"] == "isic2019/train-image/image/ISIC_0000002.jpg"
    assert (tmp_path / "out" / row["relative_path"]).is_file()


def test_jpeg_sources_break_lesion_preservation(tmp_path):
    """Hallazgo 2026-11-03: con fuentes .jpg el contrato de lesion NO se cumple.

    `config._validate_transformed_paths` exige que las rutas relativas original,
    transformada y del manifiesto coincidan EXACTAMENTE, extension incluida. Con
    fuentes .jpg la copia tiene que escribirse como JPEG, y la compresion con
    perdida altera la lesion: medido sobre un parche sintetico, 100 % de los
    pixeles de lesion cambian con calidad por defecto (delta maximo 36), 83,9 %
    con calidad 100 y sin submuestreo (delta maximo 3). Solo PNG da delta cero.

    Esto no es un detalle de formato: en el brazo tratado las dos vistas
    diferirian en tono sobre la piel Y en ruido JPEG sobre la lesion, mientras
    el control no tiene ese ruido. Es un segundo factor.

    Este test fija el hallazgo para que no se pierda. Cuando se adopte el
    espejo PNG del pool (ambos brazos leyendo del mismo espejo sin perdida),
    debe invertirse a una asercion de igualdad exacta.
    """
    inputs = _build_records_cache(tmp_path)
    generate_tone_augmentations_from_records(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
    )
    rel = "isic2019/train-image/image/ISIC_0000002.jpg"
    original = np.asarray(Image.open(inputs["image_root"] / rel).convert("RGB"))
    written = np.asarray(Image.open(tmp_path / "out" / rel).convert("RGB"))
    mask = _mask()
    assert not np.array_equal(written[mask], original[mask]), (
        "si esto pasa a ser igual, el espejo PNG ya esta en uso: invertir el test"
    )


def test_png_sources_do_preserve_the_lesion_exactly(tmp_path):
    """El mismo camino, con fuentes PNG, si cumple el contrato."""
    inputs = _build_records_cache(tmp_path)
    images = inputs["image_root"] / "isic2019" / "train-image" / "image"
    src = images / "ISIC_0000002.jpg"
    png = images / "ISIC_0000002.png"
    Image.open(src).save(png)
    src.unlink()

    jsonl = inputs["records_path"]
    record = json.loads(jsonl.read_text())
    record["source_image_path"] = str(png)
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    generate_tone_augmentations_from_records(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
    )
    rel = "isic2019/train-image/image/ISIC_0000002.png"
    original = np.asarray(Image.open(png).convert("RGB"))
    written = np.asarray(Image.open(tmp_path / "out" / rel).convert("RGB"))
    assert np.array_equal(written[_mask()], original[_mask()])


def test_records_entry_point_handles_ineligible_as_identity(tmp_path):
    inputs = _build_records_cache(tmp_path, eligible=False)
    payload = generate_tone_augmentations_from_records(
        output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
    )
    assert payload["summary"]["transformed"] == 0
    assert payload["summary"]["fallbacks_by_reason"] == {"ineligible_mask": 1}


def test_records_entry_point_refuses_a_missing_mask(tmp_path):
    inputs = _build_records_cache(tmp_path)
    (tmp_path / "cache" / "masks" / "source_fit" / "ISIC_0000002.png").unlink()
    with pytest.raises(FileNotFoundError):
        generate_tone_augmentations_from_records(
            output_root=tmp_path / "out", seed=42, source_revision=REVISION, **inputs
        )


def test_records_entry_point_refuses_a_foreign_revision(tmp_path):
    inputs = _build_records_cache(tmp_path)
    with pytest.raises(ValueError):
        generate_tone_augmentations_from_records(
            output_root=tmp_path / "out", seed=42, source_revision="otra-revision", **inputs
        )


def test_output_root_may_not_overlap_the_images(tmp_path):
    inputs = _build(tmp_path / "in")
    with pytest.raises(ValueError):
        generate_tone_augmentations(
            output_root=inputs["image_root"],
            seed=42,
            source_revision=REVISION,
            **inputs,
        )
