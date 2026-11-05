"""Materialize two-channel CIELAB tone pairs for the frozen SSL loop.

Sibling of :mod:`dermssl.pretrain.tone_transforms`, which materializes the
L*-only CIRCLe pairs under ``circle_lab_l_star_v1``. That module is canonical
and is not modified here; this one writes the same manifest shape with a
different transform, so ``--transformed-image-root`` consumes either without the
training loop knowing the difference.

Why a second transform exists at all: every inference-time tone intervention
measured on the frozen encoder degraded classification (fixed L* perturbation
-1,92 pp, data-driven tone normalization -2,49 pp, background suppression
-7,75 pp on PAD macro-F1), with the damage growing monotonically in how much of
the image was altered. Corbin & Marques (IEEE Access 11:78339, 2023) obtain
their gains by training on the transformed distribution, and their transform
moves **both** L* and b*, with b* carrying four times the displacement. The
L*-only version lands the image at an (L*, b*) pair no real skin tone occupies.

One transformed copy per image per seed, not one per training step. That is the
materialization model the frozen loop already expects: ``SimSiamPairDataset``
reads ``(original, transformed)`` and takes one view from each. ``ita_delta`` is
therefore drawn with ``step=0`` and the image digest, so the draw is
reproducible from the manifest alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dermssl.pretrain import config as config_mod
from dermssl.pretrain.tone_transforms import (
    MANIFEST_FILENAME,
    _load_mask,
    _load_records,
    _paths_overlap,
    _sha256_file,
)
from dermssl.tone_augment_lab import (
    B_PER_ITA_DEGREE,
    DEFAULT_FEATHER_SIGMA,
    L_PER_ITA_DEGREE,
    MAX_ABS_ITA_DELTA,
    augment_skin_tone_from_mask_record,
    sample_ita_delta,
    summarize_augmentation_records,
)

TRANSFORM_NAME = "lab_l_b_feathered_v1"
# The materialization draws once per image, so there is no training step to
# fold into the hash. Fixed here rather than passed in, so two callers cannot
# disagree about it and silently produce different pairs from the same seed.
MATERIALIZATION_STEP = 0


def _write(pixels: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8)).save(destination)


def _row(relative: Path, source_sha256: str, output_root: Path, record: dict, effect: dict):
    """One manifest row. Shared by the transformed and the identity paths so the
    two cannot drift into reporting different fields."""
    return {
        "relative_path": relative.as_posix(),
        "source_sha256": source_sha256,
        "transformed_sha256": _sha256_file(output_root / relative),
        "mask_sha256": record["mask_sha256"],
        "transform": TRANSFORM_NAME,
        "ita_delta": effect["ita_delta_applied"],
        "ita_delta_capped": effect["ita_delta_capped"],
        "applied": effect["applied"],
        "fallback_reason": effect["fallback_reason"],
        "delta_l_requested": effect["delta_l_requested"],
        "delta_l_realized": effect["delta_l_realized"],
        "delta_b_requested": effect["delta_b_requested"],
        "delta_b_realized": effect["delta_b_realized"],
        "gamut_clipped_pixels": effect["gamut_clipped_pixels"],
        "b_sign_flipped_pixels": effect["b_sign_flipped_pixels"],
    }


def _one(
    pixels: np.ndarray,
    lesion_mask: np.ndarray,
    record: dict,
    destination: Path,
    *,
    seed: int,
    source_sha256: str,
    feather_sigma: float,
    max_abs_ita_delta: float,
) -> dict:
    """Write one output image and return its effect record.

    Shared by both entry points so the two cannot drift apart in what they do
    to an image or in what they report about it.
    """
    if not record.get("eligible"):
        # Identity, written and counted - not an exception. The canonical
        # generator raises here, which is right for a cohort where every mask
        # is eligible; the SSL pool is not that cohort. Measured on 2026-11-03:
        # 421 of 33,620 pool masks abstain, all for degenerate area, so raising
        # would abort materialization on the first one.
        #
        # An unusable mask means the lesion cannot be protected, so the image
        # must not be transformed. It also must not vanish: dropping it would
        # make the treated arm train on a different image set than its control,
        # which is a second factor. Writing the original back gives that frame
        # two views of the same unmodified image - exactly what the control arm
        # does for every frame - and keeps the row.
        #
        # The reason is carried through from the record instead of being
        # hardcoded. A pool mask that collapsed to 0.3% of the frame and an
        # image that has no mask at all both land here, and both must be
        # declared with the delta, but they are not the same finding: the first
        # is a failure of our own semi-supervised teacher 026, which we can
        # measure and improve, and the second is an image the teacher was never
        # run on. Collapsing them under one label would make the dilution
        # impossible to attribute, which is the mistake this whole contrast
        # exists to avoid.
        _write(pixels, destination)
        return {
            "applied": False,
            "fallback_reason": record.get("quarantine_reason") or "ineligible_mask",
            "ita_delta_applied": 0.0,
            "ita_delta_capped": False,
            "delta_l_requested": 0.0,
            "delta_l_realized": 0.0,
            "delta_b_requested": 0.0,
            "delta_b_realized": 0.0,
            "gamut_clipped_pixels": 0,
            "b_sign_flipped_pixels": 0,
        }

    # Drawn from the image digest, so the manifest alone reproduces it and no
    # mutable RNG is shared with the loop's sampler. The control arm must see
    # the identical row order and batch boundaries; a shared RNG would make the
    # two arms differ in two factors instead of one.
    ita_delta = sample_ita_delta(
        seed, MATERIALIZATION_STEP, source_sha256, max_abs=max_abs_ita_delta
    )
    transformed, effect = augment_skin_tone_from_mask_record(
        pixels,
        lesion_mask,
        record,
        ita_delta=ita_delta,
        feather_sigma=feather_sigma,
        max_abs_ita_delta=max_abs_ita_delta,
    )
    _write(transformed, destination)
    return effect


def _finish(
    rows: list[dict],
    effects: list[dict],
    output_root: Path,
    seed: int,
    source_revision: str,
    feather_sigma: float,
    max_abs_ita_delta: float,
) -> dict:
    """Build, hash and write the manifest. Shared by both entry points."""
    summary = summarize_augmentation_records(effects)
    payload = {
        "schema_version": "1.0",
        "source_revision": source_revision,
        "transform": TRANSFORM_NAME,
        "seed": seed,
        "materialization_step": MATERIALIZATION_STEP,
        "parameters": {
            "b_per_ita_degree": B_PER_ITA_DEGREE,
            "l_per_ita_degree": L_PER_ITA_DEGREE,
            "max_abs_ita_delta": float(max_abs_ita_delta),
            "feather_sigma": float(feather_sigma),
        },
        # "rejected" is kept only because the canonical generator's manifest
        # has it and the two should be comparable at a glance. It is
        # STRUCTURALLY zero here and can never be anything else: an unusable
        # mask yields a counted identity copy and keeps its row, so nothing is
        # ever dropped. A field that cannot vary carries no information and can
        # be misread as "a rejection check ran and found none", so it is
        # flagged rather than left to be interpreted.
        #
        # The quantities that do vary, and that a reader actually needs, are
        # transformed_fraction and fallbacks_by_reason.
        "summary": {
            "total": len(rows),
            "transformed": summary["transformed"],
            "rejected": 0,
            "rejected_is_structurally_zero": True,
            "rejected_note": (
                "Nada se descarta nunca: una mascara inutilizable produce una copia "
                "identidad contada y conserva su fila, porque dejarla caer haria que el "
                "brazo tratado entrene sobre un conjunto distinto al de su control. "
                "Leer 'rejected: 0' como ausencia de problemas seria un error; mirar "
                "transformed_fraction y fallbacks_by_reason."
            ),
            **summary,
        },
        "transforms": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["revision"] = hashlib.sha256(canonical).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / MANIFEST_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def generate_tone_augmentations_from_records(
    *,
    image_root: str | Path,
    records_path: str | Path,
    records_root: str | Path,
    output_root: str | Path,
    seed: int,
    source_revision: str,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
    max_abs_ita_delta: float = MAX_ABS_ITA_DELTA,
) -> dict:
    """Same product as :func:`generate_tone_augmentations`, driven by a JSONL cache.

    The sibling entry point inherits two conventions from the canonical
    generator that the SSL pool cache does not follow, and both were found by
    checking rather than by a failed run:

    * ``_load_records`` expects one JSON object with a ``masks`` list. The pool
      cache is JSONL, one record per line.
    * ``_load_mask`` rebuilds the mask location by mirroring the image's
      subdirectories under ``mask_root``. The pool masks are flat, and each
      record already carries its own ``mask_path``.

    So this path reads the record's declared ``mask_path`` instead of
    reconstructing one. That is also the more defensible source: the recorded
    path is the mapping the mask generator actually wrote and hashed, while a
    reconstructed path is a guess that happens to agree.

    ``image_root`` still governs the OUTPUT layout, because
    ``config.collect_paired_images`` pairs original and transformed by the same
    relative path under two roots. The pool spans two cohorts
    (``datos/isic2019`` and ``datos/isic2020``) whose common parent is
    ``datos``, so that pairing holds.
    """
    image_root = Path(image_root)
    records_root = Path(records_root)
    output_root = Path(output_root)
    if _paths_overlap(image_root, output_root):
        raise ValueError(
            "output_root debe ser una carpeta separada y no superponerse con image_root"
        )

    rows: list[dict] = []
    effects: list[dict] = []
    seen: set[str] = set()

    for line in Path(records_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("source_revision") != source_revision:
            continue
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("cada máscara debe declarar image_id")
        if image_id in seen:
            raise ValueError(f"image_id duplicado en el cache: {image_id}")
        seen.add(image_id)

        original = Path(record["source_image_path"])
        relative = original.relative_to(image_root)
        lesion_mask = _lesion_mask_if_used(record, records_root, image_id)

        with Image.open(original) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        source_sha256 = _sha256_file(original)

        effect = _one(
            pixels,
            lesion_mask,
            record,
            output_root / relative,
            seed=seed,
            source_sha256=source_sha256,
            feather_sigma=feather_sigma,
            max_abs_ita_delta=max_abs_ita_delta,
        )
        effects.append(effect)
        rows.append(_row(relative, source_sha256, output_root, record, effect))

    if not rows:
        raise ValueError("el cache no contiene máscaras de la revisión solicitada")
    return _finish(rows, effects, output_root, seed, source_revision,
                   feather_sigma, max_abs_ita_delta)


def _lesion_mask_if_used(
    record: dict, records_root: Path, image_id: str
) -> np.ndarray | None:
    """Load the lesion mask, or return ``None`` when the record will not use it.

    An ineligible record takes the identity branch of ``_one``, which never
    touches the mask. Demanding the file anyway made it impossible to carry a
    frame that has no mask at all - which is the case for the 1,134 clinical
    photographs of the ``pad_adapt`` role: the dermoscopy teacher 026 was never
    run on them, so there is nothing on disk to load.

    Those frames still have to be in the treated arm. The control draws 25% of
    every batch from the clinical role, so leaving them out would make the two
    arms train on different data - the second factor this whole contrast exists
    to exclude. The alternative, running a dermoscopy segmenter over smartphone
    clinical photographs, would invent out-of-domain masks of unknown quality
    inside the treated arm, which is worse than declaring an untransformed
    frame.

    For an eligible record the file is still mandatory and still verified: that
    is the case where the mask decides which pixels are protected.
    """
    if not record.get("eligible"):
        return None
    mask_path = record.get("mask_path")
    if not mask_path:
        raise ValueError(f"la mascara elegible de {image_id} no declara mask_path")
    mask_file = records_root / mask_path
    if not mask_file.is_file():
        raise FileNotFoundError(f"falta la máscara declarada para {image_id}: {mask_file}")
    return np.asarray(Image.open(mask_file).convert("L"), dtype=np.uint8) > 0


def generate_tone_augmentations(
    *,
    image_root: str | Path,
    mask_root: str | Path,
    mask_manifest_path: str | Path,
    output_root: str | Path,
    seed: int,
    source_revision: str,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
    max_abs_ita_delta: float = MAX_ABS_ITA_DELTA,
) -> dict:
    """Write one augmented copy per image and return its auditable manifest.

    Every row carries the realized effect in both channels, not just the
    request. A null training result cannot be read without knowing how much of
    the displacement survived the gamut, how often b* crossed the axis, and how
    many images fell back to identity because their mask was unusable.
    """
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    output_root = Path(output_root)
    if _paths_overlap(image_root, output_root):
        raise ValueError(
            "output_root debe ser una carpeta separada y no superponerse con image_root"
        )

    records = _load_records(Path(mask_manifest_path), source_revision)
    rows: list[dict] = []
    effects: list[dict] = []

    for original in config_mod.collect_images(image_root):
        relative = original.relative_to(image_root)
        record = records.get(original.stem)
        if record is None:
            raise ValueError(f"no hay registro de máscara para {relative.as_posix()}")

        # Same asymmetry as in the records entry point: an ineligible record
        # takes the identity branch and never reads the mask, so requiring the
        # file would block frames that have none. See _lesion_mask_if_used.
        lesion_mask = _load_mask(mask_root, relative) if record.get("eligible") else None
        with Image.open(original) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0

        source_sha256 = _sha256_file(original)
        effect = _one(
            pixels,
            lesion_mask,
            record,
            output_root / relative,
            seed=seed,
            source_sha256=source_sha256,
            feather_sigma=feather_sigma,
            max_abs_ita_delta=max_abs_ita_delta,
        )
        effects.append(effect)
        rows.append(_row(relative, source_sha256, output_root, record, effect))

    return _finish(rows, effects, output_root, seed, source_revision,
                   feather_sigma, max_abs_ita_delta)
