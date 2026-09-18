"""Generación reproducible de pares de tono protegidos por máscara.

La transformación usa exclusivamente ``dermssl.circle.shift_skin_tone``: el
contrato conserva la lesión píxel a píxel y rechaza máscaras no elegibles. Los
archivos se escriben bajo una raíz independiente con la misma ruta relativa,
para que ``--transformed-image-root`` pueda emparejarlos sin modificar los
originales.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dermssl import circle
from dermssl.pretrain import config as config_mod

MANIFEST_FILENAME = "tone_transform_manifest.json"
TRANSFORM_NAME = "circle_lab_l_star_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(path: Path, source_revision: str) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("masks")
    if not isinstance(records, list):
        raise ValueError("mask_manifest debe contener una lista masks")
    by_image_id: dict[str, dict] = {}
    for raw_record in records:
        record = dict(raw_record)
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("cada máscara debe declarar image_id")
        if record.get("source_revision") != source_revision:
            continue
        if image_id in by_image_id:
            raise ValueError(f"image_id duplicado en mask_manifest: {image_id}")
        by_image_id[image_id] = record
    if not by_image_id:
        raise ValueError("mask_manifest no contiene máscaras de la revisión solicitada")
    return by_image_id


def _load_mask(mask_root: Path, relative_image_path: Path) -> np.ndarray:
    mask_path = mask_root / relative_image_path.with_suffix(".png")
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"falta la máscara PNG para {relative_image_path.as_posix()}"
        )
    return np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0


def _paths_overlap(left: Path, right: Path) -> bool:
    """Indica si dos raíces se contienen, incluso si una aún no existe."""
    left, right = left.resolve(), right.resolve()
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def generate_tone_transformations(
    *,
    image_root: str | Path,
    mask_root: str | Path,
    mask_manifest_path: str | Path,
    output_root: str | Path,
    shift_buckets: int,
    source_revision: str,
) -> dict:
    """Materializa pares CIRCLe y devuelve su manifiesto JSON auditable."""
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    output_root = Path(output_root)
    if _paths_overlap(image_root, output_root):
        raise ValueError(
            "output_root debe ser una carpeta separada y no superponerse con image_root"
        )
    records = _load_records(Path(mask_manifest_path), source_revision)
    rows = []
    for original in config_mod.collect_images(image_root):
        relative = original.relative_to(image_root)
        record = records.get(original.stem)
        if record is None:
            raise ValueError(f"no hay registro de máscara para {relative.as_posix()}")
        if not record.get("eligible"):
            raise ValueError(f"la máscara de {relative.as_posix()} no es elegible")
        lesion_mask = _load_mask(mask_root, relative)
        with Image.open(original) as image:
            rgb = image.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.float64) / 255.0
        transformed = circle.shift_skin_tone_from_mask_record(
            pixels, lesion_mask, record, shift_buckets
        )
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(transformed * 255.0).astype(np.uint8)).save(destination)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "source_sha256": _sha256_file(original),
                "transformed_sha256": _sha256_file(destination),
                "mask_sha256": record["mask_sha256"],
                "transform": TRANSFORM_NAME,
                "shift_buckets": circle.clamp_fst_shift(shift_buckets),
            }
        )
    payload = {
        "schema_version": "1.0",
        "source_revision": source_revision,
        "transform": TRANSFORM_NAME,
        "summary": {"total": len(rows), "transformed": len(rows), "rejected": 0},
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
