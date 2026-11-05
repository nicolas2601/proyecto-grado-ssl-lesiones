"""Build the DDI manifest the zero-shot evaluation will read (plan: Fase 0).

Reads the zip Pau already downloaded from the Stanford AIMI portal (Research Use
Agreement already accepted) and extracts it to ``--dest`` if not already there. It
checks, before any feature is extracted: the zip's own sha256 against the pinned
value, the published counts (656 images, skin_tone x malignant distribution per
``dermssl.ddi.EXPECTED_SKIN_TONE_COUNTS``), every image decodes, and records the
sha256 of every image. Any mismatch aborts; nothing is written. No images are
committed to this repository - the manifest holds hashes and metadata only.

    python scripts/ddi_manifest.py \
        --zip ~/Downloads/ddidiversedermatologyimages.zip \
        --dest ~/work/datos/ddi \
        --out ~/work/datos/ddi/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import ddi  # noqa: E402

MANIFEST_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True, help="where images + metadata are extracted")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_zip(zip_path: Path) -> None:
    digest = _sha256_file(zip_path)
    if digest != ddi.ZIP_SHA256:
        raise ddi.DdiMetadataError(
            f"{zip_path.name}: sha256 {digest[:16]}..., esperado {ddi.ZIP_SHA256[:16]}..."
        )


def extract(zip_path: Path, dest: Path) -> None:
    """Extract refusing any member that would land outside ``dest`` (zip-slip guard)."""
    root = dest.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise ddi.DdiMetadataError(f"{zip_path.name}: {member!r} sale del directorio destino")
        zf.extractall(root)


def check_decodes(path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a corrupt file
        raise ddi.DdiMetadataError(f"{path.name} no decodifica: {exc}") from exc


def image_rows(records: list[ddi.DdiRecord], root: Path) -> list[dict]:
    rows = []
    for record in records:
        path = root / record.file_name
        if not path.is_file():
            raise ddi.DdiMetadataError(f"{record.file_name} del metadata no existe en {root}")
        check_decodes(path)
        rows.append({
            "ddi_id": record.ddi_id, "file_name": record.file_name,
            "skin_tone": record.skin_tone, "skin_tone_label": record.skin_tone_label,
            "malignant": record.malignant, "disease": record.disease,
            "ham_label": record.ham_label, "sha256": _sha256_file(path),
        })
    return rows


def build_manifest(zip_path: Path, dest: Path) -> dict:
    verify_zip(zip_path)
    if not dest.exists() or not any(dest.iterdir()):
        extract(zip_path, dest)
    records = ddi.parse_metadata(dest / "ddi_metadata.csv")
    ddi.check_published_counts(records)
    coverage = ddi.ham_mapping_coverage(records)
    return {
        "manifest_version": MANIFEST_VERSION,
        "source_url": ddi.SOURCE_URL,
        "license": ddi.LICENSE,
        "zip_sha256": ddi.ZIP_SHA256,
        "n_images": len(records),
        "skin_tone": ddi.skin_tone_summary(records),
        "ham_mapping_coverage": coverage,
        "images": image_rows(records, dest),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.expanduser()
    if out.exists():
        print(f"ABORT: {out} ya existe; el manifiesto no se reescribe", file=sys.stderr)
        return 1
    try:
        manifest = build_manifest(args.zip.expanduser(), args.dest.expanduser())
    except ddi.DdiMetadataError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{manifest['n_images']} imagenes, mapeo HAM {manifest['ham_mapping_coverage']['n_mapped']}"
          f"/{manifest['n_images']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
