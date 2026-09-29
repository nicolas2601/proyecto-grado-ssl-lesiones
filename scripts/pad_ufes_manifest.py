"""Build the PAD-UFES-20 manifest the zero-shot evaluation reads (docs/30 2.1).

Runs after ``pad_ufes_download.py --extract``, on the pod. It checks, before
any feature is extracted: the four official files against Mendeley, the
published counts (2298 images, 1373 patients, 1641 lesions, per diagnostic),
that every ``img_id`` resolves to exactly one PNG that decodes, and records
the sha256 of every image. Any mismatch aborts; nothing is written.

    python scripts/pad_ufes_manifest.py --root ~/work/datos/pad_ufes_20 \
        --out ~/work/datos/pad_ufes_20/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pad_ufes_download import IMAGES_DIR  # noqa: E402

from dermssl import pad_ufes as pu  # noqa: E402
from dermssl import pad_ufes_files as pf  # noqa: E402

MANIFEST_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the --dest of pad_ufes_download.py")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def verify_official(root: Path) -> dict[str, str]:
    for official in pf.OFFICIAL_FILES:
        pf.verify(root / official.name, official)
    return {f.name: f.sha256 for f in pf.OFFICIAL_FILES}


def check_decodes(path: Path) -> None:
    """The bank loader swaps an unreadable image for a black one; refuse it here instead."""
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a corrupt file
        raise pf.IntegrityError(f"{path.name} no decodifica: {exc}") from exc


def image_rows(records: list[pu.PadRecord], paths: dict[str, Path], root: Path) -> list[dict]:
    rows = []
    for record in records:
        path = paths[record.img_id]
        check_decodes(path)
        rows.append({
            "img_id": record.img_id, "patient_id": record.patient_id, "lesion_id": record.lesion_id,
            "diagnostic": record.diagnostic, "ham_label": record.ham_label,
            "fst": record.fst, "fst_status": record.fst_status, "fst_bin": record.fst_bin,
            "path": str(path.relative_to(root)), "sha256": pf.sha256_of(path),
        })
    return rows


def build_manifest(root: Path) -> dict:
    archives = verify_official(root)
    records = pu.parse_metadata(root / pf.METADATA_NAME)
    pu.check_published_counts(records)
    index = pf.index_images(root / IMAGES_DIR)
    paths = pf.resolve_images([r.img_id for r in records], index)
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_doi": pf.DATASET_DOI,
        "license": pf.LICENSE,
        "official_files_sha256": archives,
        "n_images": len(records),
        "n_patients": len({r.patient_id for r in records}),
        "n_lesions": len({r.lesion_id for r in records}),
        "fst": pu.fst_summary(records),
        "images": image_rows(records, paths, root),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.expanduser()
    if out.exists():
        print(f"ABORT: {out} ya existe; el manifiesto no se reescribe", file=sys.stderr)
        return 1
    try:
        manifest = build_manifest(args.root.expanduser())
    except (pf.IntegrityError, pu.PadMetadataError) as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{manifest['n_images']} imagenes, FST {manifest['fst']['by_bin']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
