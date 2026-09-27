"""Preprocesa el pool sin etiquetas UNA sola vez a una cache uint8 (memmap).

Lee solo imagenes: nunca abre mascaras, diagnosticos ni etiquetas. Escribe:

    images.npy       (N, R, R, 3) uint8, memmap
    field.npy        (N, R, R) uint8 {0, 1}, campo valido del dermatoscopio
    manifest.csv     index, dataset, image_id, path, split (train / val)
    cache_info.json  resolucion, conteos, particion y semilla
    field_stats.json umbrales del campo y fracciones medias

La particion `val` (~2 % del pool) es la pseudo-validacion: sirve para
seleccionar el checkpoint del estudiante con la perdida sobre pseudo-etiquetas,
nunca con mascaras reales.

Ejemplo:
    python plan-segmentacion/comun/datos/prepare_cache.py \\
        --root ham10000=/datos/ham10000 --root isic2019=/datos/isic2019 \\
        --root isic2020=/datos/isic2020 --output-dir salidas/cache224
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cache_io  # noqa: E402
import field_roi  # noqa: E402
import hair_removal  # noqa: E402
import synthetic  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

SYNTHETIC_DATASET = "synthetic"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--root", action="append", default=[], metavar="NOMBRE=RUTA", help="Raiz de imagenes sin etiquetas; se repite por dataset")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida de la cache")
    parser.add_argument("--resolution", type=int, default=224, help="Lado del cuadrado final en pixeles")
    parser.add_argument("--holdout-fraction", type=float, default=0.02, help="Fraccion del pool reservada como pseudo-validacion")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la particion")
    parser.add_argument("--threads", type=int, default=8, help="Hilos de decodificacion (sin procesos ni DataLoader)")
    parser.add_argument("--limit", type=int, default=0, help="Usa solo las primeras N imagenes (0 = todas)")
    parser.add_argument("--no-dedupe", action="store_true", help="Conserva ids repetidos entre datasets")
    parser.add_argument("--synthetic", type=int, default=0, help="Genera N imagenes sinteticas (smoke test)")
    parser.add_argument("--field-black-threshold", type=float, default=field_roi.DEFAULT_BLACK_THRESHOLD, help="Umbral de luminancia (fraccion de 255) del campo del dermatoscopio")
    parser.add_argument("--field-margin-fraction", type=float, default=field_roi.DEFAULT_MARGIN_FRACTION, help="Margen hacia adentro, como fraccion del radio ajustado")
    parser.add_argument("--hair-removal", action="store_true", help="Quita el pelo (black-hat + TELEA) antes de guardar cada imagen (S2)")
    parser.add_argument("--no-field", action="store_true", help="No calcula field.npy (solo para depuracion)")
    return parser.parse_args(argv)


def _resolve_roots(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    if args.synthetic > 0:
        source = output_dir / "synthetic_source"
        synthetic.write_pool_images(source, args.synthetic, args.resolution, args.seed)
        return {SYNTHETIC_DATASET: source}
    if not args.root:
        sys.exit("error: indique al menos un --root NOMBRE=RUTA o use --synthetic N")
    return dict(cache_io.parse_root_spec(spec) for spec in args.root)


def _fill_cache(records, cache, resolution: int, threads: int, hair: bool = False) -> list[float]:
    fractions = [0.0] * len(records)

    def load(index: int) -> None:
        image = cache_io.load_resized_rgb(records[index].path, resolution)
        if hair:
            image, fractions[index] = hair_removal.remove_hair(image)
        cache[index] = image

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for count, _ in enumerate(pool.map(load, range(len(records))), start=1):
            if count % 2000 == 0:
                print(f"cache: {count}/{len(records)} imagenes", flush=True)
    cache.flush()
    return fractions


def _write_field(args: argparse.Namespace, output_dir: Path, count: int) -> dict | None:
    if args.no_field:
        return None
    images = cache_io.open_cache(output_dir / "images.npy")
    field = cache_io.create_cache(
        output_dir / "field.npy", (count, args.resolution, args.resolution), "uint8"
    )
    stats = field_roi.compute_field_cache(
        images,
        field,
        black_threshold=args.field_black_threshold,
        margin_fraction=args.field_margin_fraction,
    )
    field.flush()
    (output_dir / "field_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def _hair_info(fractions: list[float]) -> dict:
    return {
        "method": "black-hat + TELEA (docs/26 3.4)",
        "kernel": hair_removal.KERNEL,
        "threshold": hair_removal.THRESHOLD,
        "inpaint_radius": hair_removal.INPAINT_RADIUS,
        "mean_fraction": sum(fractions) / max(len(fractions), 1),
        "images_with_hair": sum(f > 0 for f in fractions),
    }


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan = cache_io.scan_image_roots(
        _resolve_roots(args, output_dir), dedupe_by_id=not args.no_dedupe
    )
    records = scan.records[: args.limit] if args.limit > 0 else scan.records
    if not records:
        sys.exit("error: no se encontraron imagenes en las raices indicadas")

    splits = cache_io.assign_splits(
        [f"{r.dataset}:{r.image_id}" for r in records], args.holdout_fraction, args.seed
    )
    cache = cache_io.create_cache(
        output_dir / "images.npy", (len(records), args.resolution, args.resolution, 3), "uint8"
    )
    fractions = _fill_cache(records, cache, args.resolution, args.threads, args.hair_removal)
    cache_io.write_manifest(output_dir / "manifest.csv", records, splits)

    info = {
        "count": len(records),
        "resolution": args.resolution,
        "per_dataset": dict(Counter(r.dataset for r in records)),
        "duplicates_dropped": scan.duplicates_dropped,
        "holdout_fraction": args.holdout_fraction,
        "val_count": splits.count(cache_io.SPLIT_VALIDATION),
        "seed": args.seed,
        "field": _write_field(args, output_dir, len(records)),
        "reads_masks_or_labels": False,
        "hair_removal": _hair_info(fractions) if args.hair_removal else None,
    }
    (output_dir / "cache_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    return info


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
