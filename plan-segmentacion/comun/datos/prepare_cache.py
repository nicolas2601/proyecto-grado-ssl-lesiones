"""Preprocesa el pool sin etiquetas UNA sola vez a una cache uint8 (memmap).

Lee solo imagenes: nunca abre mascaras, diagnosticos ni etiquetas. Escribe:

    images.npy       (N, R, R, 3) uint8, memmap
    manifest.csv     index, dataset, image_id, path, split (train / val)
    cache_info.json  resolucion, conteos, particion y semilla

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
    return parser.parse_args(argv)


def _resolve_roots(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    if args.synthetic > 0:
        source = output_dir / "synthetic_source"
        synthetic.write_pool_images(source, args.synthetic, args.resolution, args.seed)
        return {SYNTHETIC_DATASET: source}
    if not args.root:
        sys.exit("error: indique al menos un --root NOMBRE=RUTA o use --synthetic N")
    return dict(cache_io.parse_root_spec(spec) for spec in args.root)


def _fill_cache(records, cache, resolution: int, threads: int) -> None:
    def load(index: int) -> None:
        cache[index] = cache_io.load_resized_rgb(records[index].path, resolution)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for count, _ in enumerate(pool.map(load, range(len(records))), start=1):
            if count % 2000 == 0:
                print(f"cache: {count}/{len(records)} imagenes", flush=True)
    cache.flush()


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
    _fill_cache(records, cache, args.resolution, args.threads)
    cache_io.write_manifest(output_dir / "manifest.csv", records, splits)

    info = {
        "count": len(records),
        "resolution": args.resolution,
        "per_dataset": dict(Counter(r.dataset for r in records)),
        "duplicates_dropped": scan.duplicates_dropped,
        "holdout_fraction": args.holdout_fraction,
        "val_count": splits.count(cache_io.SPLIT_VALIDATION),
        "seed": args.seed,
        "reads_masks_or_labels": False,
    }
    (output_dir / "cache_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    return info


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
