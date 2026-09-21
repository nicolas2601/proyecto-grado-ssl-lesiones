"""Convierte un mapa de saliencia en pseudo-mascaras con el modulo de incertidumbre.

Entrada: `saliency.npy` (N, H, W) float. Salida: `pseudo_masks.npy` (N, H, W)
uint8 con 0 = fondo, 1 = lesion, 255 = ignorar, y `pseudo_masks_stats.json`.

* `--source ccam`: mapas crudos de CCAM. Se normalizan por imagen y se orienta
  el signo (borde frente a centro) antes de aplicar los umbrales.
* `--source student`: probabilidades de un estudiante (rondas de re-etiquetado).
  Se usan tal cual, sin normalizar ni orientar.

Umbrales de USL-Net: menor a 0,35 es fondo, mayor a 0,65 es candidato a lesion,
lo intermedio se ignora. Con `--field` los candidatos se limitan al campo del
dermatoscopio, lo de afuera queda como fondo y una region que cubre mas de
`--max-area-fraction` del campo descarta la imagen.

Con `--images` (el `images.npy` de la cache) se detectan artefactos por color
(tinta quirurgica violeta y brillos especulares), que dejan de ser candidatos y
quedan marcados como ignorar. `--no-artifact-suppression` lo desactiva. Solo usa
colores, nunca mascaras ni etiquetas.

    python make_pseudo_masks.py --saliency salidas/ccam/saliency.npy \\
        --output-dir salidas/pseudo_r0 --source ccam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cache_io  # noqa: E402
import uncertainty_module as um  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

MASKS_FILE = "pseudo_masks.npy"
STATS_FILE = "pseudo_masks_stats.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--saliency", required=True, help="Archivo .npy (N, H, W) con la saliencia")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida")
    parser.add_argument("--source", default="ccam", choices=("ccam", "student"), help="Origen de la saliencia")
    parser.add_argument("--low", type=float, default=um.DEFAULT_LOW, help="Umbral inferior (fondo)")
    parser.add_argument("--high", type=float, default=um.DEFAULT_HIGH, help="Umbral superior (lesion)")
    parser.add_argument("--min-area-fraction", type=float, default=um.DEFAULT_MIN_AREA_FRACTION, help="Area minima de una region, como fraccion de la imagen")
    parser.add_argument("--max-center-distance", type=float, default=um.DEFAULT_MAX_CENTER_DISTANCE, help="Distancia maxima al centro (fraccion de la semidiagonal)")
    parser.add_argument("--field", default=None, help="Archivo field.npy (N, H, W) con el campo valido del dermatoscopio")
    parser.add_argument("--max-area-fraction", type=float, default=um.DEFAULT_MAX_AREA_FRACTION, help="Area maxima de la region ganadora, como fraccion del campo")
    parser.add_argument("--images", default=None, help="Archivo images.npy (N, H, W, 3) de la cache, para detectar artefactos por color")
    parser.add_argument("--no-artifact-suppression", dest="artifact_suppression", action="store_false", help="No suprimir tinta ni brillos aunque se pase --images")
    parser.set_defaults(artifact_suppression=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if not 0.0 <= args.low < args.high <= 1.0:
        sys.exit(f"error: se requiere 0 <= --low < --high <= 1 (recibido {args.low}, {args.high})")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saliency = cache_io.open_cache(Path(args.saliency))
    fields = _matching_cache(args.field, saliency, "--field")
    images = None
    if args.artifact_suppression:
        images = _matching_cache(args.images, saliency, "--images")
    masks = cache_io.create_cache(output_dir / MASKS_FILE, saliency.shape, "uint8")
    from_ccam = args.source == "ccam"
    stats = um.saliency_array_to_pseudo_masks(
        saliency,
        masks,
        orient=from_ccam,
        normalize=from_ccam,
        fields=fields,
        images=images,
        low=args.low,
        high=args.high,
        min_area_fraction=args.min_area_fraction,
        max_center_distance=args.max_center_distance,
        max_area_fraction=args.max_area_fraction,
    )
    masks.flush()

    stats.update(_provenance(args))
    (output_dir / STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def _matching_cache(path: str | None, saliency, flag: str):
    """Open a per-image cache and check it has one entry per saliency map."""
    if path is None:
        return None
    cache = cache_io.open_cache(Path(path))
    if cache.shape[:3] != saliency.shape[:3]:
        sys.exit(f"error: {flag} {cache.shape} no coincide con la saliencia {saliency.shape}")
    return cache


def _provenance(args: argparse.Namespace) -> dict:
    suppressed = bool(args.artifact_suppression and args.images is not None)
    return {
        "source": args.source,
        "images": str(args.images) if suppressed else None,
        "artifact_suppression": suppressed,
        "low": args.low,
        "high": args.high,
        "min_area_fraction": args.min_area_fraction,
        "max_center_distance": args.max_center_distance,
        "max_area_fraction": args.max_area_fraction,
        "saliency": str(args.saliency),
        "field": None if args.field is None else str(args.field),
        "reads_masks_or_labels": False,
    }


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
