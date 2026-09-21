"""Genera imágenes de tono CIRCLe sin modificar las originales.

Ejemplo:
    python scripts/generate_tone_transforms.py \\
        --image-root /datos/train --mask-root /datos/masks \\
        --mask-manifest /datos/mask_manifest.json \\
        --output-root /datos/train-tone-v1 --source-revision pool-v1 \\
        --shift-buckets 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl.pretrain.tone_transforms import generate_tone_transformations  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--image-root", required=True, help="Carpeta de imágenes originales")
    parser.add_argument("--mask-root", required=True, help="Carpeta de máscaras PNG espejo")
    parser.add_argument("--mask-manifest", required=True, help="Ruta a mask_manifest.json")
    parser.add_argument("--output-root", required=True, help="Carpeta nueva para imágenes transformadas")
    parser.add_argument("--source-revision", required=True, help="Revisión de datos que debe coincidir con las máscaras")
    parser.add_argument("--shift-buckets", required=True, type=int, help="Desplazamiento FST acotado a [-2, 2]")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = generate_tone_transformations(
        image_root=args.image_root,
        mask_root=args.mask_root,
        mask_manifest_path=args.mask_manifest,
        output_root=args.output_root,
        shift_buckets=args.shift_buckets,
        source_revision=args.source_revision,
    )
    print(
        "Transformaciones generadas: "
        f"{result['summary']['transformed']} | revisión {result['revision']}"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
