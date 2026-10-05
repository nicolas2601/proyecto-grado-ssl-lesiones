"""Hoja de contacto del campo valido del dermatoscopio (auditoria visual).

El detector de campo decide, imagen por imagen, si recorta un circulo o si
conserva el cuadro completo. Esa decision es geometrica y automatica, asi que
la unica forma honesta de confiar en ella es mirarla: este script toma una
muestra de cada grupo (dataset x metodo) y dibuja el borde del campo en rojo
sobre la imagen cacheada.

Lee unicamente la cache (``images.npy``, ``field.npy``, ``manifest.csv``):
nunca abre archivos de origen ni ninguna anotacion. Escribe el PNG y, al lado,
un JSON con los identificadores muestreados para poder repetir la revision.

Ejemplo:
    python plan-segmentacion/comun/datos/field_contact_sheet.py \\
        --cache-dir salidas/cache224 --output salidas/cache224/campo.png \\
        --per-group 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cache_io  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

# Above this share of valid pixels the detector kept the whole frame.
FULL_FRAME_MIN_FRACTION = 0.999
METHOD_FULL_FRAME = "full_frame"
METHOD_CIRCULAR = "circular"
CAPTION_HEIGHT = 14
BOUNDARY_COLOR = (255, 0, 0)
BOUNDARY_WIDTH = 2


def method_of(valid_fraction: float) -> str:
    """Name the decision the field detector took for one image."""
    return METHOD_FULL_FRAME if valid_fraction >= FULL_FRAME_MIN_FRACTION else METHOD_CIRCULAR


def caption(entry: Mapping) -> str:
    """One line under which a reviewer can look the image up again."""
    return (
        f"{entry['dataset']} {entry['image_id']} "
        f"{entry['method']} valid={float(entry['valid_fraction']):.3f}"
    )


def _erode(valid: np.ndarray) -> np.ndarray:
    """Four-connected erosion; pixels outside the image count as invalid."""
    padded = np.pad(np.asarray(valid, dtype=bool), 1, constant_values=False)
    eroded = padded[1:-1, 1:-1].copy()
    for neighbour in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
        eroded &= neighbour
    return eroded


def boundary(valid: np.ndarray, width: int = BOUNDARY_WIDTH) -> np.ndarray:
    """Inner border of the valid region, ``width`` pixels thick.

    Inner on purpose: the ring always lands on pixels the detector kept, so a
    full frame still shows its border instead of drawing nothing.
    """
    valid = np.asarray(valid, dtype=bool)
    shrunk = valid
    for _ in range(max(1, width)):
        shrunk = _erode(shrunk)
    return valid & ~shrunk


def paint_boundary(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Copy of the cached image with the field border in red."""
    painted = np.array(image, dtype=np.uint8, copy=True)
    painted[boundary(valid)] = BOUNDARY_COLOR
    return painted


def group_entries(rows: Sequence[Mapping], field: np.ndarray) -> dict[str, list[dict]]:
    """One bucket per ``<dataset>/<method>``, in manifest order."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        index = int(row["index"])
        valid_fraction = float(np.asarray(field[index]).mean())
        entry = {
            "index": index,
            "dataset": row["dataset"],
            "image_id": row["image_id"],
            "method": method_of(valid_fraction),
            "valid_fraction": valid_fraction,
        }
        groups.setdefault(f"{entry['dataset']}/{entry['method']}", []).append(entry)
    return groups


def sample_groups(groups: Mapping[str, list[dict]], per_group: int, seed: int) -> dict:
    """At most ``per_group`` entries per bucket, drawn with a seeded sampler."""
    sampler = random.Random(seed)
    sampled = {}
    for key in sorted(groups):
        entries = groups[key]
        chosen = sampler.sample(entries, min(per_group, len(entries)))
        sampled[key] = sorted(chosen, key=lambda entry: entry["image_id"])
    return sampled


def _draw_row(canvas, draw, images, field, entries: Sequence[Mapping], top: int, cell: int) -> None:
    from PIL import Image

    for column, entry in enumerate(entries):
        left = column * cell
        draw.text((left + 2, top + 2), caption(entry), fill=(255, 255, 255))
        painted = paint_boundary(images[entry["index"]], field[entry["index"]])
        canvas.paste(Image.fromarray(painted), (left, top + CAPTION_HEIGHT))


def render_sheet(path: Path, images, field, sampled: Mapping[str, list[dict]], per_group: int):
    """One row per group, ``per_group`` columns wide; empty cells stay black."""
    from PIL import Image, ImageDraw

    cell = int(np.asarray(images[0]).shape[0])
    row_height = cell + CAPTION_HEIGHT
    canvas = Image.new("RGB", (per_group * cell, len(sampled) * row_height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, key in enumerate(sorted(sampled)):
        _draw_row(canvas, draw, images, field, sampled[key], index * row_height, cell)
    canvas.save(Path(path))
    return cell


def build_sheet(args: argparse.Namespace) -> dict:
    """Write the PNG and its JSON index; returns the JSON payload."""
    cache_dir = Path(args.cache_dir)
    rows = cache_io.read_manifest(cache_dir / "manifest.csv")
    if not rows:
        raise ValueError(f"la cache esta sin imagenes: {cache_dir}")
    images = cache_io.open_cache(cache_dir / "images.npy")
    field = cache_io.open_cache(cache_dir / "field.npy")
    groups = group_entries(rows, field)
    sampled = sample_groups(groups, args.per_group, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cell = render_sheet(output, images, field, sampled, args.per_group)
    payload = {
        "cache_dir": str(cache_dir),
        "output": str(output),
        "per_group": int(args.per_group),
        "seed": int(args.seed),
        "resolution": cell,
        "groups": {
            key: {"n": len(groups[key]), "sampled": [entry["image_id"] for entry in entries]}
            for key, entries in sampled.items()
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Carpeta de la cache (images.npy, field.npy, manifest.csv)")
    parser.add_argument("--output", required=True, help="Archivo PNG de salida; el JSON se escribe al lado")
    parser.add_argument("--per-group", type=int, default=8, help="Imagenes por grupo (dataset x metodo)")
    parser.add_argument("--seed", type=int, default=0, help="Semilla del muestreo")
    return parser.parse_args(argv)


def main() -> None:
    payload = build_sheet(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
