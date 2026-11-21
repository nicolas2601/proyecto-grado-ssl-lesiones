"""Side-by-side contours of the shipped segmenter (red) and the DPT candidate (green) on clinical photos.

  python clinical_overlays.py <out_png> <image> [<image> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from eval_pair import SHIPPED, dpt_session, logits_to_mask, predict_image, shipped_runner

TILE = 360
EDGE = {"shipped": (230, 40, 40), "dpt": (40, 220, 60)}


def edges(mask: np.ndarray) -> np.ndarray:
    inner = mask.copy()
    inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return mask & ~inner


def overlay(image: Image.Image, mask: np.ndarray, color) -> Image.Image:
    small = image.resize((TILE, round(TILE * image.height / image.width)), Image.Resampling.BILINEAR)
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(small.size, Image.Resampling.NEAREST)) > 127
    arr = np.asarray(small).copy()
    e = edges(m)
    for dy in (-1, 0, 1):
        arr[np.roll(e, dy, axis=0)] = color
    out = Image.fromarray(arr)
    ImageDraw.Draw(out).text((6, 6), f"area {m.mean():.2f}", fill=color)
    return out


def main(out_png: Path, paths: list[Path]) -> None:
    shipped = shipped_runner.load_session(SHIPPED / "segmenter.onnx")
    dpt = dpt_session(4)
    rows, stats = [], []
    for path in paths:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        old = logits_to_mask(shipped_runner.predict_logits(shipped, image), image.width, image.height)
        new = predict_image(dpt, image)[1]
        a, b = overlay(image, np.asarray(old, bool), EDGE["shipped"]), overlay(image, new, EDGE["dpt"])
        row = Image.new("RGB", (TILE * 2 + 8, max(a.height, b.height)), "white")
        row.paste(a, (0, 0))
        row.paste(b, (TILE + 8, 0))
        rows.append(row)
        stats.append({"image": path.name, "shipped_area": round(float(np.mean(old)), 3), "dpt_area": round(float(new.mean()), 3)})
    sheet = Image.new("RGB", (rows[0].width, sum(r.height + 8 for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + 8
    sheet.save(out_png)
    print(json.dumps(stats))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
