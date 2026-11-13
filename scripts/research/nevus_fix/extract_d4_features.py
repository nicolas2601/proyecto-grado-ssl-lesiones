#!/usr/bin/env python3
"""Extract shipped-recipe CLS4-D4 features (448 px, 8 dihedral views) and both head probabilities.

Input rows: CSV or JSONL with at least ``image_id`` and ``image_path``. The classifier checkpoint is
pinned by SHA256. Output npz: image_ids, features (N, 3072, row-L2), ham7, pad6 probabilities and a
``failed`` list for unreadable images. No label is read here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
SIZE = 448
Image.MAX_IMAGE_PIXELS = 200_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def center_square(rgb: Image.Image, fraction: float) -> Image.Image:
    """Square crop centred on the image, side = fraction of the shorter side (aspect preserved)."""
    side = max(1, round(min(rgb.size) * fraction))
    left, top = (rgb.width - side) // 2, (rgb.height - side) // 2
    return rgb.crop((left, top, left + side, top + side))


def preprocess(path: str, crop: float = 0.0) -> np.ndarray | None:
    """Shipped contract (EXIF, RGB, bicubic squash, ImageNet norm); crop > 0 squares the image first."""
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            if crop > 0:
                rgb = center_square(rgb, crop)
            resized = rgb.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    except Exception:  # noqa: BLE001 - unreadable files are reported, not fatal
        return None
    pixels = (np.asarray(resized, dtype=np.float32) / np.float32(255.0) - MEAN) / STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True, help="bundle code/ directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop", type=float, default=0.0, help="0 = shipped squash; else centre-square fraction")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("checkpoint does not match the pin")
    sys.path.insert(0, str(args.code))
    import torch
    from dermssl.release.moco_d4 import load_checkpoint

    model, _ = load_checkpoint(args.checkpoint)
    model = model.to(args.device).eval()
    rows = read_rows(args.rows)
    ids, feats, failed = [], [], []
    started = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool, torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            arrays = list(pool.map(lambda p: preprocess(p, args.crop), [r["image_path"] for r in chunk]))
            kept = [(r, a) for r, a in zip(chunk, arrays) if a is not None]
            failed += [r["image_id"] for r, a in zip(chunk, arrays) if a is None]
            if not kept:
                continue
            pixels = torch.from_numpy(np.stack([a for _, a in kept])).to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = model.features(pixels)
            feats.append(x.float().cpu().numpy())
            ids += [r["image_id"] for r, _ in kept]
            if (start // args.batch) % 100 == 0:
                print(json.dumps({"done": start + len(chunk), "of": len(rows),
                                  "s": round(time.monotonic() - started)}), flush=True)
    x = np.concatenate(feats).astype(np.float32)
    weights = {t: (model.state_dict()[f"{t}_weight"].cpu().numpy(), model.state_dict()[f"{t}_bias"].cpu().numpy())
               for t in ("ham", "pad")}
    probs = {}
    for task, (w, b) in weights.items():
        logits = x @ w.T + b
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        probs[task] = e / e.sum(1, keepdims=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, image_ids=np.asarray(ids), features=x, ham7=probs["ham"], pad6=probs["pad"],
                        failed=np.asarray(failed, dtype=str))
    print(json.dumps({"rows": len(rows), "ok": len(ids), "failed": len(failed),
                      "seconds": round(time.monotonic() - started), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
