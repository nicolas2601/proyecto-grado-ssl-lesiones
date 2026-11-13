#!/usr/bin/env python3
"""CLS4-D4 features (448 px) for a raw DINOv2 ViT-B/14 hub state dict (e.g. a new SSL backbone).

Same recipe as the shipped extractor: 8 dihedral views, CLS of blocks 8-11 with the final norm,
mean over views, row L2. No heads are applied; ham7/pad6 are written as empty arrays.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from extract_d4_features import preprocess, read_rows, sha256_file


def d4_views(pixels):
    import torch

    views = []
    for k in range(4):
        rotated = torch.rot90(pixels, k, dims=(2, 3))
        views += [rotated, torch.flip(rotated, dims=(3,))]
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--hub-repo", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if sha256_file(args.weights) != args.weights_sha256:
        raise SystemExit("weights do not match the pin")
    import torch
    from torch.nn import functional as F

    model = torch.hub.load(str(args.hub_repo), "dinov2_vitb14", source="local", pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    model = model.eval().to(args.device)
    rows = read_rows(args.rows)
    ids, feats, failed = [], [], []
    started = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool, torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            arrays = list(pool.map(preprocess, [r["image_path"] for r in chunk]))
            kept = [(r, a) for r, a in zip(chunk, arrays) if a is not None]
            failed += [r["image_id"] for r, a in zip(chunk, arrays) if a is None]
            if not kept:
                continue
            pixels = torch.from_numpy(np.stack([a for _, a in kept])).to(args.device)
            per_view = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for view in d4_views(pixels):
                    layers = model.get_intermediate_layers(view, n=[8, 9, 10, 11], norm=True, return_class_token=True)
                    per_view.append(torch.cat([cls.float() for _, cls in layers], dim=1))
            x = F.normalize(torch.stack(per_view).mean(0), p=2, dim=-1)
            feats.append(x.float().cpu().numpy())
            ids += [r["image_id"] for r, _ in kept]
            if (start // args.batch) % 100 == 0:
                print(json.dumps({"done": start + len(chunk), "of": len(rows),
                                  "s": round(time.monotonic() - started)}), flush=True)
    x = np.concatenate(feats).astype(np.float32)
    empty = np.zeros((len(ids), 0), dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, image_ids=np.asarray(ids), features=x, ham7=empty, pad6=empty,
                        failed=np.asarray(failed, dtype=str))
    print(json.dumps({"rows": len(rows), "ok": len(ids), "failed": len(failed),
                      "seconds": round(time.monotonic() - started), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
