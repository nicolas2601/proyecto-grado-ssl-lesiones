#!/usr/bin/env python3
"""Per-view CLS4 features (8 D4 views, before averaging) to study cheaper test-time recipes offline.

Output npz: image_ids, views (N, 8, 3072) float16 raw CLS4 per D4 view in the shipped view order
(identity, hflip, rot90, rot90+hflip, rot180, rot180+hflip, rot270, rot270+hflip), failed.
The shipped feature is L2(mean over the 8 views). No label is read here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from extract_d4_features import preprocess, read_rows, sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True, help="bundle code/ directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=float, default=0.0)
    parser.add_argument("--fp32", action="store_true", help="no bf16 autocast (matches the served FP32 graph)")
    parser.add_argument("--identity-only", action="store_true", help="store only the identity view (N, 1, 3072)")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("checkpoint does not match the pin")
    sys.path.insert(0, str(args.code))
    import torch
    from dermssl.release.moco_d4 import d4_views, load_checkpoint

    model, _ = load_checkpoint(args.checkpoint)
    model = model.to(args.device).eval()
    rows = read_rows(args.rows)
    ids, feats, failed = [], [], []
    started = time.monotonic()
    with ThreadPoolExecutor(8) as pool, torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            arrays = list(pool.map(lambda p: preprocess(p, args.crop), [r["image_path"] for r in chunk]))
            kept = [(r, a) for r, a in zip(chunk, arrays) if a is not None]
            failed += [r["image_id"] for r, a in zip(chunk, arrays) if a is None]
            if not kept:
                continue
            pixels = torch.from_numpy(np.stack([a for _, a in kept])).to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                views = [pixels] if args.identity_only else d4_views(pixels)
                per_view = torch.stack([model.backbone.forward_cls4(v) for v in views], dim=1)
            feats.append(per_view.float().cpu().numpy().astype(np.float16))
            ids += [r["image_id"] for r, _ in kept]
            if (start // args.batch) % 100 == 0:
                print(json.dumps({"done": start + len(chunk), "of": len(rows),
                                  "s": round(time.monotonic() - started)}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, image_ids=np.asarray(ids), views=np.concatenate(feats), failed=np.asarray(failed, dtype=str))
    print(json.dumps({"rows": len(rows), "ok": len(ids), "failed": len(failed),
                      "seconds": round(time.monotonic() - started), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
