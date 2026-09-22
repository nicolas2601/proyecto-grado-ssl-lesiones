"""Fill the torch.hub cache from a single process, before any distributed run.

With ``--nproc_per_node=2`` both ranks call ``torch.hub.load`` at the same
time. On a cold cache they download the same archive to the same path
concurrently and the extraction blows up on the half-written file:

    Downloading: ".../dinov2/zipball/main" to .../hub/main.zip
    Downloading: ".../dinov2/zipball/main" to .../hub/main.zip
    File ".../torch/hub.py", line 341, in _get_cache_or_reload
      _safe_extract_zip(cached_zipfile, hub_dir)

It only shows up on a cold cache, which is why it appeared the first time
TORCH_HOME moved to a fresh directory and never before. Running this once
first makes the race impossible: by the time the ranks start, the archive and
the weights are already on disk.

    python scripts/warm_hub_cache.py --archs dinov2_vits14 dinov2_vitb14 \\
        --receipt salidas/scratch/hub_warmup.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = "facebookresearch/dinov2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--archs", nargs="+", required=True, help="torch.hub names to fetch")
    parser.add_argument("--receipt", type=Path, required=True, help="JSON written once every architecture is cached")
    parser.add_argument("--repo", default=REPO, help="torch.hub repository")
    return parser.parse_args(argv)


def warm(archs: list[str], repo: str = REPO) -> list[dict]:
    """Download and instantiate each architecture once, in this process."""
    import torch

    entries = []
    for arch in archs:
        started = time.monotonic()
        model = torch.hub.load(repo, arch, pretrained=True)
        parameters = sum(p.numel() for p in model.parameters())
        del model
        entries.append({
            "arch": arch,
            "parameters": int(parameters),
            "seconds": round(time.monotonic() - started, 2),
        })
        print(f"[warmup] {arch}: {parameters/1e6:.1f} M parametros en cache")
    return entries


def main() -> None:
    args = parse_args()
    entries = warm(args.archs, args.repo)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps({"repo": args.repo, "archs": entries}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[warmup] recibo en {args.receipt}")


if __name__ == "__main__":
    main()
