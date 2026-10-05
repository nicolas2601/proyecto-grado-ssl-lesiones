"""Check, on a few real pool masks, which hash definition reproduces ``mask_sha256`` of ``mask_manifest.json``.

Read-only diagnostic (prints, writes nothing). For each sampled eligible entry it hashes the mask PNG under
several candidate definitions and says which one equals the manifest. The pool builder's own definition is
``dermssl.mask_manifest.mask_sha256`` (sha256 of ``"(H, W):true=lesion"`` + the boolean array bytes); E5 uses it.

    python scripts/e5_verify_mask_hash.py --pool-root ~/work/proyectonp/salidas/tone/inputs --n 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import mask_manifest  # noqa: E402


def candidates(png_bytes: bytes, mask: np.ndarray) -> dict[str, str]:
    """Hash of one mask under each definition that could have been used."""
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    return {
        "manifest_def (bool + shape)": mask_manifest.mask_sha256(mask),
        "png_file_bytes": sha(png_bytes),
        "bool_tobytes_only": sha(np.ascontiguousarray(mask, dtype=np.bool_).tobytes()),
        "uint8_0_1_tobytes": sha(mask.astype(np.uint8).tobytes()),
        "uint8_0_255_tobytes": sha((mask.astype(np.uint8) * 255).tobytes()),
    }


def check_entry(root: Path, entry: dict) -> dict[str, bool]:
    from PIL import Image  # noqa: PLC0415

    path = root / "masks" / entry["dataset"] / f"{entry['image_id']}.png"
    raw = path.read_bytes()
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    return {name: digest == entry["mask_sha256"] for name, digest in candidates(raw, mask).items()}


def sample(entries: list[dict], n: int) -> list[dict]:
    """The first ``n`` eligible entries per dataset in a fixed order (deterministic, no RNG)."""
    out, seen = [], {}
    for e in sorted((e for e in entries if e.get("eligible")), key=lambda e: (e["dataset"], e["image_id"])):
        if seen.get(e["dataset"], 0) < n:
            seen[e["dataset"]] = seen.get(e["dataset"], 0) + 1
            out.append(e)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pool-root", type=Path, required=True)
    p.add_argument("--n", type=int, default=20, help="eligible entries per dataset")
    args = p.parse_args(argv)
    root = args.pool_root.expanduser()
    payload = json.loads((root / "mask_manifest.json").read_text(encoding="utf-8"))
    entries = payload["masks"] if isinstance(payload, dict) else payload
    totals: dict[str, dict[str, int]] = {}
    for entry in sample(entries, args.n):
        for name, ok in check_entry(root, entry).items():
            totals.setdefault(entry["dataset"], {}).setdefault(name, 0)
            totals[entry["dataset"]][name] += ok
    for dataset, counts in sorted(totals.items()):
        print(dataset, {k: f"{v}/{min(args.n, sum(1 for e in entries if e.get('eligible') and e['dataset'] == dataset))}" for k, v in counts.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
