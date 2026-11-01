"""Repin the 026-031 preparer to the final -003 pool.

The pins rotated three times while the chain was debugged (-001 carried a stale
pool_contract, -002 fixed the stage keys but left selection_/refit_ keys at v2,
-003 is fully self-consistent). Each rotation changes four hashes and one path,
which is exactly the kind of edit that goes wrong by hand.

Values are read from the pool directory itself, never retyped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--preparer", type=Path, required=True)
    args = parser.parse_args()

    pins = json.loads((args.pool_dir / "pins.json").read_text(encoding="utf-8"))
    selection = pins["selection"]
    receipt_sha = hashlib.sha256((args.pool_dir / "receipt.json").read_bytes()).hexdigest()
    attestation_sha = hashlib.sha256(args.attestation.read_bytes()).hexdigest()

    text = args.preparer.read_text(encoding="utf-8")
    replacements = {
        r"preflight/source-disjoint-pools-v3-20261101-00[0-9]": f"preflight/{args.pool_dir.name}",
        r'V3_SELECTION_SHA256 = "[0-9a-f]{64}"':
            f'V3_SELECTION_SHA256 = "{selection["file_sha256"]}"',
        r'V3_SELECTION_COHORT_SHA256 = "[0-9a-f]{64}"':
            f'V3_SELECTION_COHORT_SHA256 = "{selection["cohort_sha256"]}"',
        r'V3_POOL_RECEIPT_SHA256 = "[0-9a-f]{64}"':
            f'V3_POOL_RECEIPT_SHA256 = "{receipt_sha}"',
        r"pool-use-attestation\.v3\.20261101\.00[0-9]\.json": args.attestation.name,
        r'V3_ATTESTATION_SHA256 = "[0-9a-f]{64}"':
            f'V3_ATTESTATION_SHA256 = "{attestation_sha}"',
    }
    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text)
        if count == 0:
            raise SystemExit(f"pattern never matched, pin would stay stale: {pattern}")
        print(f"  {count}x {pattern.split('=')[0].strip()[:46]}")

    args.preparer.write_text(text, encoding="utf-8")
    print(f"\nrepinned to {args.pool_dir.name}")
    print(f"  selection  {selection['file_sha256']}  ({selection['record_count']} records)")
    print(f"  receipt    {receipt_sha}")
    print(f"  attestation {attestation_sha}")


if __name__ == "__main__":
    main()
