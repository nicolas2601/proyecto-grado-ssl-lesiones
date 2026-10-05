"""Split PAD-UFES-20 by patient into target_adapt / target_test (docs/33 section 4).

Reads the manifest of ``pad_ufes_manifest.py``, deduplicates by ``img_id`` and by
the sha256 of each file's bytes, and writes one JSON with both id lists, their
counts per diagnostic and the sha256 of each list. ``target_adapt`` is the image
set for SSL adaptation (D3); ``target_test`` is read once by ``pad_e2_eval.py``
under ``target_test_read.lock``.

ASSUMPTION (not in docs/33 or docs/30): 50 % of the patients go to target_adapt
and 50 % to target_test (``--adapt-fraction 0.5``). It is recorded in the JSON
as ``fraction_source``; changing it must be declared before the split is read.

    python scripts/pad_target_split.py --manifest $P/manifest.json --pad-root $P \
        --out $P/target_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import pad_target_split as ts  # noqa: E402
from dermssl import pad_ufes_files as pf  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pad-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=ts.SEED)
    parser.add_argument("--adapt-fraction", type=float, default=ts.ADAPT_FRACTION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, hash_of=None) -> int:
    """``hash_of`` is injected by tests; by default each file's bytes are hashed."""
    args = parse_args(argv)
    out, root = args.out.expanduser(), args.pad_root.expanduser()
    if out.exists():
        print(f"ABORT: {out} ya existe; el split no se reescribe", file=sys.stderr)
        return 1
    manifest_path = args.manifest.expanduser()
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
    hash_of = hash_of or (lambda row: pf.sha256_of(root / row["path"]))
    try:
        document = ts.build_document(rows, hash_of, manifest_sha256=rp.file_sha256(manifest_path),
                                     seed=args.seed, adapt_fraction=args.adapt_fraction)
    except ts.TargetSplitError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    for side in ("target_adapt", "target_test"):
        info = document[side]
        print(f"{side}: {info['n_images']} imagenes, {info['n_patients']} pacientes, {info['by_diagnostic']}")
    print(f"sha256 {document['sha256']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
