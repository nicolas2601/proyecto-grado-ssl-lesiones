"""Frozen CLS224 export for verified derived development manifests only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROLES = {"pad_adapt", "ham_dev"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows):
    keys = ("image_id", "image_path", "identity_component_id")
    if not rows or any(not str(row.get(key, "")).strip() for row in rows for key in keys):
        raise ValueError("Expected nonempty image IDs, paths and identity components")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    return rows


def reserve_output(path):
    return Path(path).open("xb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=sorted(ALLOWED_ROLES), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in ((args.receipt, args.receipt_sha),
                           (args.manifest, args.manifest_sha), (args.weights, args.weights_sha)):
        if digest(path) != expected:
            raise ValueError(f"Provenance digest mismatch: {path}")
    rows = validate_rows([json.loads(line) for line in args.manifest.read_text().splitlines()])
    if args.manifest.stem != args.role or args.manifest.parent != args.receipt.parent:
        raise ValueError("Only a named manifest adjacent to its overlay receipt is allowed")
    bindings = json.loads((ROOT / "config/paths.json").read_text())
    frozen = Path(bindings["frozen_experiment_root"])
    sys.path[:0] = [str(ROOT / "scripts"), str(frozen / "scripts"), str(frozen / "code/src")]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from evaluate_final_refit import load_encoder
    from extract_preprocessing_variants import extract
    torch.set_num_threads(2)
    started = time.monotonic()
    with reserve_output(args.output) as stream:
        model = load_encoder(args.weights)
        features = extract(model, rows, 224, False)
        if features.shape != (len(rows), 768) or not np.isfinite(features).all():
            raise ValueError("Invalid frozen CLS feature matrix")
        np.savez_compressed(stream, features=features,
                            image_ids=np.asarray([row["image_id"] for row in rows]),
                            groups=np.asarray([row["identity_component_id"] for row in rows]))
    result = {"role": args.role, "images": len(rows), "weights_sha256": args.weights_sha,
              "manifest_sha256": args.manifest_sha, "overlay_sha256": args.receipt_sha,
              "features_sha256": digest(args.output), "seconds": time.monotonic() - started}
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
