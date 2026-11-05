"""Frozen pooled features of a bank model on a verified derived development manifest (no fine-tuning)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bank_models_20261030 import build_backbone, load_images, normalize, spec  # noqa: E402

OVERLAY = ROOT / "results/content-exclusion-overlay-v1"
ROLES = {"pad_adapt", "ham_dev"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(model_id, rows, device):
    model = build_backbone(model_id, 0).to(device).eval()
    images = load_images(rows, spec(model_id)["interp"])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            outputs.append(model(normalize(images[start:start + 64].to(device), model_id)).float().cpu())
    features = torch.cat(outputs).numpy()
    if features.ndim != 2 or len(features) != len(rows) or not np.isfinite(features).all():
        raise ValueError("Invalid frozen feature matrix")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing evidence")
    manifest = OVERLAY / f"{args.role}.jsonl"
    receipt = json.loads((OVERLAY / "receipt.json").read_text())
    if receipt["status"] != "COMPLETE" or digest(manifest) != receipt["output_sha256"][manifest.name]:
        raise ValueError("Overlay provenance mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    started = time.monotonic()
    features = extract(args.model, rows, torch.device("cuda"))
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, features=features, image_ids=np.asarray([r["image_id"] for r in rows]),
                            groups=np.asarray([r["identity_component_id"] for r in rows]))
    print(json.dumps({"model": args.model, "role": args.role, "shape": list(features.shape),
                      "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
