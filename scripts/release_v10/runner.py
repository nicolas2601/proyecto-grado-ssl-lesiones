#!/usr/bin/env python3
"""Offline V10 inference. Docker uses ONNX; PyTorch is an explicit optional runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TASKS = {"HAM": {"classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                 "output_name": "ham7_probabilities", "model_key": "MOCO_HAM7"},
         "PAD": {"classes": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
                 "output_name": "pad6_probabilities", "model_key": "MOCO_PAD6"}}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(root, expected_manifest_sha256=None, runtime=None):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if expected_manifest_sha256 and sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "dermssl.moco-bundle.v1":
        raise ValueError("Unsupported bundle schema")
    if manifest.get("tasks") != TASKS:
        raise ValueError("Native task/class contract mismatch")
    optional = {"onnx": "classifier.pt", "torch": "model.onnx", None: None}[runtime]
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or path == root:
            raise ValueError("Unsafe bundle path")
        if name == optional and not path.exists():
            continue
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Bundle digest mismatch: {name}")
    return manifest


def infer(root, pixels, runtime, task):
    if runtime == "torch":
        import torch
        from dermssl.release.moco_bundle import load_checkpoint
        torch.set_num_threads(2)
        model, _ = load_checkpoint(root / "classifier.pt")
        with torch.inference_mode():
            return model(torch.from_numpy(pixels))[0 if task == "HAM" else 1].numpy()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / "model.onnx"), options, providers=["CPUExecutionProvider"])
    return session.run([TASKS[task]["output_name"]], {"pixel_values": pixels})[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runtime", choices=("onnx", "torch"), default="onnx")
    parser.add_argument("--task", choices=tuple(TASKS), default="HAM")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = verify_bundle(root, args.manifest_sha256, runtime=args.runtime)
    sys.path.insert(0, str(root / "code"))
    from dermssl.release.moco_preprocess import preprocess_image
    with Image.open(args.image) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds 20 megapixels")
        pixels = preprocess_image(image)
    probabilities = np.asarray(infer(root, pixels, args.runtime, args.task), dtype=float)
    classes = TASKS[args.task]["classes"]
    if (probabilities.shape != (1, len(classes)) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
        raise ValueError("Runtime produced invalid probabilities")
    result = {"release": manifest["release"], "model_key": TASKS[args.task]["model_key"],
              "task": args.task, "runtime": args.runtime, "classes": classes,
              "pred": classes[int(probabilities[0].argmax())], "confidence": float(probabilities.max()),
              "probs": dict(zip(classes, probabilities[0].tolist())), "feature_dim": 3072,
              "recipe": manifest["recipe"], "scope": "Research only; not a clinical diagnosis"}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
