#!/usr/bin/env python3
"""Assemble the 448 px CLS4+D4 candidate: Docker build context (bundle/) and release-style manifest.

Layout: OUTPUT/Dockerfile, OUTPUT/requirements-onnx.txt, OUTPUT/bundle/ (ONNX runtime files, code, heads,
fixtures, classifier.pt, parity-fixtures.json, manifest.json). The ONNX runtime treats classifier.pt as optional.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermssl.release.moco_d4 import RECIPE  # noqa: E402
RELEASE_SCRIPTS = ROOT / "scripts/release_v10"
RELEASE = "v10.1.0-candidate"
PREPROCESS = {"exif_transpose": True, "color_mode": "RGB", "resize": [448, 448], "interpolation": "PIL.BICUBIC",
              "crop": None, "mean": [.485, .456, .406], "std": [.229, .224, .225]}


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_code(bundle):
    code = bundle / "code/dermssl/release"
    code.mkdir(parents=True)
    for path in (bundle / "code/dermssl/__init__.py", code / "__init__.py"):
        path.write_text("")
    for name in ("moco_bundle", "moco_d4", "moco_preprocess", "plain_dinov2"):
        shutil.copy2(ROOT / f"src/dermssl/release/{name}.py", code / f"{name}.py")
    shutil.copy2(ROOT / "src/dermssl/release/moco_preprocess.py", bundle / "preprocess.py")
    shutil.copy2(RELEASE_SCRIPTS / "runner.py", bundle / "runner.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--heads-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if len(args.source_commit) != 40:
        raise ValueError("A full source commit is required")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "bundle"
    bundle.mkdir()
    copy_code(bundle)
    for name in ("model.onnx", "parity-fixtures.json", "validation.json", "provenance.json"):
        shutil.copy2(args.build_dir / name, bundle / name)
    shutil.copytree(args.build_dir / "fixtures", bundle / "fixtures")
    (bundle / "heads").mkdir()
    for task in ("HAM", "PAD"):
        shutil.copy2(args.heads_dir / f"{task}-candidate-reader.json", bundle / "heads" / f"{task}-reader.json")
    shutil.copy2(args.build_dir / "classifier.pt", bundle / "classifier.pt")
    shutil.copy2(RELEASE_SCRIPTS / "requirements-onnx.txt", args.output / "requirements-onnx.txt")
    shutil.copy2(RELEASE_SCRIPTS / "requirements-torch.txt", bundle / "requirements-torch.txt")
    shutil.copy2(RELEASE_SCRIPTS / "Dockerfile", args.output / "Dockerfile")
    for name in ("LICENSE", "LICENSE-WEIGHTS"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, bundle / name)
    manifest = {"schema": "dermssl.moco-bundle.v2", "release": RELEASE, "seed": 42,
                "backbone_sha256": "fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e",
                "tasks": {"HAM": {"classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                                  "output_name": "ham7_probabilities", "model_key": "MOCO_HAM7"},
                          "PAD": {"classes": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
                                  "output_name": "pad6_probabilities", "model_key": "MOCO_PAD6"}},
                "source_commit": args.source_commit, "feature_dim": 3072, "views": 8,
                "input": {"name": "pixel_values", "dtype": "float32", "shape": ["batch", 3, 448, 448]},
                "preprocessing": PREPROCESS, "recipe": RECIPE, "scope": "Research only; candidate, not published"}
    files = {}
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        files[str(path.relative_to(bundle))] = sha256(path)
    manifest["files"] = files
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "files": len(files)}))


if __name__ == "__main__":
    main()
