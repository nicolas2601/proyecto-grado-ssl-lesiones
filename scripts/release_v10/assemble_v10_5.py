#!/usr/bin/env python3
"""Assemble the v10.5.0 single-view bundle from the v10.4.0 bundle (stdlib only, runs on the host).

Copies the v10.4.0 directory, swaps the ONNX graph, HAM7 reader, parity fixtures, runner and adds
the single-view checkpoint module, then rewrites manifest/provenance/validation with fresh digests.
The encoder, preprocessing, PAD6 head and fixtures images are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

RELEASE = "v10.5.0"
SCHEMA = "dermssl.moco-bundle.v3"
SCOPE = ("v10.5.0: same frozen MoCo V10 s42 encoder and multi-domain HAM7 reader recipe as v10.4.0, served with "
         "one 448 px view instead of the 8 D4 views (8x less compute); the HAM7 reader is refit on single-view "
         "CLS4 features. PAD6 head unchanged. Research prototype, not a medical device.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("src", "dst", "onnx", "fixtures", "ham", "moco-single", "runner", "validation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--classifier-sha256", required=True, help="digest of the torch checkpoint (not served)")
    args = parser.parse_args()
    if args.dst.exists():
        raise SystemExit(f"refusing to overwrite {args.dst}")
    manifest = json.loads((args.src / "manifest.json").read_text())
    if manifest.get("release") != "v10.4.0":
        raise SystemExit("source bundle is not v10.4.0")
    shutil.copytree(args.src, args.dst)
    swaps = {"model.onnx": args.onnx, "parity-fixtures.json": args.fixtures, "heads/HAM-reader.json": args.ham,
             "code/dermssl/release/moco_single.py": args.moco_single, "runner.py": args.runner,
             "validation.json": args.validation}
    for name, source in swaps.items():
        shutil.copyfile(source, args.dst / name)
    recipe = {**manifest["recipe"], "tta": "none", "views": 1}
    files = sorted(set(manifest["files"]) | set(swaps))
    provenance = json.loads((args.dst / "provenance.json").read_text())
    provenance.update({"release": RELEASE, "candidate": "448 CLS4 single view, multi-domain HAM7 reader"})
    provenance["heads"]["HAM"] = {"spec": "cell:2:3 multi-domain reader, single-view features", "C": 3.0,
                                  "sha256": sha256(args.dst / "heads/HAM-reader.json"),
                                  "replaces": manifest["files"]["heads/HAM-reader.json"]}
    write_json(args.dst / "provenance.json", provenance)
    digests = {}
    for name in files:
        path = args.dst / name
        if name == "classifier.pt":
            digests[name] = args.classifier_sha256
        elif path.is_file():
            digests[name] = sha256(path)
        else:
            raise SystemExit(f"missing bundle file {name}")
    manifest.update({"schema": SCHEMA, "release": RELEASE, "recipe": recipe, "views": 1, "scope": SCOPE,
                     "source_commit": args.source_commit, "files": digests})
    write_json(args.dst / "manifest.json", manifest)
    print(json.dumps({"manifest_sha256": sha256(args.dst / "manifest.json"), "model.onnx": digests["model.onnx"],
                      "files": len(digests)}))


if __name__ == "__main__":
    main()
