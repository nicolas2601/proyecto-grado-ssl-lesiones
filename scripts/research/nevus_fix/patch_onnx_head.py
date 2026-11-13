#!/usr/bin/env python3
"""Replace the HAM7 (and optionally PAD6) linear head initializers of the shipped V10 ONNX graph.

The encoder subgraph is untouched; only the MatMul weight [3072, K] and bias [K] of a head change.
Verification: on reference features the patched head must equal softmax(x W^T + b) to 1e-5, and the
backbone initializers must be byte-identical to the source graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

HEADS = {"HAM": ("onnx::MatMul_11673", "ham_bias", 7), "PAD": ("onnx::MatMul_11674", "pad_bias", 6)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replace(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, init in enumerate(model.graph.initializer):
        if init.name == name:
            old = numpy_helper.to_array(init)
            if old.shape != array.shape:
                raise ValueError(f"{name}: shape {array.shape} != {old.shape}")
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array.astype(np.float32), name))
            return
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--ham-head", type=Path, help="npz with weight [7,3072], bias [7], classes")
    parser.add_argument("--pad-head", type=Path, help="npz with weight [6,3072], bias [6], classes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.source) != args.source_sha256:
        raise SystemExit("source ONNX does not match the pin")
    model = onnx.load(str(args.source))
    expected = {"HAM": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "PAD": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]}
    changed = {}
    for task, path in (("HAM", args.ham_head), ("PAD", args.pad_head)):
        if path is None:
            continue
        head = np.load(path, allow_pickle=False)
        if head["classes"].tolist() != expected[task]:
            raise SystemExit(f"{task} head classes {head['classes'].tolist()} != {expected[task]}")
        w_name, b_name, k = HEADS[task]
        replace(model, w_name, np.ascontiguousarray(head["weight"].T))
        replace(model, b_name, head["bias"])
        changed[task] = sha256(path)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(args.out))
    print(json.dumps({"out": str(args.out), "sha256": sha256(args.out), "heads_changed": changed}))


if __name__ == "__main__":
    main()
