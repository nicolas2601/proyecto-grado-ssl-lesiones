#!/usr/bin/env python3
"""Build the v10.5.0 single-view classifier: checkpoint, ONNX graph and parity fixtures.

Same frozen encoder and PAD6 head as v10.4.0; the HAM7 reader is refit on single-view CLS4 features
(views_refit.py, subset "id"). Paths are the cluster study layout.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

W = Path("/home/jovyan/work/nevus-fix-20261113")
BUNDLE = Path("/home/jovyan/work/derm-v10.2-evidence/models/classifier")
sys.path.insert(0, str(W / "repo/src"))
sys.path.insert(0, str(BUNDLE))
from dermssl.release import moco_bundle, moco_d4, moco_single, torch_export  # noqa: E402
from preprocess import preprocess_image  # noqa: E402

PARENT = W / "release/classifier.pt"
PARENT_SHA = "6bcc8a0a847f314f448c2121d790d86dcf4de77388664ff7c9ac1846ef196a6d"
HEAD = W / "results/head-views-id.npz"
OUT = W / "release-v10.5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha(PARENT) != PARENT_SHA:
        raise SystemExit("v10.4.0 checkpoint does not match its pin")
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    d4, meta = moco_d4.load_checkpoint(PARENT)
    head = np.load(HEAD, allow_pickle=False)
    if head["classes"].tolist() != list(moco_bundle.TASK_CLASSES["HAM"]) or head["views"].tolist() != [0]:
        raise SystemExit("head is not the single-view HAM7 reader")
    model = moco_single.from_d4(d4, torch.from_numpy(head["weight"]).float(), torch.from_numpy(head["bias"]).float())
    provenance = {**meta["provenance"], "release": "v10.5.0", "parent_checkpoint_sha256": PARENT_SHA,
                  "ham_head": "multi-domain reader cell:2:3 refit on single-view CLS4 (views_refit.py, subset id)",
                  "ham_head_npz_sha256": sha(HEAD)}
    OUT.mkdir(parents=True)
    checkpoint = moco_single.save_checkpoint(OUT / "classifier.pt", model, meta["config"], provenance)
    onnx_path = torch_export.export_onnx(model, OUT / "model.onnx", resolution=448,
                                         output_names=moco_bundle.OUTPUT_NAMES)
    fixtures = []
    for item in json.loads((BUNDLE / "parity-fixtures.json").read_text()):
        with Image.open(BUNDLE / item["image"]) as image:
            pixels = torch.from_numpy(preprocess_image(image, 448))
        with torch.inference_mode():
            ham, pad = (t[0].double().numpy() for t in model(pixels))
        fixtures.append({**item, "expected": {"HAM": ham.tolist(), "PAD": pad.tolist()}})
    (OUT / "parity-fixtures.json").write_text(json.dumps(fixtures))
    print(json.dumps({"classifier.pt": sha(checkpoint), "model.onnx": sha(onnx_path),
                      "parity-fixtures.json": sha(OUT / "parity-fixtures.json"), "fixtures": len(fixtures)}))


if __name__ == "__main__":
    main()
