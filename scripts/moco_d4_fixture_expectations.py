#!/usr/bin/env python3
"""Expected HAM7/PAD6 probabilities for the synthetic release fixtures (native hub D4 reference, 448 px).

Run where the encoder and hub code exist. Writes parity-fixtures.json next to the given fixtures directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermssl import run_environment  # noqa: E402
from dermssl.release import moco_d4  # noqa: E402
from dermssl.release.moco_head_fit import reader_probabilities  # noqa: E402
from dermssl.release.moco_inputs import ENCODER_SHA256, sha256_file  # noqa: E402
from dermssl.release.moco_preprocess import preprocess_image  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--hub-repo", type=Path, required=True)
    parser.add_argument("--heads-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.weights) != ENCODER_SHA256:
        raise ValueError("Encoder digest mismatch")
    native = torch.hub.load(str(args.hub_repo), "dinov2_vitb14", source="local", pretrained=False)
    native.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    native.eval().cuda()
    heads = {}
    for task in ("HAM", "PAD"):
        doc = json.loads((args.heads_dir / f"{task}-candidate-reader.json").read_text())
        heads[task] = doc
    cases = []
    for path in sorted(args.fixtures.glob("synthetic-*.png"), key=lambda p: int(p.stem.split("-")[1])):
        with Image.open(path) as image:
            pixels = preprocess_image(image, moco_d4.RESOLUTION)
        views = moco_d4.d4_views(torch.from_numpy(pixels).cuda())
        with torch.inference_mode():
            per_view = [torch.cat([c.float() for _, c in native.get_intermediate_layers(
                v, n=[8, 9, 10, 11], norm=True, return_class_token=True)], dim=1) for v in views]
            features = torch.nn.functional.normalize(torch.stack(per_view).mean(0), dim=-1).cpu().numpy()
        expected = {task: reader_probabilities(features, heads[task])[0].tolist() for task in heads}
        cases.append({"image": f"fixtures/{path.name}", "image_sha256": sha256_file(path), "expected": expected})
    with args.output.open("x") as stream:
        json.dump(cases, stream, indent=2, allow_nan=False)
        stream.write("\n")
    run_environment.write_sidecar(args.output, args)
    print(json.dumps({"cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
