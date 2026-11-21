#!/usr/bin/env python3
"""Build the v10.4.2 segmenter manifest (schema dermssl.segmenter.v2) from the evidence files.

  python build_segmenter_manifest.py --package-manifest release_manifest.json \
      --paired-eval paired_isic2018_2594.json --latency latency_vps.json --onnx segmenter.onnx --out DIR

Every metric is read from an evidence file; nothing is typed in. The ONNX hash is recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RELEASE = "v10.4.2"
SIZE = 448
STEPS = [f"letterbox {SIZE} bilinear, pad right-bottom", "float32 /255", "normalize imagenet"]
ACTIVATION = "softmax over 2 channels, foreground channel 1, valid region only"
CANDIDATE = "dpt_s42"
SHIPPED = "shipped_v10_4_1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def isic_block(paired: dict) -> dict:
    model = paired["models"][CANDIDATE]
    metrics, ci = model["metrics"], model["bootstrap_ci_95"]
    return {
        "evaluated_model": "this ONNX file, backend preprocessing contract, frozen thesis evaluator",
        "n": paired["n_cases"],
        "dice": metrics["mean_dice"], "iou": metrics["mean_iou"], "hd95": metrics["mean_hd95"],
        "bootstrap_ci_95": {k: [ci[k]["ci_lower"], ci[k]["ci_upper"]] for k in ("dice", "iou", "hd95")},
        "n_hd95": ci["hd95"]["n_observations"],
        "fail_dice_lt_0_2_pct": model["fail_dice_lt_0_2_pct"],
        "previous_segmenter_same_run": {
            "release": "v10.4.1",
            "dice": paired["models"][SHIPPED]["metrics"]["mean_dice"],
            "iou": paired["models"][SHIPPED]["metrics"]["mean_iou"],
            "hd95": paired["models"][SHIPPED]["metrics"]["mean_hd95"],
        },
        "paired_dice_difference": paired["paired_dice_difference"],
    }


def build(package: dict, paired: dict, latency: dict, onnx: Path) -> dict:
    clinical = package["verified_benchmarks_seed42"]["clinical"]
    return {
        "schema": "dermssl.segmenter.v2",
        "release": RELEASE,
        "model_id": "dinov2-base-dpt-unified",
        "run": f"{package['release_name']}-s{package['selected_seed']}",
        "seed": package["selected_seed"],
        "architecture": package["architecture"],
        "selection": {"criterion": package["selection_criterion"]},
        "onnx": {"file": "segmenter.onnx", "sha256": sha256(onnx), "opset": package["opset_version"],
                 "parameters": package["total_parameters"], "source_file": package["onnx_file"],
                 "source_sha256": package["onnx_sha256"]},
        "input": {"name": "pixel_values", "shape": [1, 3, SIZE, SIZE], "dtype": "float32", "layout": "NCHW",
                  "resize": "bilinear", "preprocessing_required": STEPS,
                  "range": "ImageNet-normalised RGB inside the valid region, 0.0 in the padding"},
        "output": {"name": "logits", "shape": [1, 2, SIZE, SIZE], "dtype": "float32", "activation": ACTIVATION},
        "threshold": package["preprocess_contract"]["default_threshold"],
        "postprocess": "crop the valid region, channel-1 softmax, bilinear (align_corners=False) to the original "
                       "H,W, >= 0.5. No hair removal, no morphology, no largest component.",
        "licenses": {"dinov2_backbone": "Meta DINOv2 ViT-B/14 weights, Apache-2.0",
                     "use": "research / non-commercial"},
        "metrics": {
            "isic2018_2594": isic_block(paired),
            "clinical_313_package_report": {**{k: clinical[k] for k in ("n", "dice", "dice_ci95", "iou", "iou_ci95")},
                                            "source": "package manifest (seed 42); not re-run for this release"},
        },
        "parity": package["parity"],
        "latency_cpu": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package-manifest", "paired-eval", "latency", "onnx", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(json.loads(args.package_manifest.read_text()), json.loads(args.paired_eval.read_text()),
                     json.loads(args.latency.read_text()), args.onnx)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "segmenter_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": sha256(out), "onnx_sha256": manifest["onnx"]["sha256"]}))


if __name__ == "__main__":
    main()
