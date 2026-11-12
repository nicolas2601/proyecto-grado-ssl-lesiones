#!/usr/bin/env python3
"""Build the 448 px CLS4+D4 MoCo candidate: checkpoint, ONNX graph and parity evidence (no publishing).

Parity gates (tolerance 1e-4 on probabilities, argmax equal):
  native hub D4 features (8 views, averaged) -> portable D4 features -> torch heads -> ONNX heads.
Development drift against the stored BF16 cls4_d4 features is reported, not gated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermssl import run_environment  # noqa: E402
from dermssl.release import moco_d4, moco_inputs, torch_convert, torch_export  # noqa: E402
from dermssl.release.moco_build import load_reader, probability_gate  # noqa: E402
from dermssl.release.moco_bundle import OUTPUT_NAMES  # noqa: E402
from dermssl.release.moco_head_fit import reader_probabilities  # noqa: E402
from dermssl.release.moco_preprocess import preprocess_image  # noqa: E402
from dermssl.release.moco_validation import synthetic_fixtures, write_json  # noqa: E402
from dermssl.release.plain_dinov2 import PlainDinov2  # noqa: E402

RESOLUTION = moco_d4.RESOLUTION
SAMPLE = 8
CHUNK = 4


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("weights", "hub-repo", "heads-dir", "evidence-root", "dev-features-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def native_d4_features(native, pixels):
    """Independent reference: the hub's own intermediate CLS tokens, averaged over the same 8 views."""
    views = moco_d4.d4_views(pixels)
    with torch.inference_mode():
        per_view = []
        for view in views:
            layers = native.get_intermediate_layers(view, n=[8, 9, 10, 11], norm=True, return_class_token=True)
            per_view.append(torch.cat([cls.float() for _, cls in layers], dim=1))
        mean = torch.stack(per_view).mean(0)
        return torch.nn.functional.normalize(mean, p=2, dim=-1).cpu().numpy()


def build_models(args, readers, device):
    native = torch.hub.load(str(args.hub_repo), "dinov2_vitb14", source="local", pretrained=False)
    native.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    native.eval().to(device)
    config = torch_convert.hf_config(native, RESOLUTION)
    portable = PlainDinov2.from_config(config)
    portable.load_state_dict(torch_convert.hub_to_hf_state(native, RESOLUTION), strict=True)
    heads = {task: {name: torch.tensor(reader[name], dtype=torch.float32) for name in ("weight", "bias")}
             for task, reader in readers.items()}
    model = moco_d4.D4MoCoClassifier(portable, heads).eval()
    return native, model, config


def pixels_for(paths):
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            arrays.append(preprocess_image(image, RESOLUTION))
    return np.concatenate(arrays)


def dev_sample(evidence_root, dev_features_dir):
    paths, stored = [], []
    for task, npz_name in (("HAM", "ham_dev-cls4_d4.npz"), ("PAD", "pad_adapt-cls4_d4.npz")):
        _, _, _, rows, _ = moco_inputs.load_development(evidence_root, task)
        feats = np.load(Path(dev_features_dir) / npz_name, allow_pickle=False)["features"]
        paths += [r["image_path"] for r in rows[:SAMPLE]]
        stored.append(feats[:SAMPLE])
    return paths, np.concatenate(stored)


def run_torch(model, pixels, device):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pixels), CHUNK):
            batch = torch.from_numpy(pixels[start:start + CHUNK]).to(device)
            ham, pad = model(batch)
            outputs.append((ham.cpu().numpy(), pad.cpu().numpy()))
    return [np.concatenate([o[i] for o in outputs]) for i in (0, 1)]


def run_onnx(path, pixels):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    outputs = [[], []]
    for start in range(0, len(pixels), CHUNK):
        batch = np.ascontiguousarray(pixels[start:start + CHUNK], dtype=np.float32)
        for i, arr in enumerate(session.run(list(OUTPUT_NAMES), {"pixel_values": batch})):
            outputs[i].append(arr)
    return [np.concatenate(o) for o in outputs]


def main():
    args = arguments()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    readers = {task: load_reader(args.heads_dir / f"{task}-candidate-reader.json", task) for task in ("HAM", "PAD")}
    native, model, config = build_models(args, readers, device)
    provenance = {"candidate": "448 CLS4+D4, C by macro-F1 (grid registered)", "resolution": RESOLUTION,
                  "encoder_sha256": moco_inputs.ENCODER_SHA256,
                  "heads": {t: {"C": r["C"], "n_images": r["n_images"], "sha256": moco_inputs.sha256_file(
                      args.heads_dir / f"{t}-candidate-reader.json")} for t, r in readers.items()}}
    moco_d4.save_checkpoint(args.output / "classifier.pt", model, config, provenance)
    model, _ = moco_d4.load_checkpoint(args.output / "classifier.pt")
    model.to(device)
    torch_export.export_onnx(model.cpu(), args.output / "model.onnx", resolution=RESOLUTION,
                             output_names=OUTPUT_NAMES)
    model.to(device)

    fixtures = synthetic_fixtures(args.output / "fixtures")
    fixture_pixels = pixels_for(fixtures)
    dev_paths, stored = dev_sample(args.evidence_root, args.dev_features_dir)
    pixels = np.concatenate([fixture_pixels, pixels_for(dev_paths)])

    native.to(device)
    native_features = np.concatenate([native_d4_features(native, torch.from_numpy(pixels[s:s + CHUNK]).to(device))
                                      for s in range(0, len(pixels), CHUNK)])
    with torch.inference_mode():
        portable_features = np.concatenate([model.features(torch.from_numpy(pixels[s:s + CHUNK]).to(device)).cpu().numpy()
                                            for s in range(0, len(pixels), CHUNK)])
    feature_gap = float(np.abs(portable_features - native_features).max())
    expected = {task: reader_probabilities(native_features, readers[task]) for task in ("HAM", "PAD")}
    torch_ham, torch_pad = run_torch(model, pixels, device)
    onnx_ham, onnx_pad = run_onnx(args.output / "model.onnx", pixels)
    checks = []
    for form, (ham, pad) in (("pytorch", (torch_ham, torch_pad)), ("onnx", (onnx_ham, onnx_pad))):
        for task, actual in (("HAM", ham), ("PAD", pad)):
            gate = probability_gate(actual, expected[task])
            checks.append({"form": form, "task": task, **gate})
    passed = bool(all(c["passed"] for c in checks)) and feature_gap <= 1e-4
    offset = len(fixture_pixels)
    stored_ham = reader_probabilities(stored[:SAMPLE], readers["HAM"])
    stored_pad = reader_probabilities(stored[SAMPLE:], readers["PAD"])
    drift = {"HAM_argmax_changes_vs_stored_bf16": int(np.sum(
                 stored_ham.argmax(1) != expected["HAM"][offset:offset + SAMPLE].argmax(1))),
             "PAD_argmax_changes_vs_stored_bf16": int(np.sum(
                 stored_pad.argmax(1) != expected["PAD"][offset + SAMPLE:offset + 2 * SAMPLE].argmax(1)))}
    validation = {"passed": passed, "resolution": RESOLUTION, "views": 8, "samples": int(len(pixels)),
                  "native_to_portable_feature_max_abs": feature_gap, "checks": checks,
                  "dev_drift_scope": "BF16 stored development features vs FP32 serving: reported, not gated",
                  "dev_drift": drift,
                  "onnx_sha256": hashlib.sha256((args.output / "model.onnx").read_bytes()).hexdigest()}
    write_json(args.output / "validation.json", validation)
    run_environment.write_sidecar(args.output / "validation.json", args)
    write_json(args.output / "provenance.json", provenance)
    run_environment.write_sidecar(args.output / "provenance.json", args)
    print(json.dumps({"passed": passed, "feature_gap": feature_gap,
                      "max_probability_difference": max(c["max_abs_diff"] for c in checks)}), flush=True)


if __name__ == "__main__":
    main()
