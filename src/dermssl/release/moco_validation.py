"""Independent FP32 export parity and a separate BF16 development-drift audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dermssl.release.moco_build import probability_gate
from dermssl.release.moco_head_fit import normalized_features, reader_probabilities
from dermssl.release.moco_inputs import CLASSES, load_development, sha256_file
from dermssl.release.moco_preprocess import preprocess_image


def native_features(model, pixels):
    with torch.inference_mode():
        layers = model.get_intermediate_layers(pixels, n=[8, 9, 10, 11], norm=True, return_class_token=True)
        return torch.cat([cls.float() for _, cls in layers], dim=1)


def synthetic_fixtures(directory):
    directory = Path(directory)
    directory.mkdir()
    rng = np.random.default_rng(42)
    images = [Image.new("RGB", (224, 224), (125, 74, 32)),
              Image.fromarray(rng.integers(0, 256, (193, 257, 3), dtype=np.uint8)),
              Image.new("L", (61, 89), 129), Image.new("RGBA", (221, 243), (190, 60, 110, 150)),
              Image.new("RGB", (125, 67), (19, 130, 220)),
              Image.fromarray(np.tile(np.arange(37, dtype=np.uint8), (31, 1)))]
    paths = []
    for index, image in enumerate(images):
        path = directory / f"synthetic-{index}.png"
        if index == 4:
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
        else:
            image.save(path)
        paths.append(path)
    return paths


def fixture_pixels(paths):
    pixels = []
    for path in paths:
        with Image.open(path) as image:
            pixels.append(preprocess_image(image))
    return np.concatenate(pixels)


def onnx_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def verify_export(native, portable, onnx_path, readers, fixtures):
    from dermssl.release.moco_bundle import OUTPUT_NAMES
    session = onnx_session(onnx_path)
    if [item.name for item in session.get_outputs()] != list(OUTPUT_NAMES):
        raise ValueError("ONNX output names differ from the two native heads")
    pixels = fixture_pixels(fixtures)
    with torch.inference_mode():
        reference_features = native_features(native, torch.from_numpy(pixels)).numpy()
        converted = portable.features(torch.from_numpy(pixels)).numpy()
    feature_max = float(np.abs(converted - normalized_features(reference_features)).max())
    if not np.allclose(converted, normalized_features(reference_features), rtol=1e-4, atol=1e-4):
        raise ValueError(f"Native/portable CLS4 feature parity failed: {feature_max}")
    expected = {task: reader_probabilities(reference_features, readers[task]) for task in CLASSES}
    checks = []
    for size in (1, 2, 3):
        for start in range(0, len(pixels), size):
            batch = pixels[start:start + size]
            with torch.inference_mode():
                actual_torch = portable(torch.from_numpy(batch))
            actual_onnx = session.run(list(OUTPUT_NAMES), {"pixel_values": batch})
            for number, task in enumerate(("HAM", "PAD")):
                reference = expected[task][start:start + size]
                for form, actual in (("pytorch", actual_torch[number].numpy()), ("onnx", actual_onnx[number])):
                    gate = probability_gate(actual, reference)
                    checks.append({"form": form, "task": task, "batch_size": len(batch), **gate})
                    if not gate["passed"]:
                        raise ValueError(f"FP32 parity failed: {checks[-1]}")
    cases = [{"image": f"fixtures/{path.name}", "image_sha256": sha256_file(path),
              "expected": {task: values[index].tolist() for task, values in expected.items()}}
             for index, path in enumerate(fixtures)]
    return {"passed": True, "native_to_portable_feature_max_abs": feature_max,
            "cases": len(fixtures), "checks": checks, "precision": "FP32 native Hub and portable inference"}, cases


def development_drift(native, root, readers, device, samples_per_task=8):
    results = {}
    native.to(device)
    for task in ("HAM", "PAD"):
        cache, _, _, rows, provenance = load_development(root, task)
        rows = rows[:samples_per_task]
        arrays = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                arrays.append(preprocess_image(image))
        pixels = torch.from_numpy(np.concatenate(arrays)).to(device)
        full_precision = native_features(native, pixels).cpu().numpy()
        cached = cache[:len(rows)]
        expected = reader_probabilities(cached, readers[task])
        actual = reader_probabilities(full_precision, readers[task])
        normalized_delta = np.abs(normalized_features(full_precision) - normalized_features(cached))
        results[task] = {"n": len(rows), "sampling": "first allowed development rows, no outcome selection",
                         "manifest_sha256": provenance["manifest_sha256"],
                         "raw_feature_max_abs": float(np.abs(full_precision - cached).max()),
                         "normalized_feature_max_abs": float(normalized_delta.max()),
                         "probability_max_abs": float(np.abs(actual - expected).max()),
                         "argmax_changes": int(np.sum(actual.argmax(1) != expected.argmax(1))),
                         "scope": "Numerical BF16-evaluation/FP32-serving drift, not a new performance estimate"}
    native.cpu()
    return results


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
