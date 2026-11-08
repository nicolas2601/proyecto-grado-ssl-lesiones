#!/usr/bin/env python3
"""Build a complete seed-42 MoCo classifier and verify native/PT/ONNX parity."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermssl.release import moco_bundle, torch_convert, torch_export  # noqa: E402
from dermssl.release.moco_build import MANIFEST_RECIPE, api_report, load_reader  # noqa: E402
from dermssl.release.moco_inputs import ENCODER_SHA256, sha256_file  # noqa: E402
from dermssl.release.moco_validation import development_drift, synthetic_fixtures, verify_export, write_json  # noqa: E402
from dermssl.release.plain_dinov2 import PlainDinov2  # noqa: E402

SOURCE_NAMES = ("moco_bundle", "moco_preprocess", "moco_head_fit", "moco_inputs", "moco_build",
                "moco_validation", "plain_dinov2", "torch_convert", "torch_export", "onnx_tools")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "readers-dir", "hub-repo", "evidence-root", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sources():
    paths = [ROOT / f"src/dermssl/release/{name}.py" for name in SOURCE_NAMES]
    paths += [Path(__file__), ROOT / "scripts/fit_moco_release_readers.py", ROOT / "scripts/release_v10/runner.py"]
    paths += [ROOT / "scripts/release_v10" / name for name in ("Dockerfile", "requirements-onnx.txt", "requirements-torch.txt")]
    paths += [ROOT / "LICENSE", ROOT / "LICENSE-WEIGHTS"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def provenance(args, readers):
    resolved = json.loads((args.weights.parent / "resolved-params.json").read_text())
    training = {k: v for k, v in resolved.items() if k not in ("manifest", "init_weights")}
    if (resolved["seed"] != 42 or resolved["clinical_quota"] != 4
            or resolved["max_steps"] != 6000 or resolved["cosine_anchor"] or resolved["l2sp"]):
        raise ValueError("Source run differs from the approved proposed recipe")
    hub_sources = {str(p.relative_to(args.hub_repo)): sha256_file(p) for p in args.hub_repo.rglob("*.py")}
    return {"release": "v10.0.0", "method": "MoCo-v3-style DINOv2 ViT-B/14 quota12+4 CLS4",
            "source_commit": args.source_commit, "seed": 42, "backbone_sha256": ENCODER_SHA256,
            "training": training, "head_fits": {t: {k: v for k, v in r.items() if k not in ("weight", "bias")}
                                               for t, r in readers.items()},
            "source_sha256": sources(), "hub_source_sha256": hub_sources,
            "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                         **{name: importlib.metadata.version(name) for name in ("numpy", "onnx", "onnxruntime", "Pillow", "scikit-learn")}},
            "scope": "Research only; final development fit, PAD transductive, HAM development selection-exposed"}


def build_models(args, readers):
    native = torch.hub.load(str(args.hub_repo), "dinov2_vitb14", source="local", pretrained=False)
    native.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    native.eval()
    config = torch_convert.hf_config(native)
    portable = PlainDinov2.from_config(config)
    portable.load_state_dict(torch_convert.hub_to_hf_state(native), strict=True)
    heads = {task: {name: torch.tensor(reader[name], dtype=torch.float32) for name in ("weight", "bias")}
             for task, reader in readers.items()}
    return native, moco_bundle.MoCoClassifier(portable, heads).eval(), config


def copy_runtime(output, readers_dir):
    code = output / "code/dermssl/release"
    code.mkdir(parents=True)
    for path in (output / "code/dermssl/__init__.py", code / "__init__.py"):
        path.write_text("")
    for name in ("moco_bundle", "moco_preprocess", "plain_dinov2"):
        shutil.copy2(ROOT / f"src/dermssl/release/{name}.py", code / f"{name}.py")
    shutil.copy2(ROOT / "src/dermssl/release/moco_preprocess.py", output / "preprocess.py")
    shutil.copy2(ROOT / "scripts/release_v10/runner.py", output / "runner.py")
    for name in ("requirements-onnx.txt", "requirements-torch.txt", "Dockerfile"):
        shutil.copy2(ROOT / "scripts/release_v10" / name, output / name)
    heads_dir = output / "heads"
    heads_dir.mkdir()
    for task in ("HAM", "PAD"):
        for suffix in ("json", "npz"):
            shutil.copy2(readers_dir / f"{task}-reader.{suffix}", heads_dir / f"{task}-reader.{suffix}")
    (output / "LICENSES").write_text((ROOT / "LICENSE").read_text() + "\n\n" + (ROOT / "LICENSE-WEIGHTS").read_text())


def write_manifest(args, recorded):
    tasks = {task: {"classes": list(names), "output_name": output_name, "model_key": f"MOCO_{task}{len(names)}"}
             for (task, names), output_name in zip(moco_bundle.TASK_CLASSES.items(), moco_bundle.OUTPUT_NAMES, strict=True)}
    files = {str(p.relative_to(args.output)): sha256_file(p) for p in args.output.rglob("*")
             if p.is_file() and p.name != "manifest.json"}
    manifest = {"schema": "dermssl.moco-bundle.v1", "release": "v10.0.0", "seed": 42,
                "backbone_sha256": ENCODER_SHA256, "tasks": tasks, "recipe": MANIFEST_RECIPE,
                "source_commit": args.source_commit, "source_sha256": recorded["source_sha256"],
                "files": files, "report": "report.json", "scope": recorded["scope"],
                "feature_dim": 3072,
                "input": {"name": "pixel_values", "dtype": "float32", "shape": ["batch", 3, 224, 224]},
                "preprocessing": {"exif_transpose": True, "color_mode": "RGB", "resize": [224, 224],
                                  "interpolation": "PIL.BICUBIC", "crop": None,
                                  "mean": [.485, .456, .406], "std": [.229, .224, .225]}}
    write_json(args.output / "manifest.json", manifest)
    checksums = [f"{digest}  {name}" for name, digest in sorted(files.items())]
    checksums.append(f"{sha256_file(args.output / 'manifest.json')}  manifest.json")
    (args.output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


def main():
    args = arguments()
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ValueError("An actual committed source revision is required")
    if sha256_file(args.weights) != ENCODER_SHA256:
        raise ValueError("Source encoder SHA differs from the fixed seed-42 artifact")
    if not args.report.is_file():
        raise FileNotFoundError("Generated V10 development report is required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    readers = {task: load_reader(args.readers_dir / f"{task}-reader.json", task) for task in ("HAM", "PAD")}
    recorded = provenance(args, readers)
    native, portable, config = build_models(args, readers)
    moco_bundle.save_checkpoint(args.output / "classifier.pt", portable, config, recorded)
    portable, _ = moco_bundle.load_checkpoint(args.output / "classifier.pt")
    torch_export.export_onnx(portable, args.output / "model.onnx", output_names=moco_bundle.OUTPUT_NAMES)
    fixtures = synthetic_fixtures(args.output / "fixtures")
    validation, cases = verify_export(native, portable, args.output / "model.onnx", readers, fixtures)
    validation["bf16_to_fp32_drift"] = development_drift(native, args.evidence_root, readers, args.device)
    write_json(args.output / "validation.json", validation)
    write_json(args.output / "parity-fixtures.json", cases)
    write_json(args.output / "provenance.json", recorded)
    copy_runtime(args.output, args.readers_dir)
    report = api_report(json.loads(args.report.read_text()))
    report["deployment"] = {"seed": 42, "reader_C": {t: r["C"] for t, r in readers.items()},
                            "backbone_sha256": ENCODER_SHA256, "source_commit": args.source_commit}
    write_json(args.output / "report.json", report)
    if sources() != recorded["source_sha256"] or sha256_file(args.weights) != ENCODER_SHA256:
        raise ValueError("Build inputs changed while exporting")
    write_manifest(args, recorded)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "manifest_sha256": sha256_file(args.output / "manifest.json"),
                      "max_probability_difference": max(c["max_abs_diff"] for c in validation["checks"])}), flush=True)


if __name__ == "__main__":
    main()
