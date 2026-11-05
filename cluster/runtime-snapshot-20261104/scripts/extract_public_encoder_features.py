"""Local-only frozen public vision extraction; no text tower, labels or remote code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {"dinov2-large": ("facebook/dinov2-large", "dinov2", 224),
            "siglip2-so400m": ("google/siglip2-so400m-patch14-384", "siglip", 384),
            "siglip2-base": ("google/siglip2-base-patch16-224", "siglip", 224),
            "dinov2-small": ("facebook/dinov2-small", "dinov2", 224),
            "dinov2-giant": ("facebook/dinov2-giant", "dinov2", 224)}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_features(family, output, batch_size, dimensions):
    import torch
    if family == "dinov2":
        hidden = getattr(output, "last_hidden_state", None)
        features = hidden[:, 0] if hidden is not None else None
    elif family == "siglip":
        features = getattr(output, "pooler_output", None)
    else:
        raise ValueError("Unknown feature family")
    if features is None or features.shape != (batch_size, dimensions):
        raise ValueError("Unexpected public vision feature shape")
    if not torch.isfinite(features).all():
        raise ValueError("Public features must be finite")
    return features.detach().float()


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Preserve existing frozen features")
    repo, family, side = ENCODERS[args.encoder]
    downloaded = json.loads((args.model_dir / "download-receipt.json").read_text())
    if (downloaded["status"] != "COMPLETE" or downloaded["repo_id"] != repo
            or downloaded["gated"] is not False or downloaded["license"] != "Apache-2.0"):
        raise ValueError("Approved public model receipt required")
    for name, record in downloaded["files"].items():
        path = args.model_dir / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Downloaded public model file changed")
    if digest(args.manifest) != args.manifest_sha or digest(args.receipt) != args.receipt_sha:
        raise ValueError("Cohort provenance mismatch")
    overlay = json.loads(args.receipt.read_text())
    if (args.manifest.name not in {"pad_adapt.jsonl", "ham_dev.jsonl"} or args.manifest.parent != args.receipt.parent
            or overlay["output_sha256"][args.manifest.name] != args.manifest_sha):
        raise ValueError("Only approved development manifests (PAD adaptation, HAM dev) may be extracted")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if not rows or len({r["image_id"] for r in rows}) != len(rows) or any(not r["identity_component_id"] for r in rows):
        raise ValueError("Unique image IDs and patient groups required")
    environment = ROOT / "environments/public-encoders-20261025-v1"
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HOME=str(environment / "hf-cache"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    from torchvision import transforms as T
    from PIL import Image, ImageOps
    from transformers import Dinov2Config, Dinov2Model, SiglipVisionConfig, SiglipVisionModel
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise ValueError("Available GPU required")
    configuration = json.loads((args.model_dir / "config.json").read_text())
    if family == "dinov2":
        config, cls = Dinov2Config(**configuration), Dinov2Model
    else:
        config, cls = SiglipVisionConfig(**configuration["vision_config"]), SiglipVisionModel
    dimensions = config.hidden_size
    processor = json.loads((args.model_dir / "preprocessor_config.json").read_text())
    mean, std = processor["image_mean"], processor["image_std"]
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Invalid native image normalization")
    model, loading = cls.from_pretrained(str(args.model_dir), config=config, local_files_only=True,
                                         token=False, trust_remote_code=False, use_safetensors=True,
                                         weights_only=True, output_loading_info=True,
                                         dtype=torch.float32, attn_implementation="sdpa")
    if loading.get("missing_keys") or any(key.startswith("vision_model.") for key in loading.get("unexpected_keys", [])):
        raise ValueError("Native vision tower weights did not load completely")
    if any("text_model" in name for name, _ in model.named_parameters()):
        raise ValueError("Text tower must not be instantiated for image extraction")
    model.requires_grad_(False).eval().to("cuda:0")
    transform = T.Compose([T.Resize((side, side), interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize(mean, std)])
    features = np.empty((len(rows), dimensions), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            items = rows[start:start + 16]
            batch = []
            for row in items:
                with Image.open(row["image_path"]) as image:
                    batch.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
            batch = torch.stack(batch).to("cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(pixel_values=batch)
            features[start:start + len(items)] = select_features(family, output, len(items), dimensions).cpu().numpy()
    if any(parameter.grad is not None or parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Public encoder must remain frozen")
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, features=features, image_ids=np.asarray([r["image_id"] for r in rows]),
                            groups=np.asarray([r["identity_component_id"] for r in rows]))
    record = {"encoder": args.encoder, "repo_id": repo, "revision": downloaded["revision"],
              "model_weight_sha256": downloaded["files"]["model.safetensors"]["sha256"],
              "features_sha256": digest(args.output), "manifest_sha256": args.manifest_sha,
              "images": len(rows), "dimensions": dimensions, "input_side": side,
              "preprocessing": "Whole-image square bicubic resize,no center crop; native normalization",
              "feature": "Final normalized CLS" if family == "dinov2" else "Learned image attention pool; no text inputs",
              "pretraining_scope": "Public self-supervised encoder" if family == "dinov2" else "Public image-text pretrained comparator; not the SSL deliverable",
              "environment_setup_sha256": digest(environment / "setup-receipt.json"),
              "requirement_lock_sha256": digest(environment / "requirements-pinned.txt"),
              "runtime_torch_version": torch.__version__, "text_tower_instantiated": False,
              "allocated_gpu_hours_upper_bound": (time.monotonic() - started) / 3600,
              "seconds": time.monotonic() - started, "encoder_diagnostic_gradients": False}
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
