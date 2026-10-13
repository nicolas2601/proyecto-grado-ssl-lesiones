"""Convert reviewed public SSL encoders into the V8 local encoder formats.

No downloads are performed. The operator supplies an already obtained checkpoint and
its official source/license metadata. Unknown tensor layouts fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermssl.segmentation_methods.common.networks import ResNet50Stages
from dermssl.segmentation_methods.common.dino import PatchTransformer


ADAPTER_VERSION = "v8-public-ssl-adapter-2"
RESNET_ORIGINS = {"mocov1", "mocov2", "mocov3", "simclr"}
ALLOWED_ORIGINS = RESNET_ORIGINS | {"dino"}
RESNET_PREFIXES = (
    "module.encoder_q.", "encoder_q.", "module.base_encoder.", "base_encoder.",
    "module.backbone.", "backbone.", "module.",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_state(path: Path, origin: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a weights-only dictionary")
    state = None
    wrapper_keys = ("teacher", "state_dict", "model", "student")
    for key in wrapper_keys:
        value = payload.get(key)
        if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
            state = value
            break
    if state is None and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        state = payload
    if state is None:
        raise ValueError("No recognized tensor state dict; TensorFlow or unknown containers need a separately reviewed adapter")
    return state


def strip_resnet_prefix(state):
    for prefix in RESNET_PREFIXES:
        if any(key.startswith(prefix) for key in state):
            stripped = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if stripped and any(k.startswith(("conv1.", "bn1.", "layer1.")) for k in stripped):
                return stripped
    if any(k.startswith(("conv1.", "bn1.", "layer1.")) for k in state):
        return state
    raise ValueError("Unrecognized ResNet checkpoint prefix")


def convert_resnet(state, model):
    source = strip_resnet_prefix(state)
    target = model.state_dict()
    mapped = {}
    used = set()

    def put(dst, src):
        if src not in source:
            raise ValueError(f"Required encoder tensor missing: {src}")
        value = source[src]
        if tuple(value.shape) != tuple(target[dst].shape):
            raise ValueError(f"Shape mismatch for {src} -> {dst}: {tuple(value.shape)} != {tuple(target[dst].shape)}")
        mapped[dst] = value.detach().cpu()
        used.add(src)

    for suffix, src in (("0.weight", "conv1.weight"),):
        put("stem." + suffix, src)
    for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        put("stem.1." + suffix, "bn1." + suffix)

    for stage in range(1, 5):
        for block in range((3, 4, 6, 3)[stage - 1]):
            srcbase = f"layer{stage}.{block}"
            dstbase = f"stages.{stage - 1}.{block}"
            for src_i, dst_i in ((1, 0), (2, 3), (3, 6)):
                for suffix in ("weight", "bias"):
                    s = f"{srcbase}.conv{src_i}.{suffix}"
                    d = f"{dstbase}.body.{dst_i}.{suffix}"
                    if s in source: put(d, s)
                    elif suffix == "bias": continue
                    else: raise ValueError(f"Required encoder tensor missing: {s}")
                for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    s = f"{srcbase}.bn{src_i}.{suffix}"
                    put(f"{dstbase}.body.{dst_i + 1}.{suffix}", s)
            if f"{srcbase}.downsample.0.weight" in source:
                put(f"{dstbase}.skip.0.weight", f"{srcbase}.downsample.0.weight")
                for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    put(f"{dstbase}.skip.1.{suffix}", f"{srcbase}.downsample.1.{suffix}")

    missing = set(target) - set(mapped)
    if missing:
        raise ValueError("Adapter did not map every local encoder tensor: " + ", ".join(sorted(missing)[:8]))
    discarded = sorted(set(source) - used)
    bad = [k for k in discarded if not re.match(r"(fc|head|projector|projection_head|queue|predictor)(\.|$)", k)]
    if bad:
        raise ValueError("Unexpected unmapped ResNet keys (refusing partial load): " + ", ".join(bad[:8]))
    model.load_state_dict(mapped, strict=True)
    return mapped, discarded


def tensorflow_checkpoint_files(prefix: Path):
    text = str(prefix)
    if text.endswith(".index"):
        text = text[:-6]
    base = Path(text)
    files = sorted(base.parent.glob(base.name + ".*"))
    files = [p for p in files if p.suffix == ".index" or ".data-" in p.name]
    if not any(p.suffix == ".index" for p in files) or not any(".data-" in p.name for p in files):
        raise ValueError("SimCLR TensorFlow checkpoint needs its .index and .data shard files")
    return base, files


def tensorflow_checkpoint_sha(files):
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def convert_simclr_tensorflow(prefix: Path, model):
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow checkpoint reader is required for the official SimCLR format; no package was installed") from error
    prefix, files = tensorflow_checkpoint_files(prefix)
    reader = tf.train.load_checkpoint(str(prefix))
    names = set(reader.get_variable_to_shape_map())

    def layer_order(pattern):
        layers = set()
        for name in names:
            match = re.fullmatch(pattern, name)
            if match:
                layers.add(match.group(1))
        def index(layer):
            match = re.search(r"_(\\d+)$", layer)
            return int(match.group(1)) if match else -1
        return sorted(layers, key=index)

    conv_layers = layer_order(r"(base_model/conv2d(?:_\\d+)?)/kernel")
    bn_layers = layer_order(r"(base_model/batch_normalization(?:_\\d+)?)/(?:gamma|beta|moving_mean|moving_variance)")
    conv_targets = ["stem.0.weight"]
    bn_targets = ["stem.1"]
    for stage in range(4):
        for block_index in range((3, 4, 6, 3)[stage]):
            block = model.stages[stage][block_index]
            base = f"stages.{stage}.{block_index}"
            if isinstance(block.skip, torch.nn.Sequential):
                conv_targets.append(base + ".skip.0.weight")
                bn_targets.append(base + ".skip.1")
            conv_targets.extend(base + f".body.{index}.weight" for index in (0, 3, 6))
            bn_targets.extend(base + f".body.{index}" for index in (1, 4, 7))
    if len(conv_layers) != len(conv_targets) or len(bn_layers) != len(bn_targets):
        raise ValueError(f"SimCLR ResNet50 tensor count mismatch: conv={len(conv_layers)}/{len(conv_targets)}, bn={len(bn_layers)}/{len(bn_targets)}")

    target = model.state_dict()
    mapped = {}
    used = set()
    for layer, dst in zip(conv_layers, conv_targets, strict=True):
        key = layer + "/kernel"
        value = torch.from_numpy(reader.get_tensor(key)).permute(3, 2, 0, 1).contiguous()
        if tuple(value.shape) != tuple(target[dst].shape):
            raise ValueError(f"SimCLR convolution shape mismatch for {key} -> {dst}")
        mapped[dst] = value
        used.add(key)
    bn_fields = (("gamma", "weight"), ("beta", "bias"),
                 ("moving_mean", "running_mean"), ("moving_variance", "running_var"))
    for layer, dstbase in zip(bn_layers, bn_targets, strict=True):
        for source_suffix, target_suffix in bn_fields:
            key = layer + "/" + source_suffix
            value = torch.from_numpy(reader.get_tensor(key))
            dst = dstbase + "." + target_suffix
            if tuple(value.shape) != tuple(target[dst].shape):
                raise ValueError(f"SimCLR batch-normalization shape mismatch for {key} -> {dst}")
            mapped[dst] = value
            used.add(key)
        tracked = dstbase + ".num_batches_tracked"
        mapped[tracked] = torch.zeros_like(target[tracked])
    if set(target) != set(mapped):
        raise ValueError("SimCLR conversion did not populate the complete local encoder state")
    discarded = sorted(names - used)
    allowed = ("head_supervised/", "head_contrastive/", "global_step")
    unexpected = [name for name in discarded
                  if not name.startswith(allowed) and not name.startswith(("global_step/",))]
    if unexpected:
        raise ValueError("Unexpected SimCLR tensors outside the encoder/known heads: " + ", ".join(unexpected[:10]))
    model.load_state_dict(mapped, strict=True)
    return mapped, discarded, tensorflow_checkpoint_sha(files), [str(p) for p in files]


def normalize_dino_state(state):
    for prefix in ("module.backbone.", "backbone.", "module."):
        if any(k.startswith(prefix + "patch_embed.") for k in state):
            normalized = {}
            for key, value in state.items():
                if key.startswith(prefix):
                    normalized[key[len(prefix):]] = value
                elif key.startswith("module."):
                    normalized[key[len("module."):]] = value
                else:
                    normalized[key] = value
            return normalized
    if any(k.startswith("patch_embed.") for k in state):
        return state
    raise ValueError("Unrecognized DINO ViT checkpoint prefix")


def convert_dino(state, model):
    source = normalize_dino_state(state)
    target = model.state_dict()
    mapped = {}
    used = set()

    def put(dst, src):
        if src not in source:
            raise ValueError(f"Required DINO tensor missing: {src}")
        value = source[src]
        if tuple(value.shape) != tuple(target[dst].shape):
            raise ValueError(f"Shape mismatch for {src} -> {dst}: {tuple(value.shape)} != {tuple(target[dst].shape)}")
        mapped[dst] = value.detach().cpu()
        used.add(src)

    put("patch.weight", "patch_embed.proj.weight")
    put("patch.bias", "patch_embed.proj.bias")
    put("class_token", "cls_token")
    pos = source.get("pos_embed")
    if pos is None or pos.ndim != 3 or pos.shape[1] != 197:
        raise ValueError("Expected DINO ViT-S/16 positional embeddings for 224px input (1 CLS + 196 patches)")
    if tuple(pos[:, 1:].shape) != (1, 196, 384):
        raise ValueError("Only DINO ViT-S/16 positional embeddings are supported")
    mapped["class_position"] = pos[:, :1].detach().cpu()
    mapped["position"] = pos[:, 1:].reshape(1, 14, 14, 384).permute(0, 3, 1, 2).contiguous().cpu()
    used.add("pos_embed")

    for i in range(12):
        src = f"blocks.{i}."
        dst = f"transformer.layers.{i}."
        put(dst + "self_attn.in_proj_weight", src + "attn.qkv.weight")
        put(dst + "self_attn.in_proj_bias", src + "attn.qkv.bias")
        for a, b in (("weight", "weight"), ("bias", "bias")):
            put(dst + "self_attn.out_proj." + a, src + "attn.proj." + b)
            put(dst + "linear1." + a, src + "mlp.fc1." + b)
            put(dst + "linear2." + a, src + "mlp.fc2." + b)
            put(dst + "norm1." + a, src + "norm1." + b)
            put(dst + "norm2." + a, src + "norm2." + b)
    for suffix in ("weight", "bias"):
        put("norm." + suffix, "norm." + suffix)
    missing = set(target) - set(mapped)
    if missing:
        raise ValueError("Adapter did not map every local ViT tensor: " + ", ".join(sorted(missing)[:8]))
    discarded = sorted(set(source) - used)
    bad = [k for k in discarded if not re.match(r"(head|last_layer|ibot_head)(\.|$)", k)]
    if bad:
        raise ValueError("Unexpected unmapped DINO keys (refusing partial load): " + ", ".join(bad[:8]))
    model.load_state_dict(mapped, strict=True)
    return mapped, discarded


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origin", choices=sorted(ALLOWED_ORIGINS), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--format", choices=("pytorch", "simclr_tensorflow"), default="pytorch")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--license-id", required=True)
    p.add_argument("--producer-revision", required=True, help="Git revision of this adapter checkout")
    p.add_argument("--input-normalization", choices=("unit_rgb", "imagenet"), required=True)
    return p.parse_args()


def main():
    args = arguments()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise ValueError("Output and provenance must be new paths")
    if args.format == "simclr_tensorflow" and args.origin != "simclr":
        raise ValueError("TensorFlow input adapter is only defined for SimCLR")
    if args.format == "pytorch" and not args.source.is_file():
        raise ValueError("PyTorch source checkpoint must be a file")
    model = PatchTransformer() if args.origin == "dino" else ResNet50Stages(last_stride=1)
    source_files = None
    if args.format == "simclr_tensorflow":
        mapped, discarded, source_hash, source_files = convert_simclr_tensorflow(args.source, model)
    else:
        state = tensor_state(args.source, args.origin)
        mapped, discarded = convert_dino(state, model) if args.origin == "dino" else convert_resnet(state, model)
        source_hash = sha256(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.output)
    provenance = {
        "method": "ssl_" + args.origin,
        "protocol": "public_ssl_initialization_v8",
        "producer_revision": args.producer_revision,
        "source_url": args.source_url,
        "source_license": args.license_id,
        "source_sha256": source_hash,
        "source_files": source_files,
        "checkpoint_sha256": sha256(args.output),
        "architecture": "DINO_ViT-S/16_224" if args.origin == "dino" else "ResNet50_encoder_local_last_stride_1",
        "adapter": ADAPTER_VERSION,
        "input_normalization": args.input_normalization,
        "mapped_tensor_count": len(mapped),
        "discarded_head_keys": discarded,
        "uses_expert_masks": False,
        "uses_diagnoses": False,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
