"""Small label-free geometry and gradient diagnostic; never an optimizer step."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def spectrum(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two finite feature rows")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, np.finfo(float).tiny)
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    p = singular / singular.sum() if singular.sum() else np.zeros_like(singular)
    positive = p[p > 0]
    energy = singular ** 2
    return {"images": len(x), "dimensions": x.shape[1],
            "effective_rank_singular": float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.,
            "covariance_participation_ratio": float(energy.sum() ** 2 / np.square(energy).sum()) if energy.sum() else 0.,
            "normalized_feature_std_mean": float(x.std(axis=0).mean()),
            "top_singular_values": singular[:20].tolist(),
            "top1_energy_fraction": float(energy[0] / energy.sum()) if energy.sum() else 0.}


def gradient_summary(slices, gradients, levers):
    import torch
    result = {}
    for block, (start, stop) in slices.items():
        components = [[g for g in values[start:stop] if g is not None] for values in gradients]
        norms = [float(torch.stack([g.float().square().sum() for g in values]).sum().sqrt())
                 if values else 0. for values in components]
        paired = [(a, b) for a, b in zip(gradients[0][start:stop], gradients[1][start:stop])
                  if a is not None and b is not None]
        dot = sum(float((a.float() * b.float()).sum()) for a, b in paired)
        result[block] = {"ssl_norm": norms[0], "distill_norm": norms[1], "l2sp_norm": norms[2],
                         "weighted_distill_norm": levers.anchor_distill * norms[1],
                         "weighted_l2sp_norm": levers.anchor_l2sp * norms[2],
                         "weighted_distill_to_ssl": levers.anchor_distill * norms[1] / norms[0] if norms[0] else None,
                         "ssl_distill_cosine": dot / (norms[0] * norms[1]) if norms[0] * norms[1] else None}
    return result


def gradient_probe(model, anchors, view1, view2, levers, training):
    import torch
    import probe_preprocessing_readout
    from dermssl.pretrain import torch_adaptation as adaptation, torch_simsiam
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    was_training = model.training
    parameters, slices = [], {}
    blocks = adaptation._vit_blocks(model.backbone)
    if len(blocks) != 12:
        raise ValueError("Expected the registered twelve-block backbone")
    for index, block in enumerate(blocks[8:], start=9):
        start = len(parameters)
        parameters.extend(p for p in block.parameters() if p.requires_grad)
        slices[f"block{index:02}"] = (start, len(parameters))
    try:
        model.train(training)
        with torch.enable_grad(), torch.autocast(view1.device.type, dtype=torch.bfloat16,
                                                enabled=view1.device.type == "cuda"):
            outputs = model(view1, view2, return_features=True)
            ssl = torch_simsiam.simsiam_loss(*outputs[:4])
            distill = adaptation.distill_loss(outputs[4], anchors, view1)
            l2sp = adaptation.l2sp_penalty(model.backbone, anchors)
        if not all(torch.isfinite(loss) for loss in (ssl, distill, l2sp)):
            raise ValueError("Nonfinite diagnostic loss")
        gradients = [torch.autograd.grad(ssl, parameters, retain_graph=True, allow_unused=True),
                     torch.autograd.grad(distill, parameters, allow_unused=True),
                     torch.autograd.grad(l2sp, parameters, allow_unused=True)]
        report = {"mode": "train_bn_restored" if training else "eval_bn_fixed",
                  "ssl_loss": float(ssl.detach()), "distill_loss": float(distill.detach()),
                  "l2sp_penalty": float(l2sp.detach()),
                  "blocks": gradient_summary(slices, gradients, levers)}
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        model.train(was_training)
    return report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    import torch
    hasher = hashlib.sha256()
    for name, value in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def main(root):
    import torch
    import probe_preprocessing_readout as contract
    from dermssl.pretrain import config, sampling_contract, torch_adaptation as adaptation, torch_data, torch_simsiam
    from dermssl.pretrain.torch_resume import ResumableRandomSampler
    destination = root / "results/ssl-diagnostic-v1.json"
    if destination.exists():
        raise FileExistsError("Preserve existing SSL diagnostic evidence")
    started = time.monotonic()
    protocol, rows = contract._manifest(root)
    ids = np.array([row["image_id"] for row in rows])
    paths_binding = json.loads((root / "config/paths.json").read_text())
    frozen = Path(paths_binding["frozen_experiment_root"])
    cfg_path = frozen / "config/canonical_final_mixed_refit_s42_6000_001.json"
    cfg = config.PretrainConfig(**json.loads(cfg_path.read_text()))
    binding = json.loads((frozen / "config/canonical-comparison-protocol.json").read_text())
    if cfg.split_hash != binding["canonical_binding_sha256"]:
        raise ValueError("Canonical split binding changed")
    geometry = {}
    for name in ("S0", "refit"):
        with np.load(root / f"results/preprocessing-variants-v1/{name}_squash224.npz", allow_pickle=False) as archive:
            x = contract.align_features(archive["features"], archive["image_ids"], ids)
        geometry[name] = spectrum(x)
    public = Path(cfg.initial_weights)
    final = Path(cfg.output_dir) / "simsiam_dinov2_backbone_final.pt"
    checkpoint = Path(cfg.output_dir) / "resume.pt"
    if digest(public) != protocol["encoders"]["S0"] or digest(final) != protocol["encoders"]["refit"]:
        raise ValueError("Registered encoder weights changed")
    checkpoint_sha = digest(checkpoint)
    cached = json.loads((frozen / "evidence/torchhub-cache.json").read_text())["cached_repos"]
    if len(cached) != 1:
        raise ValueError("Expected one registered local model implementation")
    backbone = torch.hub.load(cached[0], "dinov2_vitb14", source="local", pretrained=False)
    backbone.load_state_dict(torch.load(public, map_location="cpu", weights_only=True), strict=True)
    levers = config.levers_of(cfg)
    anchors = adaptation.build_anchors(backbone, levers)
    model = torch_simsiam.SimSiam(backbone=backbone)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(saved["step"]) != 6000:
        raise ValueError("Expected completed 6000-step checkpoint")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    final_state = torch.load(final, map_location="cpu", weights_only=True)
    if any(not torch.equal(value, model.backbone.state_dict()[name]) for name, value in final_state.items()):
        raise ValueError("Full checkpoint backbone differs from registered frozen refit")
    del saved, final_state
    adaptation.freeze_blocks(model.backbone, cfg.freeze_blocks)
    image_paths = config.collect_images(cfg.image_root)
    sampling = sampling_contract.load_sampling_plan(cfg.sampling_plan, cfg.sampling_plan_sha256, image_paths)
    sampler = ResumableRandomSampler(len(image_paths), cfg.seed, **sampling,
                                    batch_size=16, source_quota=12)
    chosen = list(iter(sampler))[:48]
    if len(chosen) != 48 or cfg.batch_size_per_gpu != 16 or cfg.clinical_batch_quota != 4:
        raise ValueError("Diagnostic requires three registered 12+4 batches")
    dataset = torch_data.SimSiamPairDataset(image_paths, cfg.resolution, cfg.color_constancy_probs,
                                          cfg.seed, cfg.hue_jitter)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda:0")
    gpu_started = time.monotonic()
    model.to(device).eval()
    anchors.to(device)
    before = state_digest(model)
    measurements = []
    for start in range(0, 48, 16):
        indices = chosen[start:start + 16]
        views = [dataset[index] for index in indices]
        v1, v2 = [torch.stack([pair[view] for pair in views]).to(device) for view in (0, 1)]
        domains = np.asarray(sampling["domains"])[indices]
        if np.count_nonzero(domains == 1) != 4:
            raise ValueError("Clinical diagnostic quota changed")
        record = {"batch": start // 16, "image_ids": [image_paths[index].stem for index in indices],
                  "source_images": 12, "clinical_images": 4, "modes": []}
        for training in (False, True):
            record["modes"].append(gradient_probe(model, anchors, v1, v2, levers, training))
        measurements.append(record)
        print(f"diagnosed batch {start // 16 + 1}/3, weights unchanged so far={state_digest(model) == before}", flush=True)
    torch.cuda.synchronize()
    elapsed_gpu = time.monotonic() - gpu_started
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Diagnostic changed model state or accumulated parameter gradients")
    if digest(checkpoint) != checkpoint_sha:
        raise ValueError("Source resume checkpoint changed during diagnostic")
    report = {"scope": "Label-free endpoint diagnostic, no optimizer or weight updates",
              "geometry_scope": "Row-L2 normalized centered CLS on permitted target_adapt cache; not the whole SSL source pool",
              "geometry": geometry, "gradient_batches": measurements,
              "config_sha256": digest(cfg_path), "split_hash": cfg.split_hash,
              "checkpoint_sha256": checkpoint_sha, "checkpoint_step": 6000,
              "public_anchor_sha256": protocol["encoders"]["S0"],
              "state_before": before, "state_after": after,
              "state_unchanged": before == after, "optimizer_steps": 0,
              "sampling_plan_sha256": cfg.sampling_plan_sha256,
              "unique_diagnostic_images": len(set(chosen)), "diagnostic_image_draws": 48,
              "weights": {"distill": levers.anchor_distill, "l2sp": levers.anchor_l2sp},
              "duration_seconds": time.monotonic() - started,
              "allocated_one_gpu_wall_seconds": elapsed_gpu, "diagnostic_gpu_hours": elapsed_gpu / 3600,
              "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
              "limits": ["Three batches at the final checkpoint are not a training trajectory or causal ablation",
                         "Eval and training BatchNorm modes use identical augmented views; BN buffers restored, all tensors hash-checked unchanged",
                         "Only trainable parameters in blocks9-12 measured; earlier trainable LayerNorm/head gradients excluded",
                         "Raw weighted gradient norms/cosines do not measure AdamW parameter updates or prove a distillation bottleneck"]}
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(destination), "state_unchanged": True,
                      "diagnostic_gpu_hours": report["diagnostic_gpu_hours"], "geometry": geometry}, indent=2))


if __name__ == "__main__":
    import probe_preprocessing_readout as contract
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=contract.ROOT)
    main(parser.parse_args().root)
