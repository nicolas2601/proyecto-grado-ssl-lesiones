from __future__ import annotations

import random

import numpy as np
import torch

from dermssl.pretrain import torch_resume, torch_simsiam


def _seed_for(seed: int, tag: str, index: int) -> int:
    tag_seed = sum((i + 1) * ord(char) for i, char in enumerate(tag))
    return (seed * 1_000_003 + tag_seed * 97 + index) % (2**32)


def _finite(tensors) -> None:
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("monitor received non-finite data or model outputs")


def _effective_rank(features: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(features, dim=1).double()
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if total <= torch.finfo(singular.dtype).eps:
        return 0.0
    probs = singular / total
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum()
    return float(entropy.exp().item())


def monitor_model(
    model: torch.nn.Module,
    datasets: dict,
    device: str | torch.device,
    seed: int = 20261024,
    batch_size: int = 16,
    max_images: int = 128,
) -> dict[str, dict[str, float | int]]:
    """Measure deterministic paired SSL scalars without changing training state."""
    if batch_size <= 0 or max_images <= 0:
        raise ValueError("batch_size and max_images must be positive")
    device = torch.device(device)
    rng_state = torch_resume.capture_rng()
    module_modes = [(module, module.training) for module in model.modules()]
    parameter_grads = [
        (parameter, None if parameter.grad is None else parameter.grad.clone())
        for parameter in model.parameters()
    ]
    results = {}
    try:
        for module, _ in module_modes:
            module.training = False
        model.eval()
        for tag, dataset in sorted(datasets.items()):
            count = min(len(dataset), max_images)
            if count == 0:
                raise ValueError(f"monitor dataset {tag!r} is empty")
            indices = list(range(count))
            loss_sum = 0.0
            image_count = 0
            feature_chunks = []
            cosine_sum = 0.0
            with torch.inference_mode():
                for start in range(0, count, batch_size):
                    batch_indices = indices[start:start + batch_size]
                    pairs = []
                    for index in batch_indices:
                        item_seed = _seed_for(seed, tag, index)
                        torch.manual_seed(item_seed)
                        random.seed(item_seed)
                        np.random.seed(item_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(item_seed)
                        pairs.append(dataset[index])
                    view1 = torch.stack([pair[0] for pair in pairs]).to(device)
                    view2 = torch.stack([pair[1] for pair in pairs]).to(device)
                    _finite((view1, view2))
                    autocast = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else torch.autocast("cpu", enabled=False)
                    )
                    with autocast:
                        p1, p2, z1, z2, h1, h2 = model(view1, view2, return_features=True)
                        loss = torch_simsiam.simsiam_loss(p1, p2, z1, z2)
                    _finite((p1, p2, z1, z2, h1, h2, loss))
                    batch_n = len(view1)
                    loss_sum += float(loss.float().item()) * batch_n
                    image_count += batch_n
                    feature_chunks.append(torch.cat((h1.float(), h2.float()), dim=0).cpu())
                    cosine_sum += float(
                        torch.nn.functional.cosine_similarity(
                            h1.float(), h2.float(), dim=-1
                        ).sum().item()
                    )
            features = torch.cat(feature_chunks, dim=0)
            normalized = torch.nn.functional.normalize(features, dim=1)
            std = float(normalized.std(dim=0, unbiased=False).mean().item())
            results[tag] = {
                "ssl_loss": loss_sum / image_count,
                "feature_std": std,
                "invariance_cos": cosine_sum / image_count,
                "effective_rank": _effective_rank(features),
                "n_images": image_count,
            }
    finally:
        for module, training in module_modes:
            module.training = training
        for parameter, grad in parameter_grads:
            parameter.grad = grad
        torch_resume.restore_rng(rng_state)
    return results
