"""MULAN alignment (arXiv 2602.05845, eq. 1): one predictor per view type, shared target, per-view weights."""
import torch
import torch.nn.functional as F


def mulan_alignment(predictions, target, weights):
    missing = set(predictions) - set(weights)
    if missing:
        raise ValueError(f'missing weight for view types: {sorted(missing)}')
    target = F.normalize(target, dim=-1).detach()
    total = 0.0
    for view, pred in predictions.items():
        pred = F.normalize(pred, dim=-1)
        total = total + weights[view] * (pred - target).pow(2).sum(-1).mean()
    return total
