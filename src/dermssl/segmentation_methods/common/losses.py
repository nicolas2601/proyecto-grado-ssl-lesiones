"""Pseudo-label losses; ignore pixels never become background labels."""
import torch
from torch.nn import functional as F


def pseudo_valid(target, valid=None):
    if not torch.isfinite(target).all() or not ((target == 0) | (target == .5) | (target == 1)).all():
        raise ValueError('Targets must be pseudo-labels 0/.5/1')
    result = target != .5
    if valid is not None:
        if valid.shape != target.shape or not ((valid == 0) | (valid == 1)).all():
            raise ValueError('Matching binary valid tensor required')
        result &= valid.bool()
    if not result.any(): raise ValueError('All-uncertain batch must abstain')
    return result


def masked_bce(logits, target, valid=None):
    if logits.shape != target.shape or not torch.isfinite(logits).all():
        raise ValueError('Matching finite logits required')
    keep = pseudo_valid(target, valid)
    return F.binary_cross_entropy_with_logits(logits[keep], target[keep])


def masked_dice(logits, target, valid=None):
    keep = pseudo_valid(target, valid)
    if logits.shape != target.shape or not torch.isfinite(logits).all(): raise ValueError('Finite matching logits required')
    # Each eligible image contributes equally; fully uncertain images abstain.
    p = logits.sigmoid().flatten(1); y = target.flatten(1); k = keep.flatten(1)
    good = k.any(1); p = p[good]*k[good]; y = y[good]*k[good]
    return (1-(2*(p*y).sum(1)+1e-6)/(p.sum(1)+y.sum(1)+1e-6)).mean()
