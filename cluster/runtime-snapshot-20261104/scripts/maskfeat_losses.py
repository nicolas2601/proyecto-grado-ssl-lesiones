"""MaskFeat-style regression to frozen DINOv2 patch features (estilo MaskFeat: aproximación declarada).

Cosine regression of predictions at masked patch positions onto the frozen teacher's patch features.
"""
import torch
import torch.nn.functional as F


def maskfeat_loss(pred, target, mask):
    if not bool(mask.any()):
        raise ValueError('MaskFeat needs at least one masked patch position')
    cos = F.cosine_similarity(pred, target.detach(), dim=-1)
    return (1 - cos)[mask].mean()
