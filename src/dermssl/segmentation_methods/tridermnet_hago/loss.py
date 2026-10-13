"""Pseudo-supervised TriDerm losses; expert masks/diagnostic labels not accepted."""
import torch
from torch.nn import functional as F
from ..common.losses import masked_bce,masked_dice,pseudo_valid


def losses(outputs,target,valid=None,dice_weight=1.,bce_weight=.5,edge_weight=.3,coverage_weight=.2):
    keep=pseudo_valid(target,valid)
    p=outputs['logits'].sigmoid()
    segmentation=dice_weight*masked_dice(outputs['logits'],target,valid)+bce_weight*masked_bce(outputs['logits'],target,valid)
    # Supervise edges only where the whole 3x3 neighborhood is confident/valid.
    eroded=F.avg_pool2d(keep.float(),3,stride=1,padding=1)==1
    fill=torch.where(keep,target,torch.zeros_like(target))
    boundary=F.max_pool2d(fill,3,stride=1,padding=1)-(-F.max_pool2d(-fill,3,stride=1,padding=1))
    boundary=(boundary>0).float()
    if eroded.any():
        edge=F.binary_cross_entropy_with_logits(outputs['edge_logits'][eroded],boundary[eroded])
    else: edge=outputs['edge_logits'].sum()*0
    coverage_target=(fill*keep).sum((-2,-1))/keep.sum((-2,-1)).clamp_min(1)
    eligible=keep.flatten(1).any(1)
    predicted=outputs['coverage_logits'].sigmoid()
    coverage=F.mse_loss(predicted[eligible],coverage_target[eligible])
    area=(p*keep).sum((-2,-1))/keep.sum((-2,-1)).clamp_min(1)
    consistency=F.l1_loss(predicted[eligible],area[eligible])
    return {'segmentation':segmentation+edge_weight*edge,
            'coverage':coverage_weight*(coverage+consistency),
            'edge':edge.detach(),'consistency':consistency.detach()}
