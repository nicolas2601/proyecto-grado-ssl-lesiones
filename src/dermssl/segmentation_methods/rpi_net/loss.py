"""RPI DUL equations 20-23 with explicit log-variance rails and alpha convention."""
import torch
from torch.nn import functional as F


def dynamic_uncertainty_loss(logits, pseudos, sigma, valid=None, *, detach_alpha=True, sigma_bounds=(-8.,12.)):
    if logits.ndim!=4 or logits.shape[1]!=1 or pseudos.shape!=sigma.shape or pseudos.shape[1]!=4:
        raise ValueError('B1HW segmentation and matching B4HW pseudo/sigma maps required')
    if pseudos.shape[0]!=logits.shape[0] or pseudos.shape[-2:]!=logits.shape[-2:]: raise ValueError('Aligned RPI geometry required')
    if not all(torch.isfinite(x).all() for x in (logits,pseudos,sigma)) or ((pseudos<0)|(pseudos>1)).any():
        raise ValueError('Finite logits/sigma and pseudo targets in [0,1] required')
    if len(sigma_bounds)!=2 or not sigma_bounds[0]<sigma_bounds[1]: raise ValueError('Ordered sigma bounds required')
    if valid is None: valid=torch.ones_like(logits,dtype=torch.bool)
    if valid.shape!=logits.shape or not ((valid==0)|(valid==1)).all(): raise ValueError('Matching binary valid mask required')
    mask=valid.bool().expand_as(pseudos); count=mask.sum((-2,-1))
    if (count==0).any(): raise ValueError('Every RPI sample needs valid pixels')
    s=sigma.clamp(*sigma_bounds)
    entropy=F.softplus(s).square()  # author's defined function, not Bernoulli entropy
    average=(entropy*mask).sum((-2,-1))/count
    alpha=(average+1e-8).reciprocal()
    if detach_alpha: alpha=alpha.detach()
    weights=alpha/alpha.sum(1,keepdim=True)
    bce=F.binary_cross_entropy_with_logits(logits.expand_as(pseudos),pseudos,reduction='none')
    terms=bce*torch.exp(-s)+.5*s
    losses=(terms*mask).sum((-2,-1))/count
    loss=(weights*losses).sum(1).mean()
    return loss, {'alpha_normalized':weights.detach(), 'sigma_clipped_fraction':(s!=sigma).float().mean().detach()}
