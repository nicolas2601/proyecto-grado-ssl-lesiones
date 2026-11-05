"""Out-of-batch SIGReg anti-collapse (arXiv 2610.20158, eqs. 6-7), adapted as an additive term on SimSiam projector outputs."""
import numpy as np
from scipy.stats import norm
import torch


def gaussian_quantile_targets(values):
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[0]
    if count < 2:
        raise ValueError('OOB targets need at least two samples')
    ranks = np.argsort(np.argsort(values, axis=0), axis=0)
    return norm.ppf((ranks + 0.5) / count)


def oob_sigreg_loss(z, directions, targets):
    projected = z @ directions
    return ((projected - targets) ** 2).mean()
