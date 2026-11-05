"""Non-contrastive objectives on projector outputs (BYOL EMA target, Barlow Twins, VICReg). Installed via patch(new=) only."""
import torch
import torch.nn.functional as F

BARLOW_LAMBDA = 0.0051
VICREG_INV, VICREG_VAR, VICREG_COV = 25.0, 25.0, 1.0


@torch.no_grad()
def ema_update(target, online, momentum):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)


def barlow_loss(z1, z2):
    def standard(z):
        return (z - z.mean(0)) / (z.std(0) + 1e-6)
    a, b = standard(z1), standard(z2)
    c = a.T @ b / a.shape[0]
    on_diagonal = (torch.diagonal(c) - 1).pow(2).sum()
    off = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on_diagonal + BARLOW_LAMBDA * off


def _variance_hinge(z):
    return F.relu(1.0 - torch.sqrt(z.var(0) + 1e-4)).mean()


def _off_diagonal_covariance(z):
    centred = z - z.mean(0)
    cov = centred.T @ centred / (z.shape[0] - 1)
    off = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
    return off / z.shape[1]


def vicreg_loss(z1, z2):
    invariance = F.mse_loss(z1, z2)
    variance = _variance_hinge(z1) + _variance_hinge(z2)
    covariance = _off_diagonal_covariance(z1) + _off_diagonal_covariance(z2)
    return VICREG_INV * invariance + VICREG_VAR * variance + VICREG_COV * covariance
