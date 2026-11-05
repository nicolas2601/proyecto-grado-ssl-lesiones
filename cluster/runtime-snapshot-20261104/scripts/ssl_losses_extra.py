"""Extra SSL objectives on projector outputs (SIGReg/LeJEPA-style, TiCo-style, SwAV). Installed via patch(new=) only.

Declared approximations (see PENDIENTES.md): these follow the published idea, not a verified reimplementation
of each paper's exact constants. Each arm states its constants in the ledger before running.
"""
import torch
import torch.nn.functional as F

SIGREG_T = torch.linspace(0.2, 3.0, 17)


def sigreg_loss(z, num_projections=64):
    """Epps-Pulley style test: projections of z along random unit directions should look standard normal."""
    d = z.shape[1]
    a = F.normalize(torch.randn(d, num_projections, device=z.device, dtype=z.dtype), dim=0)
    h = z @ a  # (N, M)
    t = SIGREG_T.to(z.device, z.dtype)
    phase = h.unsqueeze(-1) * t  # (N, M, T)
    re, im = torch.cos(phase).mean(0), torch.sin(phase).mean(0)  # empirical characteristic function
    target = torch.exp(-0.5 * t * t)  # standard normal, real-valued
    stat = (re - target) ** 2 + im ** 2
    weight = torch.exp(-0.5 * t * t)
    return (stat * weight).sum(-1).mean()


class TiCoLoss(torch.nn.Module):
    """Invariance (MSE between views) plus an off-diagonal covariance penalty; the covariance is EMA-tracked."""

    def __init__(self, dim, momentum=0.99, lam=1.0):
        super().__init__()
        self.register_buffer('cov', torch.eye(dim))
        self.momentum = momentum
        self.lam = lam

    def forward(self, z1, z2):
        invariance = F.mse_loss(z1, z2)
        z = torch.cat([z1, z2], 0)
        centred = z - z.mean(0)
        batch_cov = centred.T @ centred / (z.shape[0] - 1)
        with torch.no_grad():
            self.cov.mul_(self.momentum).add_(batch_cov.detach(), alpha=1 - self.momentum)
        off = batch_cov - torch.diag(torch.diagonal(batch_cov))
        return invariance + self.lam * off.pow(2).sum() / z.shape[1]


def sinkhorn(scores, epsilon=0.05, iterations=3):
    """SwAV Sinkhorn-Knopp: balanced soft assignments with rows summing to 1 and equal prototype mass."""
    batch, protos = scores.shape
    scores = scores - scores.max(dim=1, keepdim=True).values  # row shift leaves assignments unchanged
    q = torch.exp(scores / epsilon).T  # (K, B)
    q = q / q.sum()
    for _ in range(iterations):
        q = q / q.sum(1, keepdim=True) / protos
        q = q / q.sum(0, keepdim=True) / batch
    q = (q * batch).T  # (B, K): rows sum to 1, columns sum to B / K
    return q


def swav_loss(z1, z2, prototypes, temperature=0.1, epsilon=0.05):
    s1, s2 = z1 @ prototypes.T, z2 @ prototypes.T
    with torch.no_grad():
        q1, q2 = sinkhorn(s1, epsilon), sinkhorn(s2, epsilon)
    loss = -(q1 * F.log_softmax(s2 / temperature, dim=1)).sum(1).mean()
    loss = loss - (q2 * F.log_softmax(s1 / temperature, dim=1)).sum(1).mean()
    return loss / 2
