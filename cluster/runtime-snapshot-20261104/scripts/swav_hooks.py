"""SwAV-style head (estilo SwAV: aproximación declarada en ledger).

Projection to proj_dim, L2-normalised features, trainable L2-normalised prototypes, swapped-prediction loss
with Sinkhorn assignments from ssl_losses_extra.swav_loss. No momentum teacher.
"""
import torch
import torch.nn.functional as F

from ssl_losses_extra import swav_loss


class SwAVHead(torch.nn.Module):
    def __init__(self, in_dim, proj_dim, prototypes, temperature=0.1, epsilon=0.05):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, proj_dim)
        self.prototypes = torch.nn.Parameter(torch.randn(prototypes, proj_dim) * 0.02)
        self.temperature = temperature
        self.epsilon = epsilon

    def loss(self, f1, f2):
        z1 = F.normalize(self.proj(f1), dim=-1)
        z2 = F.normalize(self.proj(f2), dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)
        return swav_loss(z1, z2, protos, temperature=self.temperature, epsilon=self.epsilon)
