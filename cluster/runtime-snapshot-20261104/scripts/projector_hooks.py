"""Barlow Twins / VICReg on SimSiam projector outputs (no predictor). Installed via patch.object(..., new=...)."""
import torch
import torch.nn.functional as F

from ssl_objectives import barlow_loss, vicreg_loss

def barlow_vicreg_loss(z1, z2):
    return barlow_loss(z1, z2) + vicreg_loss(z1, z2)


from ssl_losses_extra import TiCoLoss, sigreg_loss


def sigreg_objective(z1, z2):
    """LeJEPA-style: invariance MSE plus SIGReg on both views (lambda = 1.0, 64 projections)."""
    return F.mse_loss(z1, z2) + sigreg_loss(z1, num_projections=64) + sigreg_loss(z2, num_projections=64)


_TICO_STATE = {}


def tico_objective(z1, z2):
    """TiCo-style (approximation, see ssl_losses_extra): EMA covariance state per process."""
    if 'loss' not in _TICO_STATE:
        _TICO_STATE['loss'] = TiCoLoss(z1.shape[1]).to(z1.device)
    return _TICO_STATE['loss'](z1, z2)


OBJECTIVES = {'barlow': barlow_loss, 'vicreg': vicreg_loss, 'barlow_vicreg': barlow_vicreg_loss,
              'sigreg': sigreg_objective, 'tico': tico_objective}


def make_projector_simsiam(base):
    class ProjectorSimSiam(base):
        def forward(self, view1, view2, return_features=False, tone_view=None):
            if tone_view is not None:
                raise ValueError('projector arms do not combine with the tone view')
            h1, h2 = self.backbone(view1), self.backbone(view2)
            z1, z2 = self.projector(h1), self.projector(h2)
            if return_features:
                return z1, z2, z1.detach(), z2.detach(), h1, h2
            return z1, z2, z1.detach(), z2.detach()
    return ProjectorSimSiam


def make_projector_loss(name):
    if name not in OBJECTIVES:
        raise ValueError(f'unknown projector objective: {name}')
    objective = OBJECTIVES[name]

    def loss(z1, z2, _unused1, _unused2):
        return objective(z1, z2)
    return loss
