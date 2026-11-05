"""BYOL-style SimSiam: online predictor against an EMA target branch. Installed with patch.object(..., new=...) only."""
import copy

import torch

from ssl_objectives import ema_update


def make_byol_simsiam(base, momentum=0.996):
    class BYOLSimSiam(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_backbone = copy.deepcopy(self.backbone).requires_grad_(False)
            self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)

        def forward(self, view1, view2, return_features=False, tone_view=None):
            if tone_view is not None:
                raise ValueError('BYOL arm does not combine with the tone view')
            ema_update(self.target_backbone, self.backbone, momentum)
            ema_update(self.target_projector, self.projector, momentum)
            h1, h2 = self.backbone(view1), self.backbone(view2)
            z1, z2 = self.projector(h1), self.projector(h2)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            with torch.no_grad():
                t1 = self.target_projector(self.target_backbone(view1))
                t2 = self.target_projector(self.target_backbone(view2))
            if return_features:
                return p1, p2, t1, t2, h1, h2
            return p1, p2, t1, t2
    return BYOLSimSiam
