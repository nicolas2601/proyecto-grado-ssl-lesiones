"""SimSiam + out-of-batch SIGReg anchors (arXiv 2610.20158, adapted). Installed with patch.object(..., new=...) only."""
from PIL import Image
import torch
from torchvision import transforms as T

from oob_sigreg import gaussian_quantile_targets, oob_sigreg_loss

PROJECTIONS = 256
_PREP = T.Compose([T.Resize((224, 224), antialias=True), T.ToTensor(),
                   T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def make_oob_simsiam(base):
    class OOBSimSiam(base):
        def forward(self, view1, view2, return_features=False, tone_view=None):
            if tone_view is not None:
                raise ValueError('OOB arm does not combine with the tone view')
            h1, h2 = self.backbone(view1), self.backbone(view2)
            z1, z2 = self.projector(h1), self.projector(h2)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            self._last_z = (z1, z2)
            if return_features:
                return p1, p2, z1.detach(), z2.detach(), h1, h2
            return p1, p2, z1.detach(), z2.detach()
    return OOBSimSiam


def make_oob_lever(original, holder, lambda_ac, batch_indices, refresh=None):
    def lever(base_loss, levers, anchors, model, outputs, v1):
        loss, parts = original(base_loss, levers, anchors, model, outputs, v1)
        if refresh is not None:
            refresh(model)
        index = torch.as_tensor(batch_indices(), dtype=torch.long, device=holder['targets'].device)
        target = holder['targets'][index]
        z1, z2 = model._last_z
        term = 0.5 * (oob_sigreg_loss(z1, holder['directions'], target)
                      + oob_sigreg_loss(z2, holder['directions'], target))
        return loss + lambda_ac * term, {**parts, 'oob': term.detach()}
    return lever


def refresh_anchors(model, rows, holder, generator, device, batch=64):
    dim = model.projector[-2].out_features
    directions = torch.randn(dim, PROJECTIONS, generator=generator)
    directions = (directions / directions.norm(dim=0, keepdim=True)).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        chunks = []
        for start in range(0, len(rows), batch):
            images = torch.stack([_PREP(Image.open(r['image_path']).convert('RGB'))
                                  for r in rows[start:start + batch]]).to(device)
            chunks.append(model.projector(model.backbone(images)).float())
        projected = (torch.cat(chunks).float() @ directions.float()).cpu().numpy()
    model.train(was_training)
    holder['directions'] = directions
    holder['targets'] = torch.as_tensor(gaussian_quantile_targets(projected), dtype=torch.float32, device=device)


class RefreshSchedule:
    """Refresh before the first step and then every ``period`` optimizer steps (paper: 0.5 epoch)."""

    def __init__(self, period, action):
        self.period, self.action, self.calls = period, action, 0

    def __call__(self, model):
        if self.calls % self.period == 0:
            self.action(model)
        self.calls += 1
