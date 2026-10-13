"""USL-Net text adaptation: separate contrastive fusion/CCAM and FCN student."""
import torch
from torch import nn
from torch.nn import functional as F
from dermssl.w3_components import USLTextPolicy, usl_text_pseudolabel
from dermssl.w3_usl_models import USLTextCCAM, USLTextFCNDecoder
from ..common.networks import ResNet50Stages, image_batch
from ..common.losses import masked_bce


class ContrastiveFusion(nn.Module):
    def __init__(self, encoders=None):
        super().__init__()
        encoders = encoders or {k:ResNet50Stages(last_stride=1) for k in ('simclr','mocov1','mocov2')}
        if set(encoders) != {'simclr','mocov1','mocov2'}: raise ValueError('Three distinct contrastive origins required')
        self.encoders = nn.ModuleDict(encoders)
        for p in self.encoders.parameters(): p.requires_grad_(False)
    def train(self, mode=True):
        super().train(mode)
        for encoder in self.encoders.values(): encoder.eval()
        return self
    def forward(self, images):
        maps = []
        for encoder in self.encoders.values():
            encoder.eval()
            with torch.no_grad(): stages = encoder(images)
            size = stages[-2].shape[-2:]
            maps.extend((stages[-2], F.interpolate(stages[-1],size=size,mode='bilinear',align_corners=False)))
        return torch.cat(maps,1)


class USLSeedGenerator(nn.Module):
    def __init__(self, fusion=None, channels=9216):
        super().__init__(); self.fusion = fusion or ContrastiveFusion(); self.ccam = USLTextCCAM(channels)
    def forward(self, images):
        image_batch(images); return self.ccam(self.fusion(images), images.shape[-2:])
    def pseudo(self, images, policy=USLTextPolicy()):
        saliency = self(images).detach().cpu().numpy()[:,0]
        return [usl_text_pseudolabel(s,policy) for s in saliency]


class USLStudent(nn.Module):
    def __init__(self, encoder=None):
        super().__init__(); self.encoder = encoder or ResNet50Stages()
        self.decoder = USLTextFCNDecoder(tuple(self.encoder.channels))
    def forward(self, images):
        image_batch(images)
        return {'logits':self.decoder(self.encoder(images)[:3],images.shape[-2:])}
    @staticmethod
    def loss(outputs, targets, valid=None): return masked_bce(outputs['logits'],targets,valid)
