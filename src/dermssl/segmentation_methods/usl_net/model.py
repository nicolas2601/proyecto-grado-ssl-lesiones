"""USL-Net text adaptation: separate contrastive fusion/CCAM and FCN student."""
import torch
from torch import nn
from torch.nn import functional as F
from dermssl.w3_components import USLTextPolicy, usl_text_pseudolabel
from dermssl.w3_usl_models import USLTextCCAM, USLTextFCNDecoder
from ..common.networks import ResNet50Stages, image_batch
from ..common.losses import masked_bce


CCAM_STABILIZED_PROFILE = 'ccam_stabilized_v1'


class ContrastiveFusion(nn.Module):
    def __init__(self, encoders=None, *, profile='legacy_v8', rms_epsilon=1e-6):
        super().__init__()
        if profile not in {'legacy_v8', CCAM_STABILIZED_PROFILE}:
            raise ValueError(f'Unknown USL fusion profile: {profile}')
        if rms_epsilon <= 0:
            raise ValueError('Fusion RMS epsilon must be positive')
        self.profile = profile
        self.rms_epsilon = float(rms_epsilon)
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
        encoders=(self.encoders[origin] for origin in ('simclr','mocov1','mocov2')) if self.profile == CCAM_STABILIZED_PROFILE else self.encoders.values()
        for encoder in encoders:
            encoder.eval()
            with torch.no_grad(): stages = encoder(images)
            size = stages[-2].shape[-2:]
            maps.extend((stages[-2], F.interpolate(stages[-1],size=size,mode='bilinear',align_corners=False)))
        if self.profile == CCAM_STABILIZED_PROFILE:
            normalized = []
            for feature_map in maps:
                value = feature_map.float()
                rms = value.square().mean(dim=(1, 2, 3), keepdim=True).add(self.rms_epsilon).sqrt()
                normalized.append(value / rms)
            maps = normalized
        return torch.cat(maps,1)


class USLSeedGenerator(nn.Module):
    def __init__(self, fusion=None, channels=9216, *, profile='legacy_v8'):
        super().__init__()
        self.profile = profile
        self.fusion = fusion or ContrastiveFusion(profile=profile)
        if self.fusion.profile != profile:
            raise ValueError('USL seed and fusion profiles must match')
        self.ccam = USLTextCCAM(channels, normalize_logits=profile == CCAM_STABILIZED_PROFILE)
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
