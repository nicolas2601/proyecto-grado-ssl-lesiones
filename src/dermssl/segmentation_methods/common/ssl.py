"""Own label-free contrastive pretraining building blocks: SimCLR/MoCov1/v2."""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .networks import ResNet50Stages


class ContrastiveEncoder(nn.Module):
    def __init__(self, variant='simclr', projection_dim=128, backbone=None, pooled_channels=2048):
        super().__init__()
        if variant not in {'simclr','mocov1','mocov2'}: raise ValueError('Unknown contrastive variant')
        self.backbone = backbone or ResNet50Stages(last_stride=1)
        self.projector = nn.Linear(pooled_channels, projection_dim) if variant == 'mocov1' else nn.Sequential(
            nn.Linear(pooled_channels, pooled_channels), nn.ReLU(), nn.Linear(pooled_channels, projection_dim))
    def forward(self, images):
        features = self.backbone(images)[-1]
        return F.normalize(self.projector(features.mean((-2,-1))), dim=1)


def simclr_loss(first, second, temperature=.07):
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2 or temperature <= 0:
        raise ValueError('Matching embeddings with at least two pairs required')
    z = F.normalize(torch.cat((first, second)), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(float('-inf'))
    target = (torch.arange(len(z), device=z.device) + len(first)) % len(z)
    return F.cross_entropy(logits, target)


class MomentumContrast(nn.Module):
    def __init__(self, encoder, dimension=128, queue_size=65536, momentum=.999, temperature=.07):
        super().__init__()
        if queue_size <= 0 or not 0 < momentum < 1 or temperature <= 0: raise ValueError('Invalid MoCo parameters')
        self.online = encoder; self.key = copy.deepcopy(encoder)
        for p in self.key.parameters(): p.requires_grad_(False)
        self.register_buffer('queue', F.normalize(torch.randn(dimension, queue_size),dim=0))
        self.register_buffer('pointer', torch.zeros((), dtype=torch.long))
        self.momentum = momentum; self.temperature = temperature
    def forward(self, first, second):
        q = self.online(first)
        with torch.no_grad():
            for online, key in zip(self.online.parameters(), self.key.parameters(), strict=True):
                key.mul_(self.momentum).add_(online, alpha=1-self.momentum)
            for online, key in zip(self.online.buffers(), self.key.buffers(), strict=True): key.copy_(online)
            self.key.eval(); k = self.key(second)
        positives = (q*k).sum(1,keepdim=True)
        # Clone negatives because enqueue modifies the buffer before backward.
        negatives = q @ self.queue.detach().clone()
        loss = F.cross_entropy(torch.cat((positives,negatives),1)/self.temperature,
                               torch.zeros(len(q),dtype=torch.long,device=q.device))
        with torch.no_grad():
            for vector in k:
                self.queue[:, int(self.pointer)] = vector
                self.pointer.copy_((self.pointer+1) % self.queue.shape[1])
        return loss


class MoCoV3(nn.Module):
    """Declared local v3 adaptation: EMA encoder, predictor, symmetric batch negatives."""
    def __init__(self,encoder,dimension=128,momentum=.99,temperature=.2):
        super().__init__();self.online=encoder;self.key=copy.deepcopy(encoder)
        self.predictor=nn.Sequential(nn.Linear(dimension,dimension),nn.ReLU(),nn.Linear(dimension,dimension))
        for p in self.key.parameters():p.requires_grad_(False)
        self.momentum=momentum;self.temperature=temperature
    def forward(self,first,second):
        if len(first)<2:raise ValueError('MoCov3 local batch needs at least two images')
        q1=F.normalize(self.predictor(self.online(first)),dim=1);q2=F.normalize(self.predictor(self.online(second)),dim=1)
        with torch.no_grad():
            for a,b in zip(self.online.parameters(),self.key.parameters(),strict=True):b.mul_(self.momentum).add_(a,alpha=1-self.momentum)
            for a,b in zip(self.online.buffers(),self.key.buffers(),strict=True):b.copy_(a)
            self.key.eval();k1=self.key(first);k2=self.key(second)
        target=torch.arange(len(first),device=first.device)
        return self.temperature*(F.cross_entropy(q1@k2.T/self.temperature,target)+F.cross_entropy(q2@k1.T/self.temperature,target))
