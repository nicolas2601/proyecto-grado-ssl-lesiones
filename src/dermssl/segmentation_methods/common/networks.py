"""Own standard ResNet50 stages; no torchvision imports or weight downloads."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, incoming, planes, stride=1):
        super().__init__(); outgoing = 4 * planes
        self.body = nn.Sequential(nn.Conv2d(incoming, planes, 1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(),
                                  nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(),
                                  nn.Conv2d(planes, outgoing, 1, bias=False), nn.BatchNorm2d(outgoing))
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False), nn.BatchNorm2d(outgoing))
    def forward(self, x): return F.relu(self.body(x) + self.skip(x))


class ResNet50Stages(nn.Module):
    channels = (256, 512, 1024)
    def __init__(self, in_channels=3, last_stride=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.stages = nn.ModuleList(); incoming = 64
        for planes, blocks, stride in zip((64, 128, 256, 512), (3, 4, 6, 3), (1, 2, 2, last_stride)):
            layer = [Bottleneck(incoming, planes, stride)]
            layer += [Bottleneck(4*planes, planes) for _ in range(blocks-1)]
            self.stages.append(nn.Sequential(*layer)); incoming = 4*planes
    def shallow_features(self,images):return self.stem[0](images)
    def forward(self, x):
        x = self.stem(x); outputs = []
        for stage in self.stages: x = stage(x); outputs.append(x)
        return tuple(outputs)


class Pyramid(nn.Module):
    def __init__(self, channels, width):
        super().__init__(); self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
    def forward(self, features):
        size = features[0].shape[-2:]
        return sum(F.interpolate(layer(x), size=size, mode='bilinear', align_corners=False)
                   for layer, x in zip(self.lateral, features, strict=True))


def image_batch(x):
    if x.ndim != 4 or x.shape[1] != 3 or not torch.isfinite(x).all():
        raise ValueError('Finite BCHW RGB batch required')
