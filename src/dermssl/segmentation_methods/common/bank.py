"""Source encoders constructed locally and populated only by reviewed checkpoints."""
import json
import torch
from torch import nn
from pathlib import Path
from .networks import ResNet50Stages
from .dino import PatchTransformer
from .checkpoints import load_reviewed


def source_bank(path,origins):
    document=json.loads(Path(path).read_text())
    if set(document.get('sources',{}))!=set(origins):raise ValueError('Source bank must match the prescribed SSL origins exactly')
    result={}
    for origin in origins:
        source=document['sources'][origin]
        if origin=='dino':model=PatchTransformer(**source.get('architecture',{}))
        else:model=ResNet50Stages(last_stride=1)
        provenance=load_reviewed(model,source['checkpoint'],source['provenance'],expected_method='ssl_'+origin)
        normalization=provenance.get('input_normalization')
        if normalization not in {'unit_rgb','imagenet'}:raise ValueError('Source checkpoint input normalization must be explicit')
        if normalization=='imagenet':model=NormalizedEncoder(model)
        model.eval();result[origin]=model
    return result


class NormalizedEncoder(nn.Module):
    """Apply declared external SSL normalization identically to shallow/deep paths."""
    def __init__(self,base):
        super().__init__();self.base=base
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def forward(self,x):return self.base((x-self.mean)/self.std)
    def shallow_features(self,x):return self.base.shallow_features((x-self.mean)/self.std)
