"""Source-bank integration for RPI's four pseudo-label generators."""
from torch import nn
import numpy as np
from .pseudolabels import four_sources,gradcampp_energy


class RPISeedGenerator(nn.Module):
    def __init__(self,mocov3,mocov2,dino):
        super().__init__();self.mocov3=mocov3;self.mocov2=mocov2;self.dino=dino
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,images,seed=42):
        # Required gradient is only for class-agnostic CAM, no diagnostic head.
        image=images.detach().clone().requires_grad_(True)
        self.mocov3.eval();self.mocov2.eval();self.dino.eval()
        shallow3=self.mocov3.shallow_features(image)
        deep3=self.mocov3(image)[-1]
        cam=gradcampp_energy(deep3,images.shape[-2:]).cpu().numpy()[:,0]
        shallow2=self.mocov2.shallow_features(image).detach().cpu().numpy().transpose(0,2,3,1)
        dino=self.dino(image).detach().cpu().numpy().transpose(0,2,3,1)
        shallow3=shallow3.detach().cpu().numpy().transpose(0,2,3,1)
        return np.stack([four_sources(shallow3[i],cam[i],shallow2[i],dino[i],images.shape[-2:],seed)
                         for i in range(len(images))])
