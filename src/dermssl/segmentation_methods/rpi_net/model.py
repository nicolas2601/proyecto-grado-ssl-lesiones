"""Own RPI-Net IPC/PIA model; bounded token grid and declared pseudo encoder."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from ..common.networks import ResNet50Stages, Pyramid, image_batch


class CrossAttention(nn.Module):
    def __init__(self, width):
        super().__init__(); self.q=nn.Linear(width,width); self.k=nn.Linear(width,width); self.v=nn.Linear(width,width)
    def forward(self, query, context):
        q=self.q(query).unsqueeze(1); k=self.k(context).unsqueeze(1); v=self.v(context).unsqueeze(1)
        return F.scaled_dot_product_attention(q,k,v,dropout_p=0.).squeeze(1)


class PseudoInteraction(nn.Module):
    def __init__(self, width, sources=4):
        super().__init__(); self.sources=sources; self.ipc=CrossAttention(width); self.pia=CrossAttention(width)
    def forward(self, image_tokens, pseudo_tokens):
        if len(pseudo_tokens)!=self.sources: raise ValueError('Four RPI pseudo-label sources required')
        ipc=[self.ipc(image_tokens,p) for p in pseudo_tokens]
        # Eq.14: attend to each OTHER source, not only self-attention within i.
        return [ipc[i]+sum(self.pia(ipc[i],ipc[j]) for j in range(self.sources) if j!=i)
                for i in range(self.sources)]


class RPINet(nn.Module):
    def __init__(self, image_encoder=None, pseudo_encoder=None, width=256, token_grid=14):
        super().__init__()
        if token_grid < 1 or token_grid > 28: raise ValueError('Token grid must be 1..28; no full-resolution quadratic attention')
        self.image_encoder=image_encoder or ResNet50Stages()
        self.pseudo_encoder=pseudo_encoder or ResNet50Stages(in_channels=1)
        self.image_pyramid=Pyramid(self.image_encoder.channels,width)
        self.pseudo_pyramid=Pyramid(self.pseudo_encoder.channels,width)
        self.interaction=PseudoInteraction(width); self.segmentation=nn.Conv2d(width,1,1)
        self.uncertainty=nn.Conv2d(width,1,1); self.token_grid=token_grid
    def forward(self, images, pseudos=None):
        image_batch(images)
        if pseudos is None:
            feature=self.image_pyramid(self.image_encoder(images)[:3])
            return {'logits':F.interpolate(self.segmentation(feature),size=images.shape[-2:],mode='bilinear',align_corners=False)}
        if pseudos.shape!=(len(images),4,*images.shape[-2:]) or not torch.isfinite(pseudos).all() or ((pseudos<0)|(pseudos>1)).any():
            raise ValueError('Four finite pseudo-label maps in [0,1] aligned to image required')
        image_feature=self.image_pyramid(self.image_encoder(images)[:3])
        n=len(images); size=images.shape[-2:]; grid=(self.token_grid,self.token_grid)
        flat_pseudo=pseudos.reshape(n*4,1,*size)
        pf=self.pseudo_pyramid(self.pseudo_encoder(flat_pseudo)[:3])
        pf=F.adaptive_avg_pool2d(pf,grid).reshape(n,4,-1,*grid)
        img_tokens=F.adaptive_avg_pool2d(image_feature,grid).flatten(2).transpose(1,2)
        pt=[pf[:,i].flatten(2).transpose(1,2) for i in range(4)]
        refined=self.interaction(img_tokens,pt)
        sigmas=[self.uncertainty(t.transpose(1,2).reshape(n,-1,*grid)) for t in refined]
        sigma=F.interpolate(torch.cat(sigmas,1),size=size,mode='bilinear',align_corners=False)
        # Eq.17: mu is from IMAGE branch, never IPC/PIA output.
        logits=F.interpolate(self.segmentation(image_feature),size=size,mode='bilinear',align_corners=False)
        return {'logits':logits,'sigma':sigma}
