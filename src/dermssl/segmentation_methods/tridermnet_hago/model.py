"""Own TriDermNet adaptation: paper region/edge/token/TER operations, no diagnosis."""
import torch
from torch import nn
from torch.nn import functional as F
from ..common.networks import image_batch


def block(incoming,outgoing,kernel=3,dilation=1,stride=1):
    return nn.Sequential(nn.Conv2d(incoming,outgoing,kernel,padding=dilation*(kernel//2),dilation=dilation,stride=stride),
                         nn.BatchNorm2d(outgoing),nn.ReLU())


def sobel(features):
    base=features.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    kernel=base[None,None].expand(features.shape[1],1,3,3)
    dx=F.conv2d(features,kernel,padding=1,groups=features.shape[1])
    dy=F.conv2d(features,kernel.transpose(-1,-2),padding=1,groups=features.shape[1])
    return torch.sqrt(dx.square()+dy.square()+1e-8)


class TriDermNet(nn.Module):
    def __init__(self,width=32,token_dim=64,patch_size=4,heads=4,max_token_side=28):
        super().__init__()
        if token_dim%heads or patch_size<1 or max_token_side<1: raise ValueError('Invalid token configuration')
        # Stride-2 stem + bounded token pooling are explicit resource adaptations.
        self.stem=block(3,width,stride=2)
        self.regions=nn.ModuleList(block(width,width,dilation=d) for d in (1,2,4))
        self.region_compress=block(3*width,width,kernel=1)
        self.edge=nn.Sequential(block(width,width),block(width,width)); self.edge_gate=nn.Conv2d(width,width,1)
        self.patch=nn.Conv2d(width,token_dim,patch_size,stride=patch_size)
        self.position=nn.Parameter(torch.zeros(1,token_dim,max_token_side,max_token_side))
        self.attention=nn.MultiheadAttention(token_dim,heads,batch_first=True)
        self.norm1=nn.LayerNorm(token_dim);self.norm2=nn.LayerNorm(token_dim)
        self.feedforward=nn.Sequential(nn.Linear(token_dim,4*token_dim),nn.GELU(),nn.Linear(4*token_dim,token_dim))
        self.spatial=block(2*width,width)
        self.gamma=nn.Linear(token_dim,width);self.beta=nn.Linear(token_dim,width)
        self.context=nn.Linear(width,token_dim)
        self.cross=nn.MultiheadAttention(token_dim,heads,batch_first=True)
        self.segmentation=nn.Conv2d(width,1,1);self.edge_head=nn.Conv2d(width,1,1)
        # Global coverage surrogate replaces diagnostic classification for W3.
        self.coverage=nn.Linear(token_dim,1);self.max_token_side=max_token_side
    def forward(self,images):
        image_batch(images); f=self.stem(images)
        region=f+self.region_compress(torch.cat([b(f) for b in self.regions],1))
        edge_raw=self.edge(sobel(f));edge=edge_raw*torch.sigmoid(self.edge_gate(edge_raw))
        patch=self.patch(f)
        if max(patch.shape[-2:])>self.max_token_side:
            patch=F.adaptive_avg_pool2d(patch,(self.max_token_side,self.max_token_side))
        pos=F.interpolate(self.position,size=patch.shape[-2:],mode='bilinear',align_corners=False)
        tokens=(patch+pos).flatten(2).transpose(1,2)
        attended,_=self.attention(tokens,tokens,tokens,need_weights=False)
        tokens=self.norm1(tokens+attended);tokens=self.norm2(tokens+self.feedforward(tokens))
        spatial=self.spatial(torch.cat((region,edge),1));c=tokens.mean(1)
        fused=self.gamma(c)[...,None,None]*spatial+self.beta(c)[...,None,None]
        v=self.context(spatial.mean((-2,-1))).unsqueeze(1)
        attended_spatial,_=self.cross(tokens,v,v,need_weights=False)
        # Singleton GAP key has softmax=1; retain token information explicitly.
        fused_tokens=tokens+attended_spatial
        size=images.shape[-2:]
        return {'logits':F.interpolate(self.segmentation(fused),size=size,mode='bilinear',align_corners=False),
                'edge_logits':F.interpolate(self.edge_head(edge),size=size,mode='bilinear',align_corners=False),
                'coverage_logits':self.coverage(fused_tokens.mean(1)), 'tokens':tokens}
