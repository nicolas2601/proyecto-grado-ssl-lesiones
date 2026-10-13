"""Own configurable ViT + DINO-style EMA distillation for label-free feature banks.

This supplies a local-pretraining alternative; it does not pretend to load the
unspecified original RPI DINO checkpoint or reproduce its pretraining budget.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class PatchTransformer(nn.Module):
    def __init__(self,dimension=384,depth=12,heads=6,patch_size=16,image_size=224):
        super().__init__()
        if dimension%heads or image_size%patch_size:raise ValueError('ViT patch/head dimensions incompatible')
        self.patch=nn.Conv2d(3,dimension,patch_size,stride=patch_size)
        self.class_token=nn.Parameter(torch.zeros(1,1,dimension));side=image_size//patch_size
        self.position=nn.Parameter(torch.zeros(1,dimension,side,side));self.class_position=nn.Parameter(torch.zeros(1,1,dimension))
        layer=nn.TransformerEncoderLayer(dimension,heads,4*dimension,dropout=0.,activation='gelu',batch_first=True,norm_first=True,layer_norm_eps=1e-6)
        self.transformer=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False);self.norm=nn.LayerNorm(dimension,eps=1e-6)
    def tokens(self,images):
        feature=self.patch(images);n,c,h,w=feature.shape
        pos=F.interpolate(self.position,size=(h,w),mode='bicubic',align_corners=False)
        sequence=(feature+pos).flatten(2).transpose(1,2)
        cls=self.class_token.expand(n,-1,-1)+self.class_position
        return self.norm(self.transformer(torch.cat((cls,sequence),1))),h,w
    def forward(self,images):
        tokens,h,w=self.tokens(images)
        return tokens[:,1:].transpose(1,2).reshape(len(images),-1,h,w)
    def global_feature(self,images):return self.tokens(images)[0][:,0]


class DinoDistillation(nn.Module):
    def __init__(self,backbone=None,dimension=384,out_dim=1024,momentum=.996,center_momentum=.9,
                 student_temperature=.1,teacher_temperature=.04):
        super().__init__()
        if not 0<momentum<1 or not 0<center_momentum<1 or min(student_temperature,teacher_temperature)<=0:
            raise ValueError('Invalid distillation controls')
        self.backbone=backbone or PatchTransformer(dimension=dimension)
        self.head=nn.Sequential(nn.Linear(dimension,dimension),nn.GELU(),nn.Linear(dimension,out_dim))
        self.teacher_backbone=copy.deepcopy(self.backbone);self.teacher_head=copy.deepcopy(self.head)
        for module in (self.teacher_backbone,self.teacher_head):
            for p in module.parameters():p.requires_grad_(False)
        self.register_buffer('center',torch.zeros(1,out_dim))
        self.momentum=momentum;self.center_momentum=center_momentum;self.st=student_temperature;self.tt=teacher_temperature
    def forward(self,first,second):
        s1=self.head(self.backbone.global_feature(first));s2=self.head(self.backbone.global_feature(second))
        with torch.no_grad():
            for online,teacher in [(self.backbone,self.teacher_backbone),(self.head,self.teacher_head)]:
                for a,b in zip(online.parameters(),teacher.parameters(),strict=True):b.mul_(self.momentum).add_(a,alpha=1-self.momentum)
            self.teacher_backbone.eval();self.teacher_head.eval()
            t1=self.teacher_head(self.teacher_backbone.global_feature(first));t2=self.teacher_head(self.teacher_backbone.global_feature(second))
            p1=F.softmax((t1-self.center)/self.tt,dim=1);p2=F.softmax((t2-self.center)/self.tt,dim=1)
            batch_center=torch.cat((t1,t2)).mean(0,keepdim=True)
            self.center.mul_(self.center_momentum).add_(batch_center,alpha=1-self.center_momentum)
        return -.5*((p1*F.log_softmax(s2/self.st,dim=1)).sum(1).mean()+
                    (p2*F.log_softmax(s1/self.st,dim=1)).sum(1).mean())
