import unittest
import torch
from torch import nn
from dermssl.segmentation_methods.usl_net.model import USLStudent,ContrastiveFusion,USLSeedGenerator
from dermssl.segmentation_methods.usl_net.ccam import existing_ccam_loss
from dermssl.segmentation_methods.common.ssl import simclr_loss,ContrastiveEncoder,MomentumContrast,MoCoV3
from dermssl.segmentation_methods.common.dino import PatchTransformer,DinoDistillation


class TinyEncoder(nn.Module):
    channels=(4,8,16)
    def __init__(self,in_channels=3):
        super().__init__();self.layers=nn.ModuleList([nn.Conv2d(in_channels,4,3,padding=1),nn.Conv2d(4,8,3,stride=2,padding=1),nn.Conv2d(8,16,3,stride=2,padding=1),nn.Conv2d(16,32,3,padding=1)])
    def forward(self,x):
        result=[]
        for layer in self.layers:x=layer(x).relu();result.append(x)
        return tuple(result)


class USLTest(unittest.TestCase):
    def test_student_gradient_and_ignore(self):
        model=USLStudent(TinyEncoder());x=torch.rand(2,3,16,16)
        target=torch.zeros(2,1,16,16);target[:,:,4:12,4:12]=1;target[:,:,:2]=.5
        outputs=model(x);outputs['logits'].retain_grad();loss=model.loss(outputs,target)
        loss.backward();self.assertEqual(outputs['logits'].shape,target.shape)
        self.assertEqual(outputs['logits'].grad[:,:,:2].abs().sum(),0)
        self.assertGreater(model.decoder.output.weight.grad.abs().sum(),0)
    def test_frozen_three_source_fusion_and_ccam_objective(self):
        encoders={name:TinyEncoder() for name in ('simclr','mocov1','mocov2')}
        fusion=ContrastiveFusion(encoders);model=USLSeedGenerator(fusion,channels=3*(16+32))
        x=torch.rand(2,3,16,16);features=fusion(x)
        foreground=1-torch.sigmoid(model.ccam.background(features))
        loss=existing_ccam_loss(features,foreground);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(model.ccam.background.weight.grad)
        self.assertTrue(all(p.grad is None for p in fusion.parameters()))
    def test_simclr_prefers_matched_pairs(self):
        first=torch.eye(4);matched=simclr_loss(first,first,temperature=.1)
        wrong=simclr_loss(first,first.roll(1,0),temperature=.1)
        self.assertLess(matched,wrong)
    def test_moco_queue_wrap_and_backward(self):
        encoder=ContrastiveEncoder('mocov1',projection_dim=4,backbone=TinyEncoder(),pooled_channels=32)
        model=MomentumContrast(encoder,dimension=4,queue_size=3)
        loss=model(torch.rand(2,3,8,8),torch.rand(2,3,8,8));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(int(model.pointer),2)
        model(torch.rand(2,3,8,8),torch.rand(2,3,8,8));self.assertEqual(int(model.pointer),1)
        self.assertTrue(all(p.grad is None for p in model.key.parameters()))
    def test_local_mocov3_and_dino_gradients(self):
        encoder=ContrastiveEncoder('mocov2',projection_dim=4,backbone=TinyEncoder(),pooled_channels=32)
        v3=MoCoV3(encoder,dimension=4);loss=v3(torch.rand(2,3,8,8),torch.rand(2,3,8,8));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        vit=PatchTransformer(dimension=8,depth=1,heads=2,patch_size=4,image_size=16)
        dino=DinoDistillation(vit,dimension=8,out_dim=16)
        loss=dino(torch.rand(2,3,16,16),torch.rand(2,3,16,16));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(vit(torch.rand(2,3,16,16)).shape,(2,8,4,4))
        self.assertTrue(all(p.grad is None for p in dino.teacher_backbone.parameters()))


if __name__=='__main__':unittest.main()
