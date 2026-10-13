import unittest
import torch
from torch import nn
from dermssl.segmentation_methods.tridermnet_hago.model import TriDermNet
from dermssl.segmentation_methods.tridermnet_hago.loss import losses
from dermssl.segmentation_methods.tridermnet_hago.hago import HAGO


class TriDermTest(unittest.TestCase):
    def test_region_edge_token_and_surrogate_backward(self):
        model=TriDermNet(width=4,token_dim=8,patch_size=2,heads=2,max_token_side=8)
        x=torch.rand(2,3,32,32);target=torch.zeros(2,1,32,32);target[:,:,10:22,10:22]=1;target[:,:,:4]=.5
        output=model(x);tasks=losses(output,target);total=tasks['segmentation']+tasks['coverage'];total.backward()
        self.assertEqual(output['logits'].shape,target.shape)
        for parameter in [model.region_compress[0].weight,model.edge[0][0].weight,model.patch.weight,model.gamma.weight,model.coverage.weight]:
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
    def test_hago_suppresses_dominant_and_paper_sign_is_explicit(self):
        for convention,expected in [('suppress_dominant',False),('paper_logistic',True)]:
            model=nn.Linear(1,1,bias=False);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
            x=model(torch.ones(1,1));hago=HAGO(model,optimizer,warmup_epochs=0,balance=convention)
            metadata=hago.step(100*x.sum(),x.sum(),torch.zeros(1,1),0)
            self.assertEqual(metadata['segmentation_weight']>.5,expected)
            self.assertTrue(torch.isfinite(model.weight).all());self.assertFalse(metadata['noise_applied'])
    def test_hago_warmup_and_overconfident_entropy_are_finite(self):
        model=nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        control=HAGO(model,optimizer,warmup_epochs=1)
        x=model(torch.ones(1,1));meta=control.step(x.square().sum(),x.square().sum(),torch.full((1,1),100.),0)
        self.assertTrue(meta['warmup'])
        x=model(torch.ones(1,1));meta=control.step(x.square().sum(),x.square().sum(),torch.full((1,1),100.),1)
        self.assertTrue(torch.isfinite(torch.tensor(meta['entropy'])))
    def test_pseudo_loss_rejects_diagnostic_targets(self):
        model=TriDermNet(width=4,token_dim=8,patch_size=2,heads=2)
        output=model(torch.rand(1,3,16,16))
        with self.assertRaises(ValueError):losses(output,torch.full((1,1,16,16),3.))


if __name__=='__main__':unittest.main()
