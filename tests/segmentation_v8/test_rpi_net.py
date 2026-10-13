import unittest
import numpy as np
import torch
from torch.nn import functional as F
from dermssl.segmentation_methods.rpi_net.model import RPINet,CrossAttention,PseudoInteraction
from dermssl.segmentation_methods.rpi_net.loss import dynamic_uncertainty_loss
from dermssl.segmentation_methods.rpi_net.pseudolabels import cluster_foreground,gradcampp_energy
from test_usl_net import TinyEncoder


class RPITest(unittest.TestCase):
    def test_attention_matches_scaled_dot_product_equation(self):
        attention=CrossAttention(2)
        with torch.no_grad():
            for layer in [attention.q,attention.k,attention.v]:layer.weight.copy_(torch.eye(2));layer.bias.zero_()
        q=torch.tensor([[[1.,0.],[0.,1.]]]);k=torch.tensor([[[2.,0.],[0.,2.]]])
        expected=torch.softmax(q@k.transpose(1,2)/(2**.5),dim=-1)@k
        torch.testing.assert_close(attention(q,k),expected)
    def test_other_sources_affect_pia(self):
        torch.manual_seed(7);module=PseudoInteraction(4);image=torch.rand(1,4,4)
        maps=[torch.rand(1,4,4) for _ in range(4)]
        before=module(image,maps)[0];maps[3]=maps[3]+2
        after=module(image,maps)[0];self.assertFalse(torch.allclose(before,after))
    def test_dul_equation_at_zero_sigma_and_extremes(self):
        logits=torch.zeros(2,1,8,8,requires_grad=True);target=torch.ones(2,4,8,8)
        sigma=torch.zeros_like(target,requires_grad=True)
        loss,meta=dynamic_uncertainty_loss(logits,target,sigma)
        self.assertAlmostEqual(float(loss.detach()),float(np.log(2)),places=5)
        loss.backward();self.assertTrue(torch.isfinite(sigma.grad).all())
        torch.testing.assert_close(meta['alpha_normalized'].sum(1),torch.ones(2))
        extreme=torch.full_like(target,100.,requires_grad=True)
        loss,meta=dynamic_uncertainty_loss(logits,target,extreme)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(float(meta['sigma_clipped_fraction']),1.)
    def test_image_only_inference_and_all_branches_receive_gradients(self):
        model=RPINet(TinyEncoder(),TinyEncoder(in_channels=1),width=8,token_grid=4)
        image=torch.rand(2,3,16,16);target=(torch.rand(2,4,16,16)>.5).float()
        output=model(image,target);loss,_=dynamic_uncertainty_loss(output['logits'],target,output['sigma']);loss.backward()
        self.assertGreater(model.uncertainty.weight.grad.abs().sum(),0)
        self.assertGreater(model.segmentation.weight.grad.abs().sum(),0)
        model.eval()
        with torch.no_grad():torch.testing.assert_close(model(image)['logits'],model(image,target)['logits'])
    def test_feature_clustering_polarity_and_class_agnostic_cam(self):
        features=np.ones((8,8,2));features[2:6,2:6]=5
        mask=cluster_foreground(features,(16,16),seed=42)
        self.assertEqual(mask[8,8],1);self.assertEqual(mask[0,0],0)
        activation=torch.arange(16.).reshape(1,1,4,4).requires_grad_(True)
        cam=gradcampp_energy(activation,(8,8))
        self.assertEqual(cam.shape,(1,1,8,8));self.assertTrue(torch.isfinite(cam).all())


if __name__=='__main__':unittest.main()
