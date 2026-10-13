import unittest
import numpy as np
import torch
from torch.nn import functional as F
from dermssl.segmentation_methods.rpi_net.model import RPINet,CrossAttention,PseudoInteraction,adaptive_average_tokens
from dermssl.segmentation_methods.rpi_net.loss import dynamic_uncertainty_loss
from dermssl.segmentation_methods.rpi_net.pseudolabels import cluster_foreground,gradcampp_energy
from test_usl_net import TinyEncoder


class RPITest(unittest.TestCase):
    def test_token_pool_matches_reference_values_and_gradients(self):
        for height, width, grid in ((16,16,(4,4)), (7,9,(4,5)), (2,3,(5,4))):
            with self.subTest(shape=(height,width), grid=grid):
                x=torch.randn(2,3,height,width,dtype=torch.float64,requires_grad=True)
                reference=x.detach().clone().requires_grad_(True)
                actual=adaptive_average_tokens(x,grid)
                expected=F.adaptive_avg_pool2d(reference,grid)
                weights=torch.randn_like(actual)
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                actual.backward(weights);expected.backward(weights)
                torch.testing.assert_close(x.grad,reference.grad,rtol=1e-12,atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for regression')
    def test_token_pool_cuda_backward_is_repeatable(self):
        previous=torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for height,width,grid in ((16,16,(4,4)), (7,9,(4,5))):
                base=torch.arange(2*3*height*width,device='cuda',dtype=torch.float32).reshape(2,3,height,width)
                outputs=[];gradients=[]
                for _ in range(2):
                    x=base.detach().clone().requires_grad_(True)
                    pooled=adaptive_average_tokens(x,grid)
                    pooled.square().sum().backward()
                    outputs.append(pooled.detach());gradients.append(x.grad.detach())
                self.assertTrue(torch.equal(outputs[0],outputs[1]))
                self.assertTrue(torch.equal(gradients[0],gradients[1]))
        finally:
            torch.use_deterministic_algorithms(previous)


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
