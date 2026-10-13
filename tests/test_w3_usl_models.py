import unittest
import torch
from dermssl.w3_usl_models import masked_bce_logits, USLTextCCAM, USLTextFCNDecoder


class USLModelsTest(unittest.TestCase):
    def test_bce_learns_both_classes_and_ignores_uncertainty(self):
        logits = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0., .5, 1.])
        masked_bce_logits(logits, target).backward()
        self.assertGreater(logits.grad[0], 0)
        self.assertEqual(logits.grad[1], 0)
        self.assertLess(logits.grad[2], 0)

    def test_all_uncertain_is_not_a_zero_loss_success(self):
        with self.assertRaises(ValueError):
            masked_bce_logits(torch.zeros(2), torch.full((2,), .5))

    def test_ccam_complements_background_and_has_one_by_one_kernel(self):
        model = USLTextCCAM(4)
        with torch.no_grad():
            model.background.weight.zero_(); model.background.bias.fill_(2.)
            out = model(torch.zeros(1, 4, 7, 7))
        self.assertEqual(out.shape, (1, 1, 224, 224))
        self.assertEqual(model.background.kernel_size, (1, 1))
        torch.testing.assert_close(out, torch.full_like(out, 1 - torch.sigmoid(torch.tensor(2.))))

    def test_fcn_output_geometry_and_gradient_from_each_scale(self):
        torch.manual_seed(42)
        model = USLTextFCNDecoder(channels=(2, 3, 4))
        features = tuple(torch.randn(1, c, n, n, requires_grad=True)
                         for c, n in [(2, 8), (3, 4), (4, 2)])
        out = model(features, output_size=(32, 32))
        self.assertEqual(out.shape, (1, 1, 32, 32))
        out.mean().backward()
        for x in features:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum(), 0)


if __name__ == '__main__': unittest.main()
