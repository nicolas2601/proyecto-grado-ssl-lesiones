"""New spatial contract checks; no downloaded weights, images, or training."""
import json
from pathlib import Path
import tempfile
import runpy
import unittest
import numpy as np
import torch
from torch import nn
from dermssl.segmentation_methods.common.networks import (
    TensorFlowSameMaxPool2d, ResNet50Stages, public_resnet_encoder)
from dermssl.segmentation_methods.common.bank import source_bank


def window_oracle(array):
    height, width = array.shape
    out_h, out_w = (height + 1) // 2, (width + 1) // 2
    top = max((out_h - 1) * 2 + 3 - height, 0) // 2
    left = max((out_w - 1) * 2 + 3 - width, 0) // 2
    output = np.empty((out_h, out_w), dtype=array.dtype)
    for row in range(out_h):
        for col in range(out_w):
            start_h, start_w = 2 * row - top, 2 * col - left
            valid = array[max(0, start_h):min(height, start_h + 3),
                          max(0, start_w):min(width, start_w + 3)]
            output[row, col] = valid.max()
    return output


class SimCLRSpatialProfileTests(unittest.TestCase):
    def test_even_stem_uses_trailing_padding_and_correct_first_window(self):
        image = torch.arange(112 * 112, dtype=torch.float32).reshape(1, 1, 112, 112)
        same = TensorFlowSameMaxPool2d()(image)
        symmetric = nn.MaxPool2d(3, stride=2, padding=1)(image)
        self.assertEqual(tuple(same.shape), (1, 1, 56, 56))
        self.assertEqual(same[0, 0, 0, 0].item(), 226)
        self.assertEqual(symmetric[0, 0, 0, 0].item(), 113)
        np.testing.assert_array_equal(same[0, 0].numpy(), window_oracle(image[0, 0].numpy()))

    def test_odd_rectangular_and_negative_edges_match_independent_windows(self):
        rng = np.random.default_rng(42)
        for shape in ((111, 113), (8, 9), (7, 10), (1, 1), (1, 2)):
            with self.subTest(shape=shape):
                array = -rng.uniform(1, 20, shape).astype(np.float32)
                actual = TensorFlowSameMaxPool2d()(torch.from_numpy(array)[None, None])
                np.testing.assert_array_equal(actual[0, 0].numpy(), window_oracle(array))

    def test_profiles_preserve_parameter_schema_and_moco_defaults(self):
        with torch.device('meta'):
            default = ResNet50Stages(last_stride=1)
            simclr = public_resnet_encoder('simclr')
            moco = public_resnet_encoder('mocov2')
        self.assertIsInstance(default.stem[3], nn.MaxPool2d)
        self.assertIsInstance(moco.stem[3], nn.MaxPool2d)
        self.assertIsInstance(simclr.stem[3], TensorFlowSameMaxPool2d)
        self.assertEqual({k: tuple(v.shape) for k, v in default.state_dict().items()},
                         {k: tuple(v.shape) for k, v in simclr.state_dict().items()})

    def test_bank_rejects_legacy_simclr_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = root / 'provenance.json'
            provenance.write_text(json.dumps({'stem_pooling': 'symmetric'}))
            bank = root / 'bank.json'
            bank.write_text(json.dumps({'sources': {'simclr': {
                'checkpoint': str(root / 'absent.pt'), 'provenance': str(provenance)}}}))
            with self.assertRaisesRegex(ValueError, 'TensorFlow SAME'):
                source_bank(bank, ['simclr'])

    def test_tf_conversion_rejects_wrong_profile_before_tensorflow_import(self):
        root = Path(__file__).resolve().parents[2]
        adapter = runpy.run_path(str(root / 'scripts/prepare_public_ssl_weights_v8.py'))
        with torch.device('meta'):
            model = ResNet50Stages(last_stride=1)
        with self.assertRaisesRegex(ValueError, 'TensorFlow SAME'):
            adapter['convert_simclr_tensorflow'](Path('absent-checkpoint'), model)


if __name__ == '__main__':
    unittest.main()
