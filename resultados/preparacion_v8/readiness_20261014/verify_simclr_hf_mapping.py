"""One new CPU comparison of real HF weights under explicitly aligned profiles."""
from pathlib import Path
import json
import sys
import torch
from torch import nn

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'repo_snapshot/src'))
sys.path.insert(0, str(root / 'hf_simclr_review/upstream'))
from dermssl.segmentation_methods.common.bank import source_bank
import resnet_wider

torch.set_num_threads(2)
folder = root / 'hf_simclr_review'
state = torch.load(folder / 'resnet50-1x.pth', map_location='cpu', weights_only=True)['state_dict']
reference = resnet_wider.resnet50x1().eval()
reference.load_state_dict(state, strict=True)
bank_file = folder / 'simclr-bank.local-candidate.json'
bank_file.write_text(json.dumps({'sources': {'simclr': {
    'checkpoint': str(folder / 'converted/encoder.pt'),
    'provenance': str(folder / 'converted/encoder.pt.json')}}}, indent=2) + '\n')
local = source_bank(bank_file, ['simclr'])['simclr']
image = (torch.arange(3 * 224 * 224, dtype=torch.float32).reshape(1, 3, 224, 224) % 257) / 256


def stages(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    result = []
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        x = layer(x)
        result.append(x)
    return result


with torch.inference_mode():
    published_profile = stages(reference, image)
    # Independent fixed-window reference for the tested 112x112 even stem.
    reference.maxpool = nn.Sequential(nn.ConstantPad2d((0, 1, 0, 1), float('-inf')),
                                      nn.MaxPool2d(3, stride=2, padding=0))
    # Explicit V8 dense-feature adaptation, identical weights and parameter schema.
    reference.layer4[0].conv2.stride = (1, 1)
    reference.layer4[0].downsample[0].stride = (1, 1)
    aligned_profile = stages(reference, image)
    actual = local(image)
    errors = []
    for expected, observed in zip(aligned_profile, actual, strict=True):
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
        errors.append(float((observed - expected).abs().max()))
    symmetric_differences = [float((published_profile[i] - actual[i]).abs().max()) for i in range(3)]

report = {'status': 'passed_explicitly_aligned_pytorch_reference', 'torch_version': torch.__version__,
          'device': 'cpu', 'source_tensor_count': len(state), 'mapped_encoder_tensor_count': len(local.state_dict()),
          'discarded_classifier_keys': ['fc.weight', 'fc.bias'], 'input_normalization': 'unit_rgb',
          'max_absolute_error_by_stage': errors, 'stage_shapes': [list(x.shape) for x in actual],
          'unaltered_publisher_symmetric_vs_corrected_profile_max_abs_first_three_stages': symmetric_differences,
          'reference_changes': ['even-stem TensorFlow SAME maxpool (0 before, 1 after)', 'last-stage stride 2 to 1'],
          'comparison_scope': 'real downloaded PyTorch weights and explicit V8 spatial/dense profile; not original TensorFlow checkpoint parity',
          'original_google_archive_identity_verified': False, 'tensorflow_numerical_equivalence_verified': False,
          'real_images_read': 0, 'synthetic_inputs': 1, 'training_started': False, 'cluster_deployment_verified': False}
(folder / 'mapping-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
