"""Feature-predictor distillation for SimSiam; inspired by, not a reproduction of, CaSSLe."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from overlay_ssl_runtime import run as baseline_run

ROOT = Path(__file__).resolve().parents[1]


def make_feature_predictor(model):
    dim = model.backbone.embed_dim
    hidden = model.predictor[0].out_features
    # Keep baseline augmentation RNG unchanged when adding the new head.
    with torch.random.fork_rng(devices=[]):
        return nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.BatchNorm1d(hidden),
                             nn.ReLU(inplace=True), nn.Linear(hidden, dim))


def feature_distill_loss(student, teacher, predictor):
    projected = predictor(student)
    return (1. - F.cosine_similarity(projected.float(), teacher.detach().float(), dim=-1)).mean()


def validate_hook_config(config):
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if config.get('runtime_hook_sha256') != code_sha:
        raise ValueError('Predictor distillation runtime hook digest mismatch')
    base = config['base_config']
    if (config.get('arm') != 'simsiam-distill-with-predictor' or config.get('lr_multiplier') != 1
            or base.get('freeze_blocks') != 8 or base.get('layer_decay') != .75
            or base.get('clinical_batch_quota') != 4 or base.get('seed') != 42):
        raise ValueError('Only the baseline recipe with predictor feature distillation is allowed')
    return code_sha


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    code_sha = validate_hook_config(config)
    frozen = Path(json.loads((ROOT / 'config/paths.json').read_text())['frozen_experiment_root'])
    sys.path.insert(0, str(frozen / 'code/src'))
    from dermssl.pretrain import torch_adaptation, torch_simsiam
    original_model = torch_simsiam.SimSiam
    models, initial_heads = [], []

    class PredictorDistillSimSiam(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.distill_predictor = make_feature_predictor(self)
            initial_heads.append({name: p.detach().cpu().clone() for name, p in self.distill_predictor.named_parameters()})
            models.append(self)

    def distill(student, anchors, original_view):
        if len(models) != 1 or anchors.teacher is None:
            raise ValueError('One registered predictor and frozen public teacher required')
        with torch.no_grad():
            target = anchors.teacher(original_view)
        return feature_distill_loss(student, target, models[0].distill_predictor)

    with patch.object(torch_simsiam, 'SimSiam', PredictorDistillSimSiam), \
            patch.object(torch_adaptation, 'distill_loss', new=distill):
        result = baseline_run(config_path)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != code_sha:
        raise ValueError('Predictor hook source changed during training')
    changed = any(not torch.equal(p.detach().cpu(), initial_heads[0][name])
                  for name, p in models[0].distill_predictor.named_parameters())
    if not changed:
        raise ValueError('Predictor parameters never updated; refusing completion')
    receipt = {'arm': config['arm'], 'hook_sha256': code_sha,
               'predictor_parameters_updated': True, 'target': 'Frozen public DINOv2 CLS',
               'scope': 'Feature-level predictor distillation,not literal CaSSLe projector-space reproduction'}
    (Path(config['base_config']['output_dir']) / 'predictor-distillation-receipt.json').write_text(json.dumps(receipt, indent=2))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    run(parser.parse_args().config)
