"""Preserve the native baseline view pipeline except its crop-scale lower bound."""
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from overlay_ssl_runtime import run as baseline_run

ROOT = Path(__file__).resolve().parents[1]


def crop_geometric(native_factory, resolution, scale):
    if tuple(scale) != (.2, 1.):
        raise ValueError('Expected the unchanged baseline crop-scale policy')
    return native_factory(resolution, scale=(.4, 1.))


def validate_crop_config(config):
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if config.get('arm') != 'crop-scale-0.4' or config.get('runtime_hook_sha256') != code_sha:
        raise ValueError('Crop runtime hook or arm does not match the sealed config')
    baseline = json.loads((ROOT / 'config/ssl-overlay-v1/A-s42.json').read_text())
    operational = {'output_dir', 'max_steps', 'checkpoint_every_steps', 'resume_checkpoint'}
    expected = {k: v for k, v in baseline['base_config'].items() if k not in operational}
    actual = {k: v for k, v in config['base_config'].items() if k not in operational}
    if actual != expected or config['base_config']['max_steps'] not in (40, 6000):
        raise ValueError('Crop arm changes a baseline hyperparameter')
    for key in ('manifest', 'manifest_sha256', 'overlay_receipt', 'overlay_receipt_sha256', 'lr_multiplier'):
        if config[key] != baseline[key]:
            raise ValueError('Crop arm changes baseline sampling or provenance')
    return code_sha


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    code_sha = validate_crop_config(config)
    frozen = Path(json.loads((ROOT / 'config/paths.json').read_text())['frozen_experiment_root'])
    sys.path.insert(0, str(frozen / 'code/src'))
    from dermssl.pretrain import torch_data
    replacement = partial(crop_geometric, torch_data._geometric)
    with patch.object(torch_data, '_geometric', new=replacement):
        result = baseline_run(config_path)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != code_sha:
        raise ValueError('Crop hook source changed during training')
    record = {'arm': config['arm'], 'hook_sha256': code_sha, 'crop_scale': [.4, 1.],
              'baseline_crop_scale': [.2, 1.], 'other_baseline_hyperparameters_unchanged': True,
              'scope': 'Reuse native geometric factory;flips,photometry,hue0.02,sampling and objective unchanged'}
    (Path(config['base_config']['output_dir']) / 'crop-scale-receipt.json').write_text(json.dumps(record, indent=2))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    run(parser.parse_args().config)
