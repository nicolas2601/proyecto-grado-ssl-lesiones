"""DIET-CP-style continued pretraining (estilo DIET-CP: aproximación declarada en ledger).

Each image is its own class (its index in the sealed fit manifest). Linear head over the CLS token,
cross-entropy with label smoothing 0.3, one augmented view per sample. First 5 % of steps train only the head;
afterwards the last two blocks train. Runtime only; sealed manifest (source_fit + pad_adapt); no test roles.
"""
import copy  # noqa: F401  (kept for parity with the sealed runtime layout)
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from dino_ibot_runtime import PUBLIC_SHA, augment, digest, validate_dino_config
from diet_losses import backbone_trainable, diet_loss

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = 12


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    validate_dino_config(config)
    base = config['base_config']
    rows = [json.loads(line) for line in Path(config['manifest']).read_text().splitlines()]
    if digest(config['manifest']) != config['manifest_sha256']:
        raise ValueError('Manifest digest mismatch')
    random.seed(base['seed']); np.random.seed(base['seed']); torch.manual_seed(base['seed'])
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    hub = next(Path.home().glob('.cache/torch/hub/facebookresearch_dinov2_main'))
    student = torch.hub.load(str(hub), 'dinov2_vitb14', source='local', pretrained=False)
    if digest(base['initial_weights']) != PUBLIC_SHA:
        raise ValueError('Public initializer changed')
    student.load_state_dict(torch.load(base['initial_weights'], map_location='cpu', weights_only=True))
    student.to(device)
    student.requires_grad_(False)
    head = torch.nn.Linear(768, len(rows)).to(device)
    out = Path(base['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    aug = augment()
    optimizer = torch.optim.AdamW([{'params': [], 'lr': 1e-4}, {'params': list(head.parameters()), 'lr': 1e-4}],
                                  weight_decay=0.05)
    param_groups = optimizer.param_groups
    for step in range(1, base['max_steps'] + 1):
        trainable = backbone_trainable(step, base['max_steps'], BLOCKS)
        for b, blk in enumerate(student.blocks):
            blk.requires_grad_(b in trainable)
        param_groups[0]['params'] = [p for blk in student.blocks for p in blk.parameters() if p.requires_grad]
        idxs = random.sample(range(len(rows)), config['batch_size'])
        batch = [rows[i] for i in idxs]
        view = torch.stack([aug(Image.open(r['image_path']).convert('RGB')) for r in batch]).to(device)
        target = torch.tensor(idxs, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cls = student.forward_features(view)['x_norm_clstoken']
        loss = diet_loss(head(cls.float()), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(json.dumps({'step': step, 'loss': float(loss), 'trainable_blocks': sorted(trainable)}), flush=True)
    torch.save(student.state_dict(), out / 'simsiam_dinov2_backbone_final.pt')
    result = {'status': 'COMPLETE', 'steps': base['max_steps'],
              'backbone_sha256': digest(out / 'simsiam_dinov2_backbone_final.pt'),
              'max_memory_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result
