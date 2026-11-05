"""Self-contained DINO+iBOT continued pretraining (EMA teacher, KoLeo, feature anchor).

Scope: runtime only. Uses the sealed fit manifest (source_fit + pad_adapt); never reads test roles.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
from swav_hooks import SwAVHead
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SHA = '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
ALLOWED_ROLES = {'source_fit', 'pad_adapt'}
SEALED = {'freeze_blocks': 8, 'weights': [1.0, 1.0, 0.1], 'prototypes': 65536, 'seed': 42}


def validate_dino_config(config):
    if not set(config['manifest_roles']) <= ALLOWED_ROLES:
        raise ValueError('Training roles outside source_fit/pad_adapt')
    for key, value in SEALED.items():
        if key in config and config[key] != value:
            raise ValueError(f'DINO+iBOT recipe differs from sealed value for {key}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def augment():
    return T.Compose([T.RandomResizedCrop(224, scale=(0.4, 1.0), antialias=True), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Head(nn.Module):
    def __init__(self, dim, prototypes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 2048), nn.GELU(), nn.Linear(2048, 256))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(256, prototypes, bias=False))

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


def sharpen_center_torch(logits, center, temp):
    return F.softmax((logits - center) / temp, dim=-1)


def koleo_torch(features):
    unit = F.normalize(features, dim=-1)
    dist = torch.cdist(unit, unit)
    self_pair = torch.eye(len(unit), dtype=torch.bool, device=unit.device)
    dist = dist.masked_fill(self_pair, float('inf'))
    return -torch.log(dist.min(dim=1).values + 1e-8).mean()


def masks_for(batch, patches, ratio, generator):
    keep = torch.rand(batch, patches, generator=generator) >= ratio
    return ~keep


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
    anchor = copy.deepcopy(student).eval().to(device).requires_grad_(False)
    for block in student.blocks[:SEALED['freeze_blocks']]:
        block.requires_grad_(False)
    student.patch_embed.requires_grad_(False)
    student.pos_embed.requires_grad_(False)
    student.cls_token.requires_grad_(False)
    student.to(device)
    head_s = Head(768, SEALED['prototypes']).to(device)
    swav_head = SwAVHead(768, 128, 1024).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    head_t = copy.deepcopy(head_s).eval().requires_grad_(False)
    dino_center = torch.zeros(SEALED['prototypes'], device=device)
    ibot_center = torch.zeros(SEALED['prototypes'], device=device)
    params = [p for p in student.parameters() if p.requires_grad] + list(head_s.parameters())
    optimizer = torch.optim.AdamW([{'params': [p for p in student.parameters() if p.requires_grad], 'lr': 1e-5},
                                   {'params': list(head_s.parameters()) + list(swav_head.parameters()), 'lr': 1e-4}], weight_decay=0.0)
    out = Path(base['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(base['seed'])
    aug = augment()
    weights = SEALED['weights']
    for step in range(1, base['max_steps'] + 1):
        batch = random.sample(rows, config['batch_size'])
        views = [torch.stack([aug(Image.open(r['image_path']).convert('RGB')) for r in batch]).to(device)
                 for _ in range(2)]
        mask = masks_for(len(batch), 256, 0.3, generator).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                t_out = [teacher.forward_features(v) for v in views]
                t_cls = [head_t(o['x_norm_clstoken'].float()) for o in t_out]
                t_patch = head_t(t_out[1]['x_norm_patchtokens'].float())
            s_out = [student.forward_features(views[0]), student.forward_features(views[1])]
            s_cls = [head_s(o['x_norm_clstoken'].float()) for o in s_out]
            s_patch = head_s(s_out[1]['x_norm_patchtokens'].float())
            anchor_cls = anchor.forward_features(views[0])['x_norm_clstoken'].float()
        sw = swav_head.loss(s_out[0]['x_norm_clstoken'].float(), s_out[1]['x_norm_clstoken'].float())
        distill = (1 - F.cosine_similarity(s_out[0]['x_norm_clstoken'].float(), anchor_cls, dim=-1)).mean()
        loss = sw + distill
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            m = base.get('teacher_momentum', 0.994)
            for pt, ps in zip(list(teacher.parameters()) + list(head_t.parameters()),
                              list(student.parameters()) + list(head_s.parameters())):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            dino_center.mul_(0.9).add_(t_cls[0].float().mean(0).detach() * 0.1 + t_cls[1].float().mean(0).detach() * 0.1)
            ibot_center.mul_(0.9).add_(t_patch.float().flatten(0, 1).mean(0).detach() * 0.1)
        print(json.dumps({'step': step, 'loss': float(loss), 'swav': float(sw), 'distill': float(distill)}), flush=True)
    torch.save(student.state_dict(), out / 'simsiam_dinov2_backbone_final.pt')
    result = {'status': 'COMPLETE', 'steps': base['max_steps'],
              'backbone_sha256': digest(out / 'simsiam_dinov2_backbone_final.pt'),
              'max_memory_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run(sys.argv[1])
