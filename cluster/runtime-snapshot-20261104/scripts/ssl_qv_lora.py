"""Q/V-only low-rank updates for a fused DINOv2 attention projection."""
import torch
from torch import nn
from torch.nn import functional as F


class QVLoRA(nn.Module):
    def __init__(self, base, rank=16):
        super().__init__()
        if (not isinstance(base, nn.Linear) or base.out_features != 3 * base.in_features
                or not isinstance(rank, int) or not 0 < rank <= base.in_features):
            raise ValueError('Expected fused QKV linear and valid positive rank')
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        for name in ('q', 'v'):
            a = torch.empty(rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
            nn.init.kaiming_uniform_(a, a=5 ** .5)
            self.register_parameter(f'lora_{name}_a', nn.Parameter(a))
            self.register_parameter(f'lora_{name}_b', nn.Parameter(self.weight.new_zeros(self.in_features, rank)))

    def forward(self, x):
        q = F.linear(F.linear(x, self.lora_q_a), self.lora_q_b)
        v = F.linear(F.linear(x, self.lora_v_a), self.lora_v_b)
        return F.linear(x, self.weight, self.bias) + torch.cat((q, torch.zeros_like(q), v), dim=-1)

    def effective_weight(self):
        q = self.lora_q_b @ self.lora_q_a
        v = self.lora_v_b @ self.lora_v_a
        return self.weight + torch.cat((q, torch.zeros_like(q), v), dim=0)


def install_qv_lora(backbone, rank=16):
    blocks = list(backbone.blocks)
    if (len(blocks) != 12 or not isinstance(rank, int) or rank < 1
            or any(not isinstance(b.attn.qkv, nn.Linear) or b.attn.qkv.out_features != 3 * b.attn.qkv.in_features
                   or rank > b.attn.qkv.in_features for b in blocks)):
        raise ValueError('Expected twelve native ViT-B fused-QKV blocks and valid rank')
    backbone.requires_grad_(False)
    blocks[-1].requires_grad_(True)
    for module in backbone.modules():
        if isinstance(module, nn.LayerNorm):
            module.requires_grad_(True)
    for block in blocks[:-1]:
        block.attn.qkv = QVLoRA(block.attn.qkv, rank)


def merged_backbone_state(backbone):
    state = {name: value.detach().cpu().clone() for name, value in backbone.state_dict().items()
             if not any(part.startswith('lora_') for part in name.split('.'))}
    for name, module in backbone.named_modules():
        if isinstance(module, QVLoRA):
            state[f'{name}.weight'] = module.effective_weight().detach().cpu().clone()
    return state


def explora_l2sp_penalty(backbone, anchors):
    if anchors.initial is None:
        raise ValueError('Public starting weights required for L2-SP')
    terms = [(p.float() - anchors.initial[name].float()).square().sum()
             for name, p in backbone.named_parameters()
             if p.requires_grad and not any(part.startswith('lora_') for part in name.split('.'))]
    for name, module in backbone.named_modules():
        if isinstance(module, QVLoRA):
            terms.append((module.effective_weight().float() - anchors.initial[f'{name}.weight'].float()).square().sum())
    return torch.stack(terms).sum() if terms else next(backbone.parameters()).new_zeros(())


def explora_param_groups(model, levers):
    if levers.layer_decay != 1.:
        raise ValueError('ExPLoRA-style arm disables layer decay')
    base_lr = 7.8125e-6
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        rate = 1e-4 if '.lora_' in name else (1e-5 if name.startswith('backbone.') else base_lr)
        groups.setdefault(rate, []).append(param)
    return [{'params': params, 'lr_scale': rate / base_lr} for rate, params in sorted(groups.items())]
