"""Levers that keep continual SimSiam from erasing what DINOv2 already knows.

All of them are OFF by default, and off means the v6 run: one optimizer group
over ``model.parameters()`` and a loss that IS the SimSiam loss. docs/28 fixes
which combination is used and with which values; this module only makes each
lever exist, testable in isolation.

* ``freeze_blocks``: patch embedding plus the first N transformer blocks stop
  training. LayerNorms keep training everywhere (docs/28): they are cheap to
  adapt and are where a domain shift in intensity statistics shows first.
* ``layer_decay``: learning rate scaled by gamma ** (depth from the top), so
  early layers move less than late ones.
* ``anchor_l2sp``: squared distance to a frozen copy of the starting weights
  (L2-SP), a pull on the PARAMETERS.
* ``anchor_distill``: 1 - cosine between the CLS of the backbone in training
  and the CLS of the frozen public DINOv2 on the original view, a pull on the
  FEATURES, which is stronger than L2-SP where many weights map to the same
  function.
* ``tone_consistency_weight``: a second term, added (never multiplied), that
  pulls the backbone feature of the tone-shifted view towards the feature of
  its original. It lives on the backbone features, not on the SimSiam
  projector/predictor tensors, and the original side is detached: that is
  what keeps it from being the SimSiam loss counted twice, the defect removed
  on 2026-09-22.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from dermssl.pretrain.levers import Levers

EMBEDDING_PARAMS = ("cls_token", "pos_embed", "register_tokens", "mask_token")


@dataclass(eq=False)
class Anchors:
    """Frozen references built from the STARTING weights, before any resume."""

    initial: dict[str, torch.Tensor] | None = None
    teacher: nn.Module | None = None

    def to(self, device) -> "Anchors":
        if self.initial is not None:
            self.initial = {k: v.to(device) for k, v in self.initial.items()}
        if self.teacher is not None:
            self.teacher.to(device)
        return self


def _vit_blocks(backbone: nn.Module) -> list[nn.Module]:
    """Blocks in depth order; DINOv2 may chunk them into nested ModuleLists."""
    blocks = []
    for item in backbone.blocks:
        if isinstance(item, nn.ModuleList):
            blocks.extend(b for b in item if not isinstance(b, nn.Identity))
        else:
            blocks.append(item)
    return blocks


def _embedding_parameters(backbone: nn.Module) -> list[nn.Parameter]:
    params = list(backbone.patch_embed.parameters())
    params += [getattr(backbone, n) for n in EMBEDDING_PARAMS if isinstance(getattr(backbone, n, None), nn.Parameter)]
    return params


def _norm_parameter_ids(module: nn.Module) -> set[int]:
    return {id(p) for m in module.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()}


def freeze_blocks(backbone: nn.Module, n_blocks: int) -> None:
    """Freeze the embeddings and the first ``n_blocks``; LayerNorms stay trainable."""
    if n_blocks == 0:
        return
    blocks = _vit_blocks(backbone)
    if n_blocks > len(blocks):
        raise ValueError(f"freeze_blocks={n_blocks} pero el backbone tiene {len(blocks)} bloques")
    keep = _norm_parameter_ids(backbone)
    frozen = _embedding_parameters(backbone)
    for block in blocks[:n_blocks]:
        frozen += list(block.parameters())
    for param in frozen:
        if id(param) not in keep:
            param.requires_grad = False


def _depths(model: nn.Module) -> tuple[dict[int, int], int]:
    """Depth per parameter id: 0 embeddings, i+1 block i, top = everything else."""
    backbone = model.backbone
    blocks = _vit_blocks(backbone)
    depth = {id(p): 0 for p in _embedding_parameters(backbone)}
    for index, block in enumerate(blocks):
        depth.update({id(p): index + 1 for p in block.parameters()})
    return depth, len(blocks) + 1


def param_groups(model: nn.Module, levers: Levers) -> list[dict]:
    """Optimizer groups. Default: ONE group over ``model.parameters()`` (v6)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if levers.layer_decay == 1.0:
        if len(trainable) == len(list(model.parameters())):
            return [{"params": list(model.parameters())}]
        return [{"params": trainable}]
    depth_of, top = _depths(model)
    by_scale: dict[float, list[nn.Parameter]] = {}
    for param in trainable:
        scale = levers.layer_decay ** (top - depth_of.get(id(param), top))
        by_scale.setdefault(scale, []).append(param)
    return [{"params": params, "lr_scale": scale} for scale, params in sorted(by_scale.items())]


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def build_anchors(backbone: nn.Module, levers: Levers) -> Anchors:
    """Snapshot the starting backbone. Call BEFORE loading a resume checkpoint."""
    anchors = Anchors()
    if levers.anchor_l2sp > 0:
        anchors.initial = {n: p.detach().clone() for n, p in backbone.named_parameters()}
    if levers.anchor_distill > 0:
        teacher = copy.deepcopy(backbone).eval()
        for param in teacher.parameters():
            param.requires_grad = False
        anchors.teacher = teacher
    return anchors


def l2sp_penalty(backbone: nn.Module, anchors: Anchors) -> torch.Tensor:
    """Sum of squared distances to the starting weights, trainable params only."""
    if anchors.initial is None:
        raise ValueError("l2sp_penalty sin pesos iniciales: construí los anclajes con anchor_l2sp > 0")
    terms = [
        (param.float() - anchors.initial[name].float()).pow(2).sum()
        for name, param in backbone.named_parameters()
        if param.requires_grad
    ]
    if not terms:
        return torch.zeros((), device=next(backbone.parameters()).device)
    return torch.stack(terms).sum()


def distill_loss(student_cls: torch.Tensor, anchors: Anchors, original_view: torch.Tensor) -> torch.Tensor:
    """1 - cosine between the trained CLS and the frozen public CLS."""
    if anchors.teacher is None:
        raise ValueError("distill_loss sin teacher: construí los anclajes con anchor_distill > 0")
    with torch.no_grad():
        target = anchors.teacher(original_view)
    return (1.0 - F.cosine_similarity(student_cls.float(), target.float(), dim=-1)).mean()


def tone_consistency(h_original: torch.Tensor, h_tone: torch.Tensor) -> torch.Tensor:
    """1 - cosine(tone feature, detached original feature)."""
    return (1.0 - F.cosine_similarity(h_tone.float(), h_original.detach().float(), dim=-1)).mean()


def compose_objective(
    ssl_loss: torch.Tensor,
    levers: Levers,
    *,
    tone: torch.Tensor | None = None,
    l2sp: torch.Tensor | None = None,
    distill: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SimSiam loss plus every active lever, each weighted and ADDED."""
    loss, parts = ssl_loss, {}
    for name, weight, term in (
        ("tone_consistency", levers.tone_consistency_weight, tone),
        ("anchor_l2sp", levers.anchor_l2sp, l2sp),
        ("anchor_distill", levers.anchor_distill, distill),
    ):
        if weight == 0:
            continue
        if term is None:
            raise ValueError(f"{name} tiene peso {weight} pero no se calculó su término")
        loss = loss + weight * term
        parts[name] = term.detach()
    return loss, parts


@dataclass
class LossTotals:
    """Running sums that survive a resume, averaged over the steps summed.

    run.json used to divide sums restarted at zero on resume by the step
    count carried across it, which reported -0.047 for an arm whose logged
    curve ends at -0.92.
    """

    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, **values: float | None) -> None:
        for name, value in values.items():
            if value is None:
                continue
            self.sums[name] = self.sums.get(name, 0.0) + float(value)
            self.counts[name] = self.counts.get(name, 0) + 1

    def mean(self, name: str) -> float | None:
        count = self.counts.get(name, 0)
        return self.sums[name] / count if count else None

    def state(self) -> dict:
        return {"sums": dict(self.sums), "counts": dict(self.counts)}

    @classmethod
    def from_state(cls, state: dict | None) -> "LossTotals":
        if not state:
            return cls()
        return cls(sums=dict(state["sums"]), counts=dict(state["counts"]))
