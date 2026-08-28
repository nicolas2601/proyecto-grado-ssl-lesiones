"""DINO pretraining on the skin pool - both ViT arms live here.

The continual arm starts from the released DINOv2 ViT-S/14 weights (trained
WITHOUT labels on LVD-142M, so the label-free invariant holds) and continues
the DINO self-distillation objective on dermoscopic images. The hypothesis:
adapting the encoder to skin texture, lesion borders and dermatoscope
artifacts beats the frozen generalist (v5's winner) at equal compute.

The scratch arm (``pretrained=False``) is the purist counterpart: identical
architecture and objective with random init, so the comparison against the
continual arm isolates ONE variable - the starting weights - and quantifies
exactly what the 142M external images contribute (docs/16, section 2).

Student and teacher share the architecture; only the student receives
gradients. The teacher is an exponential moving average of the student
(momentum ramped by ``schedules.ema_momentum_at``) and its outputs are
centered and sharpened - the DINO recipe against collapse.

The head output dimension defaults to 4096, not the paper's 65536: with a
48-70k image pool a 65k-prototype head is mostly dead weight. Configurable,
and worth revisiting once the lab shares the config of a real DINOv2 run.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_HIDDEN = 2048
HEAD_BOTTLENECK = 256


class DinoHead(nn.Module):
    """MLP + L2 bottleneck + weight-normalized prototype layer (DINO)."""

    def __init__(self, in_dim: int, out_dim: int = 4096) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN, HEAD_BOTTLENECK),
        )
        self.prototypes = nn.utils.weight_norm(
            nn.Linear(HEAD_BOTTLENECK, out_dim, bias=False)
        )
        self.prototypes.weight_g.data.fill_(1.0)
        self.prototypes.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1)
        return self.prototypes(x)


class DinoContinual(nn.Module):
    """DINO student-teacher pair over a DINOv2 backbone.

    ``pretrained=True`` is the continual arm (public LVD-142M weights);
    ``pretrained=False`` is the purist arm: identical architecture and
    objective, random init, so the two runs differ in exactly one variable.
    """

    def __init__(
        self, backbone_name: str, head_out_dim: int = 4096, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.student_backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=pretrained
        )
        embed_dim = self.student_backbone.embed_dim
        self.student_head = DinoHead(embed_dim, head_out_dim)

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_parameters():
            p.requires_grad = False

    def teacher_parameters(self):
        yield from self.teacher_backbone.parameters()
        yield from self.teacher_head.parameters()

    def student_parameters(self):
        yield from self.student_backbone.parameters()
        yield from self.student_head.parameters()

    def student_forward(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.student_head(self.student_backbone(c)) for c in crops]

    @torch.no_grad()
    def teacher_forward(self, global_crops: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.teacher_head(self.teacher_backbone(c)) for c in global_crops]

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for ps, pt in zip(self.student_parameters(), self.teacher_parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


class DinoLoss(nn.Module):
    """Cross-entropy between sharpened teacher and student, across crop pairs.

    The center is an EMA of teacher logits; subtracting it before the teacher
    softmax prevents one prototype from dominating (collapse mode 1), while
    the low teacher temperature sharpens targets (against uniform collapse,
    mode 2). Both mechanisms are the DINO originals.
    """

    def __init__(
        self,
        out_dim: int = 4096,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self, student_out: list[torch.Tensor], teacher_out: list[torch.Tensor]
    ) -> torch.Tensor:
        student_logp = [
            F.log_softmax(s / self.student_temp, dim=-1) for s in student_out
        ]
        teacher_prob = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_out
        ]

        total, n_terms = student_out[0].new_zeros(()), 0
        for t_idx, t_prob in enumerate(teacher_prob):
            for s_idx, s_logp in enumerate(student_logp):
                if s_idx == t_idx:
                    continue  # never distill a view onto itself
                total = total + (-(t_prob * s_logp).sum(dim=-1)).mean()
                n_terms += 1
        self._update_center(teacher_out)
        return total / max(n_terms, 1)

    @torch.no_grad()
    def _update_center(self, teacher_out: list[torch.Tensor]) -> None:
        batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum
        )
