"""Tensor helpers for the MoCo-v3-style SSL arm on DINOv2 ViT-B/14.

MoCo-v3-style only: momentum key encoder, symmetric InfoNCE with in-batch negatives (no memory queue).
This is not a reproduction of the official MoCo v3 recipe.
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn

TEMPERATURE = 0.2
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_DIM = 768
HEAD_HIDDEN_DIM = 2048
PROJECTION_DIM = 256
ALLOWED_ROLES = frozenset({"source_fit", "pad_adapt"})
FORBIDDEN_LABEL_FIELDS = frozenset({"dx", "label", "labels", "diagnosis", "class", "target"})


def _contrastive_term(query, key, temperature):
    logits = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).T / temperature
    targets = torch.arange(query.shape[0], device=query.device)
    return F.cross_entropy(logits, targets)


def moco_symmetric_loss(q1, q2, k1, k2, temperature=TEMPERATURE):
    """Mean of q1-vs-k2 and q2-vs-k1 InfoNCE; positives are index-aligned, negatives are in-batch."""
    if not (q1.shape == q2.shape == k1.shape == k2.shape) or q1.ndim != 2:
        raise ValueError("q1, q2, k1, k2 must share shape (batch, dim)")
    return (_contrastive_term(q1, k2, temperature) + _contrastive_term(q2, k1, temperature)) / 2


@torch.no_grad()
def ema_update(teacher, student, momentum):
    """Exact EMA over parameters and buffers; floating buffers are averaged, integer buffers copied."""
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.mul_(momentum).add_(student_param, alpha=1 - momentum)
    for (_, teacher_buf), (_, student_buf) in zip(teacher.named_buffers(), student.named_buffers()):
        if teacher_buf.is_floating_point():
            teacher_buf.mul_(momentum).add_(student_buf, alpha=1 - momentum)
        else:
            teacher_buf.copy_(student_buf)


def build_teacher(student):
    """Frozen, eval-mode deep copy of the student module, used as the momentum encoder."""
    return copy.deepcopy(student).eval().requires_grad_(False)


def apply_freeze_plan(backbone, freeze_blocks):
    """Freeze everything, then re-enable blocks at index >= freeze_blocks and the final norm."""
    backbone.requires_grad_(False)
    for block in backbone.blocks[freeze_blocks:]:
        block.requires_grad_(True)
    backbone.norm.requires_grad_(True)


def build_heads():
    """Projector and predictor MLPs; the predictor exists only on the student side."""
    projector = nn.Sequential(nn.Linear(EMBED_DIM, HEAD_HIDDEN_DIM), nn.GELU(),
                              nn.Linear(HEAD_HIDDEN_DIM, PROJECTION_DIM))
    predictor = nn.Sequential(nn.Linear(PROJECTION_DIM, HEAD_HIDDEN_DIM), nn.GELU(),
                              nn.Linear(HEAD_HIDDEN_DIM, PROJECTION_DIM))
    return projector, predictor


def cosine_anchor_loss(features, anchor):
    """Optional cosine anchor to the frozen public DINOv2 CLS features (declared in result.json)."""
    return (1 - F.cosine_similarity(features, anchor, dim=-1)).mean()


def make_view_transform():
    """224 crop, flip, and ImageNet normalization; same contract as dino_ibot_runtime.augment."""
    from torchvision import transforms as T  # lazy: torchvision is only needed on the GPU host

    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def check_manifest_rows(rows):
    """Validates row schema only: image_path and allowed role; label-bearing keys are rejected, values never read."""
    if not rows:
        raise ValueError("Manifest contains no rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("image_path"):
            raise ValueError(f"manifest row {index} must contain image_path")
        if row.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"manifest row {index} has a role outside source_fit/pad_adapt")
        forbidden = FORBIDDEN_LABEL_FIELDS & row.keys()
        if forbidden:
            raise ValueError(f"manifest row {index} carries label fields: {sorted(forbidden)}")
