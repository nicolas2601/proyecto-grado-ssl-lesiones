"""Stage 2 joint regularization: an MSE term between the trainable probe's
prediction on the original image and its FST-shifted synthetic counterpart
(``dermssl.circle.shift_skin_tone``), applied only to the probe head. The
backbone stays frozen for the whole of Stage 2 - the same "no label, no
gradient ever touches the backbone" promise v5's frozen probes already make,
carried into Stage 2 training (spec.md, circle-adaptation, "Joint
Regularization at the Probe Level").

Why this module exists, and what it is NOT
--------------------------------------------
This is Stage 2's GPU-side machinery, built ahead of the real Stage 1 Kaggle
run (PR3's ``scripts/kaggle_fst_manifest.py`` is still a pending manual step -
see docs/20). Nothing here measures or claims a real fairness improvement:
these are the training primitives (freeze contract, probe-only optimizer,
joint MSE loss, one training step) that a human will wire into an actual
Kaggle run once ``fst_manifest.csv`` exists and Stage 1's fairness gate
reports measurable bias (``dermssl.circle.should_run_stage2``).

GPU-only: imports torch at module level, same convention as
``finetune/torch_run.py`` and ``dermssl/torch_unet.py``. Cannot run in this
environment (no GPU, no torch installed) - checked by ``py_compile`` + ast
only (``tests/test_circle_torch_static.py``), mirrors
``tests/test_torch_unet_static.py``, itself modeled on
``tests/test_pretrain_torch_layer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_backbone(backbone: nn.Module) -> None:
    """Set every backbone parameter's ``requires_grad`` to False, in place,
    and put the backbone in eval mode.

    Called once before Stage 2 training starts. ``stage2_train_step`` also
    asserts this invariant on every step (defense in depth: a caller that
    forgets to freeze, or that later re-enables a parameter by mistake, fails
    loudly instead of silently updating the backbone).
    """
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()


class ProbeHead(nn.Module):
    """The only trainable part of Stage 2: one linear layer on top of frozen
    backbone features, exactly like v5's ``logreg`` probe but trained with
    gradient descent instead of scikit-learn so the joint MSE term can flow
    into it."""

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_probe_optimizer(probe: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """Optimizer scoped to the probe head ONLY - never the backbone.

    A separate function (rather than inlining ``torch.optim.AdamW(probe.
    parameters())`` at every call site) so the "backbone parameters never
    reach an optimizer" invariant has exactly one place to audit.
    """
    return torch.optim.AdamW(probe.parameters(), lr=lr)


def joint_regularization_loss(
    probe: nn.Module,
    features_original: torch.Tensor,
    features_shifted: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    reg_weight: float = 1.0,
) -> torch.Tensor:
    """Task loss on the original features, plus an MSE regularization term
    between the probe's output on the original and FST-shifted features.

    ``features_original``/``features_shifted`` are frozen-backbone outputs
    for the same batch, with the shifted half produced by
    ``dermssl.circle.shift_skin_tone`` upstream (numpy, CPU) before the
    backbone forward pass. Never a replacement for the task loss - added to
    it, per spec.md req 3.
    """
    logits_original = probe(features_original)
    logits_shifted = probe(features_shifted)
    task_loss = criterion(logits_original, targets)
    regularization = F.mse_loss(logits_original, logits_shifted)
    return task_loss + reg_weight * regularization


def stage2_train_step(
    backbone: nn.Module,
    probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_original: torch.Tensor,
    batch_shifted: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    reg_weight: float = 1.0,
) -> dict:
    """One Stage 2 training step.

    Asserts the frozen-backbone invariant BEFORE touching the optimizer (spec.md
    req 3 scenario: "backbone parameters have requires_grad=False ... only
    probe-head parameters receive updates"). Backbone features are computed
    under ``torch.no_grad()`` for both the original and FST-shifted batch, so
    the backbone contributes no autograd graph at all - not merely "no
    gradient by virtue of requires_grad=False", but structurally unreachable
    by ``loss.backward()``.
    """
    assert all(not p.requires_grad for p in backbone.parameters()), (
        "Stage 2 invariant violated: backbone must stay frozen"
    )
    backbone.eval()
    with torch.no_grad():
        features_original = backbone(batch_original)
        features_shifted = backbone(batch_shifted)

    optimizer.zero_grad(set_to_none=True)
    loss = joint_regularization_loss(
        probe, features_original, features_shifted, targets, criterion, reg_weight=reg_weight
    )
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}
