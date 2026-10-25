"""Semi-supervised Teacher-Student trainer with multi-scale semantic foundation model guidance.

Implements:
1. Student network (trainable segmentation architecture).
2. Teacher network (EMA-updated from student weights).
3. Supervised loss (BCE + Dice) over the certified labeled subset (e.g. PH2).
4. Consistency regularization over the unlabeled pool (e.g. ISIC 2019) with
   DermLIP semantic logit fusion and Shannon-entropy uncertainty gating.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dermssl.segmentation_methods.dermlip_slic.tripleprior import SuperpixelAffinityLoss

LOGGER = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Two-layer 3x3 Convolution block with BatchNorm and ReLU."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DermLIPSegmentationUNet(nn.Module):
    """Standard U-Net segmentation network with skip connections for skin lesion segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        c = self.center(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


def compute_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss over logits for single-channel binary segmentation."""
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)

    intersection = (probs_flat * targets_flat).sum(dim=-1)
    cardinality = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
    dice_coeff = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice_coeff.mean()


class TeacherStudentTrainer:
    """Semi-Supervised Teacher-Student trainer with EMA and semantic prior fusion."""

    def __init__(
        self,
        student: nn.Module,
        teacher: Optional[nn.Module] = None,
        device: str = "cpu",
        ema_decay: float = 0.99,
        gamma_semantic: float = 0.5,
        kappa_uncertainty: float = 0.693147,  # ln(2)
        supervised_weight: float = 1.0,
        unsupervised_weight: float = 1.0,
        superpixel_affinity_weight: float = 0.0,
    ) -> None:
        self.device = device
        self.student = student.to(device)
        if teacher is not None:
            self.teacher = teacher.to(device)
        else:
            self.teacher = copy.deepcopy(student).to(device)

        # Freeze teacher weights from backpropagation
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

        self.ema_decay = ema_decay
        self.gamma_semantic = gamma_semantic
        self.kappa_uncertainty = kappa_uncertainty
        self.supervised_weight = supervised_weight
        self.unsupervised_weight = unsupervised_weight
        self.superpixel_affinity_weight = superpixel_affinity_weight
        self.superpixel_affinity_loss_fn = SuperpixelAffinityLoss()

    def update_teacher(self, step: Optional[int] = None) -> None:
        """Update teacher model parameters and BatchNorm running statistics via EMA.

        If step is provided, applies adaptive EMA warm-up:
            decay = min(1.0 - 1.0 / (step + 1), self.ema_decay)
        to prevent initial random weights from dominating the teacher model.
        """
        decay = self.ema_decay
        if step is not None:
            decay = min(1.0 - 1.0 / (float(step) + 1.0), self.ema_decay)

        with torch.no_grad():
            # 1. Update trainable weights
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

            # 2. Synchronize non-trainable buffers (BatchNorm running_mean, running_var, etc.)
            for t_buf, s_buf in zip(self.teacher.buffers(), self.student.buffers()):
                if t_buf.is_floating_point():
                    t_buf.data.mul_(decay).add_(s_buf.data, alpha=1.0 - decay)
                else:
                    t_buf.data.copy_(s_buf.data)

    def supervised_loss(self, student_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Compute combined BCE + Dice loss on labeled images."""
        loss_bce = F.binary_cross_entropy_with_logits(student_logits, target_masks)
        loss_dice = compute_dice_loss(student_logits, target_masks)
        return loss_bce + loss_dice

    def unsupervised_loss(
        self,
        student_logits_strong: torch.Tensor,
        teacher_logits_weak: torch.Tensor,
        semantic_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute consistency loss regularized by DermLIP semantic guidance and entropy gating.

        Returns:
            Tuple of (loss, mean_uncertainty_weight, pseudo_labels)
        """
        eps = 1e-6
        with torch.no_grad():
            fused_logits = teacher_logits_weak
            if semantic_probs is not None:
                # Project semantic probabilities into logit space
                sem_clamped = torch.clamp(semantic_probs, eps, 1.0 - eps)
                sem_logits = torch.log(sem_clamped / (1.0 - sem_clamped))
                fused_logits = fused_logits + self.gamma_semantic * sem_logits

            calibrated_probs = torch.sigmoid(fused_logits)

            # Binary Shannon entropy in base 2: H_2(p) = -p log2(p) - (1-p) log2(1-p) in [0, 1].
            # At p=0.5 (maximum clinical uncertainty), H_2(0.5) = 1.0.
            # With kappa = ln(2) ~= 0.693147, alpha = exp(-1.0 / ln(2)) ~= 0.23629,
            # attenuating noisy unannotated regions down to ~23.6% of full weight.
            p_c = torch.clamp(calibrated_probs, eps, 1.0 - eps)
            entropy = -(p_c * torch.log2(p_c) + (1.0 - p_c) * torch.log2(1.0 - p_c))

            # Confidence weighting: alpha(x) = exp(-H_2(x) / kappa)
            alpha_weights = torch.exp(-entropy / self.kappa_uncertainty)

            # Hard pseudo-label threshold at 0.5
            pseudo_targets = (calibrated_probs > 0.5).float()

        # Weighted pixel-wise BCE loss
        bce_pixel = F.binary_cross_entropy_with_logits(
            student_logits_strong, pseudo_targets, reduction="none"
        )
        weighted_loss = (alpha_weights * bce_pixel).mean()
        return weighted_loss, alpha_weights.mean(), pseudo_targets

    def training_step(
        self,
        labeled_batch: Tuple[torch.Tensor, torch.Tensor],
        unlabeled_batch: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        lambda_u: float = 1.0,
        unlabeled_slic_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Perform a single forward-backward optimization step across labeled & unlabeled pairs."""
        images_l, masks_l = labeled_batch
        images_l = images_l.to(self.device)
        masks_l = masks_l.to(self.device)

        images_u_weak, images_u_strong, sem_probs_u = unlabeled_batch
        images_u_weak = images_u_weak.to(self.device)
        images_u_strong = images_u_strong.to(self.device)
        if sem_probs_u is not None:
            sem_probs_u = sem_probs_u.to(self.device)

        # 1. Supervised forward on labeled images
        student_logits_l = self.student(images_l)
        loss_sup = self.supervised_loss(student_logits_l, masks_l)

        # 2. Unsupervised forward on unlabeled images
        with torch.no_grad():
            teacher_logits_w = self.teacher(images_u_weak)

        student_logits_s = self.student(images_u_strong)
        loss_unsup, mean_alpha, _ = self.unsupervised_loss(
            student_logits_s, teacher_logits_w, sem_probs_u
        )

        # 3. Superpixel affinity loss (differentiable boundary inductive bias)
        loss_sp = torch.tensor(0.0, device=self.device)
        if self.superpixel_affinity_weight > 0.0 and unlabeled_slic_labels is not None:
            student_probs_s = torch.sigmoid(student_logits_s)
            loss_sp = self.superpixel_affinity_loss_fn(student_probs_s, unlabeled_slic_labels)

        total_loss = (
            self.supervised_weight * loss_sup
            + lambda_u * self.unsupervised_weight * loss_unsup
            + lambda_u * self.superpixel_affinity_weight * loss_sp
        )

        return {
            "total_loss": total_loss,
            "loss_sup": loss_sup,
            "loss_unsup": loss_unsup,
            "loss_sp": loss_sp,
            "mean_alpha": mean_alpha,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: Any) -> Dict[str, float]:
        """Evaluate both student and teacher models on a validation/test dataset."""
        self.student.eval()
        self.teacher.eval()
        s_dices, s_ious = [], []
        t_dices, t_ious = [], []
        validation_cases = 0

        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            s_preds = (torch.sigmoid(self.student(images)) > 0.5).float()
            t_preds = (torch.sigmoid(self.teacher(images)) > 0.5).float()

            for sp, tp, m in zip(s_preds, t_preds, masks):
                validation_cases += 1
                # Student metrics
                s_inter = (sp * m).sum().item()
                s_tot = sp.sum().item() + m.sum().item()
                s_dices.append((2.0 * s_inter + 1e-6) / (s_tot + 1e-6))
                s_ious.append((s_inter + 1e-6) / (s_tot - s_inter + 1e-6))

                # Teacher metrics
                t_inter = (tp * m).sum().item()
                t_tot = tp.sum().item() + m.sum().item()
                t_dices.append((2.0 * t_inter + 1e-6) / (t_tot + 1e-6))
                t_ious.append((t_inter + 1e-6) / (t_tot - t_inter + 1e-6))

        s_dice_mean = float(np.mean(s_dices)) if s_dices else 0.0
        s_iou_mean = float(np.mean(s_ious)) if s_ious else 0.0
        t_dice_mean = float(np.mean(t_dices)) if t_dices else 0.0
        t_iou_mean = float(np.mean(t_ious)) if t_ious else 0.0

        return {
            "student_dice": s_dice_mean,
            "student_iou": s_iou_mean,
            "teacher_dice": t_dice_mean,
            "teacher_iou": t_iou_mean,
            "ph2_validation_cases": validation_cases,
        }
