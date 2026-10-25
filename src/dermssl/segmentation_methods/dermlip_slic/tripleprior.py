"""Derm-TriplePrior: Boundary and Superpixel Affinity Losses and Peripheral Skin Ring Calibration.

Implements the novel differentiable inductive biases for the proposed architecture:
1. SuperpixelAffinityLoss (L_sp):
   Differentiable intra-cluster consistency loss enforcing that student predicted
   probabilities within each coherent SLIC superpixel remain homogeneous.
2. PeripheralRingNormalizer:
   Extracts dynamic perilesional healthy skin representations from peripheral FOV
   superpixels, calibrating semantic DermLIP similarities to the patient's specific
   phototype (Fitzpatrick invariance).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class SuperpixelAffinityLoss(nn.Module):
    """Differentiable intra-superpixel variance loss.

    Enforces that student network predictions within the same morphological cluster
    (SLIC superpixel) exhibit minimal variance, guiding the gradient to respect
    true cutaneous boundaries and fine notches without blurring or erosion.

    Mathematical definition:
        L_sp = (1 / K) * sum_{k=1}^K Var_{u in S_k}(p(u))
             = (1 / K) * sum_{k=1}^K ( (1 / |S_k|) sum_{u in S_k} (p(u) - mean_k)^2 )
    """

    def __init__(self, eps: float = 1e-7, max_superpixels: int = 400) -> None:
        super().__init__()
        self.eps = eps
        self.max_superpixels = max_superpixels

    def forward(
        self,
        student_probs: torch.Tensor,
        slic_labels: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Calculate the superpixel affinity loss for a batch.

        Args:
            student_probs: [B, 1, H, W] or [B, H, W] predicted probabilities in [0, 1].
            slic_labels: [B, H, W] integer tensor or ndarray of superpixel segment IDs.

        Returns:
            Scalar differentiable loss tensor.
        """
        if student_probs.dim() == 4:
            probs = student_probs.squeeze(1)  # [B, H, W]
        else:
            probs = student_probs

        device = probs.device
        if isinstance(slic_labels, np.ndarray):
            labels = torch.from_numpy(slic_labels).to(device=device, dtype=torch.long)
        else:
            labels = slic_labels.to(device=device, dtype=torch.long)

        batch_size = probs.shape[0]
        batch_losses = []

        for b in range(batch_size):
            p_b = probs[b]      # [H, W]
            l_b = labels[b]     # [H, W]

            unique_ids = torch.unique(l_b)
            if len(unique_ids) > self.max_superpixels:
                # Subsample to keep memory & compute strictly bounded
                perm = torch.randperm(len(unique_ids), device=device)[: self.max_superpixels]
                unique_ids = unique_ids[perm]

            k_count = unique_ids.shape[0]
            if k_count == 0:
                continue

            # Vectorized variance computation per superpixel
            # One-hot mask for selected superpixels: [K, H, W]
            masks = (l_b.unsqueeze(0) == unique_ids.view(-1, 1, 1)).float()  # [K, H, W]
            cluster_sizes = masks.sum(dim=(1, 2)).clamp(min=1.0)              # [K]

            # Mean probability inside each cluster
            cluster_means = (masks * p_b.unsqueeze(0)).sum(dim=(1, 2)) / cluster_sizes  # [K]

            # Variance inside each cluster: mean((p - mean)^2)
            diffs = (p_b.unsqueeze(0) - cluster_means.view(-1, 1, 1)) * masks  # [K, H, W]
            cluster_vars = (diffs ** 2).sum(dim=(1, 2)) / cluster_sizes        # [K]

            batch_losses.append(cluster_vars.mean())

        if not batch_losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(batch_losses).mean()


class PeripheralRingNormalizer:
    """Dynamic perilesional healthy skin reference extractor.

    In dermatoscopy, the outer perimeter of the field-of-view (FOV) represents
    uninvolved, healthy surrounding skin in >98% of cases. By extracting the
    average semantic and color signature from this peripheral ring, we calibrate
    lesion vs. skin contrast specifically to the patient's individual baseline.
    """

    def __init__(self, ring_margin_frac: float = 0.08) -> None:
        """Initialize peripheral ring normalizer.

        Args:
            ring_margin_frac: Fraction of image dimensions used as peripheral border (e.g. 8%).
        """
        self.ring_margin_frac = ring_margin_frac

    def create_peripheral_ring_mask(self, height: int, width: int) -> np.ndarray:
        """Create a binary mask selecting the peripheral ring of the image."""
        mask = np.zeros((height, width), dtype=bool)
        m_h = int(height * self.ring_margin_frac)
        m_w = int(width * self.ring_margin_frac)

        mask[:m_h, :] = True
        mask[-m_h:, :] = True
        mask[:, :m_w] = True
        mask[:, -m_w:] = True
        return mask

    def extract_skin_baseline(
        self,
        image_rgb: np.ndarray,
        dense_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """Extract baseline color (CIE-Lab / RGB) and optional dense VLM baseline vector.

        Args:
            image_rgb: [H, W, 3] uint8 numpy array.
            dense_tokens: Optional [C, H_tok, W_tok] dense patch embeddings from PanDerm.

        Returns:
            Tuple of (mean_skin_color_rgb, optional_mean_skin_token).
        """
        h, w = image_rgb.shape[:2]
        ring_mask = self.create_peripheral_ring_mask(h, w)
        ring_pixels = image_rgb[ring_mask]

        if len(ring_pixels) > 0:
            mean_color = ring_pixels.mean(axis=0)
        else:
            mean_color = np.array([128.0, 128.0, 128.0])

        mean_token = None
        if dense_tokens is not None:
            # Downsample ring mask to token grid
            _, h_tok, w_tok = dense_tokens.shape
            tok_mask = self.create_peripheral_ring_mask(h_tok, w_tok)
            tok_mask_t = torch.from_numpy(tok_mask).to(dense_tokens.device)
            # Pool token vectors from peripheral region
            selected_tokens = dense_tokens[:, tok_mask_t]  # [C, N_ring]
            if selected_tokens.shape[1] > 0:
                mean_token = F.normalize(selected_tokens.mean(dim=1, keepdim=True), p=2, dim=0)

        return mean_color, mean_token
