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
        self.fp_dropout = nn.Dropout2d(p=0.3)

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

    def forward_with_fp(self, x: torch.Tensor) -> torch.Tensor:
        """UniMatch feature-level dropout perturbation forward pass (CVPR 2023)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.fp_dropout(self.enc4(self.pool3(e3)))

        c = self.fp_dropout(self.center(self.pool4(e4)))

        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


class _ResNetBasicBlock(nn.Module):
    """Standard ResNet BasicBlock matching torchvision.models.resnet34 state_dict layout."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _ResNetBottleneck(nn.Module):
    """Standard ResNet Bottleneck matching torchvision.models.resnet50 state_dict layout."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        width = planes
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _PureTorchResNetBackbone(nn.Module):
    """Pure-PyTorch ResNet-34/50 encoder compatible with official torchvision ImageNet checkpoints."""

    _URLS = {
        "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
        "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    }

    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if backbone_name == "resnet34":
            block_cls = _ResNetBasicBlock
            layers = [3, 4, 6, 3]
        elif backbone_name == "resnet50":
            block_cls = _ResNetBottleneck
            layers = [3, 4, 6, 3]
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name!r}")

        self.layer1 = self._make_layer(block_cls, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_cls, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_cls, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_cls, 512, layers[3], stride=2)

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                self._URLS[backbone_name], progress=True, check_hash=True
            )
            encoder_state = {
                k: v for k, v in state_dict.items() if not k.startswith("fc.")
            }
            self.load_state_dict(encoder_state, strict=True)

    def _make_layer(
        self, block_cls: Any, planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block_cls.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_cls.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block_cls.expansion),
            )
        layers = [block_cls(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block_cls.expansion
        for _ in range(1, blocks):
            layers.append(block_cls(self.inplanes, planes))
        return nn.Sequential(*layers)


class DermLIPResNetUNet(nn.Module):
    """Pretrained ResNet-34/50 U-Net with full-resolution skip connection and UniMatch FP stream.

    Addresses confirmation bias in low-label semi-supervised segmentation (Arazo et al., IJCNN 2020;
    UniMatch, Yang et al., CVPR 2023) while preserving full-resolution (1/1) spatial detail for
    micro-lesions (<2% area) via a parallel full-resolution stem skip connection.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        fp_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if backbone_name == "resnet34":
            ch = [64, 64, 128, 256, 512]
        elif backbone_name == "resnet50":
            ch = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name!r}")

        base = _PureTorchResNetBackbone(backbone_name=backbone_name, pretrained=pretrained)

        # Full-resolution (1/1) shallow boundary branch for micro-lesion fidelity
        self.stem_full = ConvBlock(in_channels, 32)

        # Pretrained ResNet encoder stages: 1/2, 1/4, 1/8, 1/16, 1/32
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool0 = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.fp_dropout = nn.Dropout2d(p=fp_dropout_p)

        # Multi-scale U-Net decoder with bilinear upsampling (checkerboard-free)
        self.dec4 = ConvBlock(ch[4] + ch[3], 256)
        self.dec3 = ConvBlock(256 + ch[2], 128)
        self.dec2 = ConvBlock(128 + ch[1], 64)
        self.dec1 = ConvBlock(64 + ch[0], 64)
        self.dec0 = ConvBlock(64 + 32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if x.shape[1] == 3:
            x_norm = (x - self.imagenet_mean.to(x.dtype)) / self.imagenet_std.to(x.dtype)
        else:
            x_norm = x
        s0 = self.stem_full(x_norm)
        e0 = self.enc0(x_norm)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return s0, e0, e1, e2, e3, e4

    def _decode(
        self,
        s0: torch.Tensor,
        e0: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        e4: torch.Tensor,
    ) -> torch.Tensor:
        u4 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([u4, e3], dim=1))

        u3 = F.interpolate(d4, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e2], dim=1))

        u2 = F.interpolate(d3, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        u1 = F.interpolate(d2, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e0], dim=1))

        u0 = F.interpolate(d1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([u0, s0], dim=1))
        return self.final(d0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0, e0, e1, e2, e3, e4 = self._encode(x)
        return self._decode(s0, e0, e1, e2, e3, e4)

    def forward_with_fp(self, x: torch.Tensor) -> torch.Tensor:
        """UniMatch feature-level perturbation forward pass (Yang et al., CVPR 2023)."""
        s0, e0, e1, e2, e3, e4 = self._encode(x)
        e3_fp = self.fp_dropout(e3)
        e4_fp = self.fp_dropout(e4)
        return self._decode(s0, e0, e1, e2, e3_fp, e4_fp)

    def get_parameter_groups(self, base_lr: float, weight_decay: float = 1e-4) -> list[Dict[str, Any]]:
        """Differential learning rates: 0.5x on pretrained ResNet encoder, 2.0x on U-Net decoder."""
        encoder_modules = [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]
        encoder_params = [p for m in encoder_modules for p in m.parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids]
        return [
            {"params": encoder_params, "lr": base_lr * 0.5, "weight_decay": weight_decay},
            {"params": decoder_params, "lr": base_lr * 2.0, "weight_decay": weight_decay},
        ]


def build_segmentation_model(
    backbone: str = "scratch",
    in_channels: int = 3,
    num_classes: int = 1,
    base_channels: int = 16,
    pretrained: bool = True,
) -> nn.Module:
    """Factory function for constructing either the scratch U-Net or pretrained ResNet U-Net."""
    if backbone in {"scratch", "unet", "unet_scratch"}:
        return DermLIPSegmentationUNet(
            in_channels=in_channels, num_classes=num_classes, base_channels=base_channels
        )
    if backbone in {"resnet34", "resnet50"}:
        return DermLIPResNetUNet(
            backbone_name=backbone,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    raise ValueError(f"Unsupported segmentation backbone: {backbone!r}")


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
        pseudo_label_threshold_high: float = 0.65,
        pseudo_label_threshold_low: float = 0.35,
        soft_kl_weight: float = 0.0,
        use_log_linear_fusion: bool = False,
        copy_paste_prob: float = 0.0,
        use_fp_dropout: bool = False,
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
        self.pseudo_label_threshold_high = pseudo_label_threshold_high
        self.pseudo_label_threshold_low = pseudo_label_threshold_low
        self.soft_kl_weight = float(np.clip(soft_kl_weight, 0.0, 1.0))
        self.use_log_linear_fusion = bool(use_log_linear_fusion)
        self.copy_paste_prob = float(np.clip(copy_paste_prob, 0.0, 1.0))
        self.use_fp_dropout = bool(use_fp_dropout)

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

    def _apply_copy_paste_scale_jitter(
        self,
        images_l: torch.Tensor,
        masks_l: torch.Tensor,
        images_u_weak: torch.Tensor,
        teacher_logits_w: torch.Tensor,
        sem_probs_u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale-jittered Copy-Paste / BCP augmentation (Ghiasi et al., CVPR 2021; Bai et al., CVPR 2023).

        Extracts real foreground lesions from PH2 using their ground-truth mask, downscales them
        into the micro/small lesion regime (0.18x–0.65x linear scale), feathers the boundary,
        and pastes them onto clean real dermoscopic skin backgrounds from the unlabeled batch.
        """
        if self.copy_paste_prob <= 0.0:
            return images_l, masks_l

        b_size, _, h, w = images_l.shape
        if h < 16 or w < 16 or images_u_weak.shape != images_l.shape:
            return images_l, masks_l

        out_imgs = images_l.clone()
        out_masks = masks_l.clone()
        with torch.no_grad():
            t_probs = torch.sigmoid(teacher_logits_w)
            for b in range(b_size):
                if float(torch.rand(1).item()) >= self.copy_paste_prob:
                    continue
                fg_mask = masks_l[b, 0] > 0.5
                if float(fg_mask.float().mean().item()) < 0.03:
                    continue

                ys, xs = torch.where(fg_mask)
                if ys.numel() < 16:
                    continue
                y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
                x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
                box_h, box_w = y1 - y0, x1 - x0
                if box_h < 8 or box_w < 8:
                    continue

                # Sample linear scale factor in [0.18, 0.65] -> area scale ~3% to 42% of original lesion
                scale = float(torch.empty(1).uniform_(0.18, 0.65).item())
                new_h = max(8, min(int(round(h * 0.70)), int(round(box_h * scale))))
                new_w = max(8, min(int(round(w * 0.70)), int(round(box_w * scale))))

                crop_img = images_l[b : b + 1, :, y0:y1, x0:x1]
                crop_mask = masks_l[b : b + 1, :, y0:y1, x0:x1].float()

                resized_img = F.interpolate(
                    crop_img, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                resized_mask = F.interpolate(
                    crop_mask, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                # Feather the pasted lesion boundary (3x3 box filter) to avoid artificial step edges
                alpha_paste = F.avg_pool2d(resized_mask, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
                hard_paste_mask = (resized_mask >= 0.5).float()

                # Construct a clean dermoscopic skin canvas from the unlabeled image
                u_img = images_u_weak[b].clone()
                u_lesion = t_probs[b, 0] > 0.35
                if sem_probs_u is not None and sem_probs_u.shape[-2:] == (h, w):
                    u_lesion = u_lesion | (sem_probs_u[b, 0] > 0.35)
                # Also exclude dark camera corners (vignettes)
                u_valid_skin = (~u_lesion) & (u_img.mean(dim=0) > 0.12)
                if int(u_valid_skin.sum().item()) >= 64:
                    skin_pixels = u_img[:, u_valid_skin]  # [3, N_skin]
                    n_skin = skin_pixels.shape[1]
                    bad_mask = ~u_valid_skin
                    n_bad = int(bad_mask.sum().item())
                    if n_bad > 0:
                        rand_idx = torch.randint(0, n_skin, (n_bad,), device=u_img.device)
                        u_img[:, bad_mask] = skin_pixels[:, rand_idx]
                        # Smooth replaced region so texture is continuous
                        u_img_blur = F.avg_pool2d(
                            u_img.unsqueeze(0), kernel_size=5, stride=1, padding=2
                        ).squeeze(0)
                        u_img = torch.where(bad_mask.unsqueeze(0), u_img_blur, u_img)

                margin_y = max(4, int(0.08 * h))
                margin_x = max(4, int(0.08 * w))
                max_py = max(margin_y + 1, h - new_h - margin_y)
                max_px = max(margin_x + 1, w - new_w - margin_x)
                py = int(torch.randint(margin_y, max_py, (1,)).item()) if max_py > margin_y else max(0, (h - new_h) // 2)
                px = int(torch.randint(margin_x, max_px, (1,)).item()) if max_px > margin_x else max(0, (w - new_w) // 2)

                canvas_img = u_img.clone()
                canvas_mask = torch.zeros((1, h, w), device=images_l.device, dtype=masks_l.dtype)

                a_patch = alpha_paste[0]  # [1, new_h, new_w]
                canvas_img[:, py : py + new_h, px : px + new_w] = (
                    (1.0 - a_patch) * canvas_img[:, py : py + new_h, px : px + new_w]
                    + a_patch * resized_img[0]
                )
                canvas_mask[:, py : py + new_h, px : px + new_w] = hard_paste_mask[0]

                out_imgs[b] = canvas_img
                out_masks[b] = canvas_mask

        return out_imgs, out_masks

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
            teacher_probs = torch.sigmoid(teacher_logits_weak)
            has_active_semantic = (
                semantic_probs is not None
                and self.gamma_semantic > 0.0
                and bool((semantic_probs - 0.5).abs().max().item() > 1e-4)
            )
            if has_active_semantic and semantic_probs is not None:
                sem_clamped = torch.clamp(semantic_probs, 1e-4, 1.0 - 1e-4)
                # Pixel-wise teacher confidence in [0, 1]
                teacher_conf = (2.0 * (teacher_probs - 0.5).abs()).clamp(0.0, 1.0)
                # Confidence-gated semantic fusion: blend stronger where teacher is uncertain
                gate_weight = (self.gamma_semantic * (1.0 - 0.5 * teacher_conf)).clamp(0.0, 1.0)
                if self.use_log_linear_fusion:
                    # Log-linear probability fusion (Product of Experts, TiSage MICCAI 2026)
                    t_logits = torch.logit(teacher_probs.clamp(1e-4, 1.0 - 1e-4))
                    s_logits = torch.logit(sem_clamped)
                    fused_probs = torch.sigmoid((1.0 - gate_weight) * t_logits + gate_weight * s_logits)
                else:
                    fused_probs = (1.0 - gate_weight) * teacher_probs + gate_weight * sem_clamped
                # Peripheral ring / FOV background anchor suppression against over-segmentation.
                # Attenuate teacher false alarms whenever the peripheral calibrated semantic prior indicates background.
                bg_suppress_mask = (sem_clamped < 0.35) & (teacher_probs > 0.50)
                fused_probs = torch.where(
                    bg_suppress_mask,
                    0.30 * teacher_probs + 0.70 * sem_clamped,
                    fused_probs,
                )
                fused_logits = torch.logit(fused_probs.clamp(1e-4, 1.0 - 1e-4))
                calibrated_probs = torch.sigmoid(fused_logits)
            else:
                calibrated_probs = teacher_probs

            # Binary Shannon entropy in base 2: H_2(p) = -p log2(p) - (1-p) log2(1-p) in [0, 1].
            # At p=0.5 (maximum clinical uncertainty), H_2(p=0.5) = 1.0.
            # With kappa = ln(2) ~= 0.693147, alpha = exp(-1.0 / ln(2)) ~= 0.23629,
            # attenuating noisy unannotated regions down to ~23.6% of full weight.
            p_c = torch.clamp(calibrated_probs, eps, 1.0 - eps)
            entropy = -(p_c * torch.log2(p_c) + (1.0 - p_c) * torch.log2(1.0 - p_c))

            # Confidence weighting: alpha(x) = exp(-H_2(x) / kappa)
            alpha_weights = torch.exp(-entropy / self.kappa_uncertainty)

            # Confident pseudo-label thresholding with uncertainty band masking:
            # Confident foreground (p >= tau_high) -> 1.0
            # Confident background (p <= tau_low)  -> 0.0
            # Ambiguous boundary band (tau_low < p < tau_high) -> weight zeroed out for hard target
            confident_fg = calibrated_probs >= self.pseudo_label_threshold_high
            confident_bg = calibrated_probs <= self.pseudo_label_threshold_low
            confident_mask = confident_fg | confident_bg

            pseudo_targets = confident_fg.float()
            effective_weights = alpha_weights * confident_mask.float()

        # 1. Confidence-weighted pixel-wise BCE loss on confident regions
        bce_pixel = F.binary_cross_entropy_with_logits(
            student_logits_strong, pseudo_targets, reduction="none"
        )
        bce_unsup = (effective_weights * bce_pixel).mean()

        # 2. Confidence-weighted Soft Dice loss for global overlap & small lesion sensitivity
        student_probs_strong = torch.sigmoid(student_logits_strong)
        inter_u = (student_probs_strong * pseudo_targets * effective_weights).sum(dim=(1, 2, 3))
        union_u = ((student_probs_strong + pseudo_targets) * effective_weights).sum(dim=(1, 2, 3))
        dice_u = 1.0 - ((2.0 * inter_u + 1e-5) / (union_u + 1e-5)).mean()

        loss_hard = 0.5 * bce_unsup + 0.5 * dice_u

        # 3. Optional continuous entropy-weighted Soft KL / Bernoulli cross-entropy distillation
        # (UA-MT, Yu et al., MICCAI 2019; TiSage conf_weighted_kl) so micro-lesions in the
        # transition zone still receive smooth gradient guidance without hard-threshold bias.
        if self.soft_kl_weight > 0.0:
            soft_bce_pixel = F.binary_cross_entropy_with_logits(
                student_logits_strong, calibrated_probs.detach(), reduction="none"
            )
            loss_soft_kl = (alpha_weights * soft_bce_pixel).mean()
            weighted_loss = (1.0 - self.soft_kl_weight) * loss_hard + self.soft_kl_weight * loss_soft_kl
        else:
            weighted_loss = loss_hard

        return weighted_loss, effective_weights.mean(), pseudo_targets

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

        # Teacher forward on unmixed weak unlabeled view
        with torch.no_grad():
            teacher_logits_w_raw = self.teacher(images_u_weak)

        # Optional Scale-Jittered Copy-Paste / BCP augmentation on labeled batch
        if self.copy_paste_prob > 0.0:
            images_l, masks_l = self._apply_copy_paste_scale_jitter(
                images_l, masks_l, images_u_weak, teacher_logits_w_raw, sem_probs_u
            )

        # 1. Supervised forward on labeled images
        student_logits_l = self.student(images_l)
        loss_sup = self.supervised_loss(student_logits_l, masks_l)

        # 2. Unsupervised forward on unlabeled images with CutMix consistency (when B >= 2)
        teacher_logits_w = teacher_logits_w_raw.clone()
        sem_probs_u_mixed = sem_probs_u.clone() if sem_probs_u is not None else None

        slic_labels_mixed: Optional[Union[torch.Tensor, np.ndarray]] = unlabeled_slic_labels
        batch_u, _, h_u, w_u = images_u_strong.shape
        if batch_u >= 2 and h_u >= 4 and w_u >= 4 and teacher_logits_w.shape[-2:] == (h_u, w_u):
            area_ratio = float(torch.empty(1).uniform_(0.25, 0.50).item())
            cut_h = max(1, min(h_u - 1, int(round(h_u * (area_ratio ** 0.5)))))
            cut_w = max(1, min(w_u - 1, int(round(w_u * (area_ratio ** 0.5)))))
            cy = int(torch.randint(0, h_u - cut_h + 1, (1,)).item())
            cx = int(torch.randint(0, w_u - cut_w + 1, (1,)).item())

            cutmix_mask = torch.zeros((batch_u, 1, h_u, w_u), device=self.device, dtype=images_u_strong.dtype)
            cutmix_mask[:, :, cy : cy + cut_h, cx : cx + cut_w] = 1.0

            images_u_strong = (
                (1.0 - cutmix_mask) * images_u_strong
                + cutmix_mask * torch.roll(images_u_strong, shifts=1, dims=0)
            )
            teacher_logits_w = (
                (1.0 - cutmix_mask) * teacher_logits_w
                + cutmix_mask * torch.roll(teacher_logits_w, shifts=1, dims=0)
            )
            if sem_probs_u_mixed is not None and sem_probs_u_mixed.shape[-2:] == (h_u, w_u):
                sem_probs_u_mixed = (
                    (1.0 - cutmix_mask) * sem_probs_u_mixed
                    + cutmix_mask * torch.roll(sem_probs_u_mixed, shifts=1, dims=0)
                )
            if unlabeled_slic_labels is not None:
                if isinstance(unlabeled_slic_labels, np.ndarray):
                    slic_t = torch.from_numpy(unlabeled_slic_labels).to(device=self.device, dtype=torch.long)
                else:
                    slic_t = unlabeled_slic_labels.to(device=self.device, dtype=torch.long)
                if slic_t.shape == (batch_u, h_u, w_u):
                    slic_offset = int(slic_t.max().item()) + 1 if slic_t.numel() > 0 else 1000
                    slic_rolled = torch.roll(slic_t, shifts=1, dims=0) + slic_offset
                    slic_labels_mixed = torch.where(cutmix_mask.squeeze(1) > 0.5, slic_rolled, slic_t)

        student_logits_s = self.student(images_u_strong)
        loss_unsup_s, mean_alpha, _ = self.unsupervised_loss(
            student_logits_s, teacher_logits_w, sem_probs_u_mixed
        )

        # UniMatch dual-stream feature perturbation consistency (Yang et al., CVPR 2023)
        if self.use_fp_dropout and hasattr(self.student, "forward_with_fp"):
            student_logits_fp = self.student.forward_with_fp(images_u_weak)
            loss_unsup_fp, _, _ = self.unsupervised_loss(
                student_logits_fp, teacher_logits_w_raw, sem_probs_u
            )
            loss_unsup = 0.5 * loss_unsup_s + 0.5 * loss_unsup_fp
        else:
            loss_unsup = loss_unsup_s

        # 3. Superpixel affinity loss (differentiable boundary inductive bias)
        loss_sp = torch.tensor(0.0, device=self.device)
        if self.superpixel_affinity_weight > 0.0 and slic_labels_mixed is not None:
            student_probs_s = torch.sigmoid(student_logits_s)
            loss_sp = self.superpixel_affinity_loss_fn(student_probs_s, slic_labels_mixed)

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


DermLIPMeanTeacherTrainer = TeacherStudentTrainer
