"""U-Net lesion segmentation for Fitzpatrick skin-tone estimation (Phase 3).

Why this module exists
-----------------------
``fitzpatrick.estimate_fst`` needs a lesion mask to exclude the lesion before
computing skin ITA (D3 in design.md). The pure-numpy Otsu fallback in
``pretrain.lesion_guided`` ships today and stays the FALLBACK; this module is
the PRIMARY: a small U-Net trained once on the ISIC 2018 Task 1 challenge
(its own held-out image/mask pairs - spec.md requirement 1), run on the FST
target pool on a Kaggle GPU session.

GPU-only, ``py_compile``/``ast``-tested only (tests/test_torch_unet_static.py
- there is no torch, no GPU, in the unit-test environment). This module is
NOT part of ``dermssl.__init__``'s ``MODULE_ORDER``: the notebook-build test
inlines and EXECUTES every ``MODULE_ORDER`` module, and a module-level
``import torch`` here would break that test on a machine with no GPU stack -
exactly the reason ``dermssl.pretrain.torch_*`` modules are excluded from
that package's pure-module guard too.

D4 in design.md: the GPU/CPU seam is ``fst_manifest.csv``, produced once by
``scripts/kaggle_fst_manifest.py`` (which imports this module) and consumed
everywhere else (``metrics.stratified_summary``, ``gates.fairness_gate``) as
plain CPU-testable CSV rows. Nothing downstream of the manifest ever imports
this module.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ISIC 2018 Task 1 images vary in size; every image/mask pair is resized to
# this square resolution before entering the network, and the predicted mask
# is resized back to the original image's shape in ``infer_mask``.
INPUT_RESOLUTION = 256

# Sigmoid probability above this threshold is "lesion" in the binary mask.
MASK_THRESHOLD = 0.5

_BASE_CHANNELS = 32


class DoubleConv(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU blocks - the U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small U-Net for binary lesion segmentation.

    Four encoder/decoder stages (32/64/128/256 channels) - deep enough to
    segment dermoscopy lesions at ``INPUT_RESOLUTION``, shallow enough to
    train on a single Kaggle T4 within a smoke budget of a few minutes.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = _BASE_CHANNELS) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels * (2**i) for i in range(4))

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, c4 * 2)

        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        e1 = self.enc1(x)  # -> (B, c1, H, W)
        e2 = self.enc2(self.pool(e1))  # -> (B, c2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # -> (B, c3, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # -> (B, c4, H/8, W/8)

        b = self.bottleneck(self.pool(e4))  # -> (B, c4*2, H/16, W/16)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # -> (B, 1, H, W) logits


@dataclass
class UNetTrainingResult:
    """Best held-out checkpoint and its auditable validation history."""

    model: UNet
    best_epoch: int
    best_val_dice: float
    best_val_iou: float
    history: list[dict[str, float]]


@dataclass
class MaskPrediction:
    """Binary lesion mask plus mean sigmoid certainty in [0, 1]."""

    mask: np.ndarray
    confidence: float


class ISIC2018SegmentationDataset(Dataset):
    """Image/mask pairs from the ISIC 2018 Task 1 challenge layout.

    ``tschandl/isic2018-challenge-task1-data-segmentation`` (the Kaggle
    mirror confirmed for task 0.1) ships JPEG images and PNG masks with a
    shared stem plus the official ``_segmentation`` suffix on the mask -
    ``ISIC_0000000.jpg`` / ``ISIC_0000000_segmentation.png``.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        resolution: int = INPUT_RESOLUTION,
    ) -> None:
        self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
        self.mask_dir = Path(mask_dir)
        self.resolution = resolution
        self._resize_image = T.Resize((resolution, resolution))
        self._resize_mask = T.Resize(
            (resolution, resolution), interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _mask_path(self, image_path: Path) -> Path:
        return self.mask_dir / f"{image_path.stem}_segmentation.png"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        mask = Image.open(self._mask_path(image_path)).convert("L")

        img = self._resize_image(img)
        mask = self._resize_mask(mask)

        img_t = T.functional.to_tensor(img)
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        return img_t, mask_t


@torch.no_grad()
def evaluate_segmentation(model: UNet, val_loader, device: torch.device) -> dict[str, float]:
    """Compute global held-out BCE, Dice and IoU over one validation loader."""
    model.eval()
    loss_sum = 0.0
    pixels = 0
    intersection = 0
    predicted = 0
    target = 0
    union = 0
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, masks, reduction="sum").item())
        pixels += masks.numel()
        pred_mask = torch.sigmoid(logits) > MASK_THRESHOLD
        true_mask = masks >= 0.5
        intersection += int((pred_mask & true_mask).sum().item())
        predicted += int(pred_mask.sum().item())
        target += int(true_mask.sum().item())
        union += int((pred_mask | true_mask).sum().item())

    eps = 1e-7
    return {
        "loss": loss_sum / max(1, pixels),
        "dice": (2.0 * intersection + eps) / (predicted + target + eps),
        "iou": (intersection + eps) / (union + eps),
    }


def train_unet(
    train_loader,
    val_loader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> UNetTrainingResult:
    """Train on one split and restore the checkpoint with best held-out Dice."""
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_pixels = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * masks.numel()
            train_pixels += masks.numel()

        validation = evaluate_segmentation(model, val_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss_sum / max(1, train_pixels),
            "val_loss": validation["loss"],
            "val_dice": validation["dice"],
            "val_iou": validation["iou"],
        }
        history.append(row)
        print(
            f"epoch={epoch}/{epochs} train_loss={row['train_loss']:.5f} "
            f"val_loss={row['val_loss']:.5f} val_dice={row['val_dice']:.5f} "
            f"val_iou={row['val_iou']:.5f}"
        )
        if validation["dice"] > best_val_dice:
            best_val_dice = validation["dice"]
            best_val_iou = validation["iou"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("U-Net training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return UNetTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_val_dice=best_val_dice,
        best_val_iou=best_val_iou,
        history=history,
    )


def save_checkpoint(result: UNetTrainingResult, path: str) -> None:
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "best_epoch": result.best_epoch,
            "best_val_dice": result.best_val_dice,
            "best_val_iou": result.best_val_iou,
            "history": result.history,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> UNet:
    model = UNet().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def infer_mask_with_confidence(
    model: UNet, img: np.ndarray, device: torch.device
) -> MaskPrediction:
    """Run mask inference and return binary mask plus sigmoid certainty.

    The bool ``(H, W)`` mask, ``True`` = lesion, is the exact contract
    ``fitzpatrick.estimate_fst``'s ``mask`` parameter expects (D3): resized
    to ``INPUT_RESOLUTION`` for the forward pass, then the predicted
    probability map is resized back to the image's original shape before
    thresholding. Confidence is mean distance from the decision boundary,
    scaled to [0, 1]; it is used only to trigger the documented fallback.
    """
    model.eval()
    h, w = img.shape[:2]

    img_t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)
    img_t = F.interpolate(
        img_t, size=(INPUT_RESOLUTION, INPUT_RESOLUTION), mode="bilinear", align_corners=False
    )

    logits = model(img_t)
    probs = torch.sigmoid(logits)
    probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)

    probs = probs.squeeze(0).squeeze(0)
    mask = probs > MASK_THRESHOLD
    confidence = float((torch.abs(probs - MASK_THRESHOLD) * 2.0).mean().item())
    return MaskPrediction(mask=mask.cpu().numpy().astype(bool), confidence=confidence)


def infer_mask(model: UNet, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Compatibility wrapper returning only the binary lesion mask."""
    return infer_mask_with_confidence(model, img, device).mask
