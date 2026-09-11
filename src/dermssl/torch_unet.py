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


def train_unet(
    train_loader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> UNet:
    """Train a fresh ``UNet`` with BCE-with-logits loss, Adam optimizer.

    Runs once on Kaggle (D4); the caller is responsible for saving the
    result via :func:`save_checkpoint` if it should outlive the session.
    """
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

    return model


def save_checkpoint(model: UNet, path: str) -> None:
    torch.save({"model_state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, device: torch.device) -> UNet:
    model = UNet().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def infer_mask(model: UNet, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Run ``model`` on one sRGB image, float ``(H, W, 3)`` in ``[0, 1]``.

    Returns a bool ``(H, W)`` mask, ``True`` = lesion - the exact contract
    ``fitzpatrick.estimate_fst``'s ``mask`` parameter expects (D3): resized
    to ``INPUT_RESOLUTION`` for the forward pass, then the predicted
    probability map is resized back to the image's original shape before
    thresholding, so the returned mask always matches ``img.shape[:2]``.
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

    mask = probs.squeeze(0).squeeze(0) > MASK_THRESHOLD
    return mask.cpu().numpy().astype(bool)
