"""Segmentacion de lesiones con U-Net para la estimacion de fototipo Fitzpatrick.

``fitzpatrick.estimate_fst`` necesita una mascara de lesion para excluirla
antes de calcular el ITA de la piel. El fallback Otsu puro-numpy de
``pretrain.lesion_guided`` ya existe y se mantiene como FALLBACK; este modulo
es el PRIMARIO: un U-Net chico entrenado una vez sobre el challenge ISIC
2018 Task 1 (con sus propios pares imagen/mascara de validacion), corrido
sobre el pool objetivo de FST en una sesion Kaggle con GPU.

Solo GPU, solo testeado por ``py_compile``/``ast``
(``tests/test_torch_unet_static.py`` - no hay torch ni GPU en el entorno de
tests unitarios). Este modulo NO forma parte de ``MODULE_ORDER`` de
``dermssl.__init__``: el test de build del notebook inlinea y EJECUTA cada
modulo de ``MODULE_ORDER``, y un ``import torch`` a nivel de modulo aca
romperia ese test en una maquina sin stack de GPU - la misma razon por la
que los modulos ``dermssl.pretrain.torch_*`` estan excluidos del chequeo de
modulos puros de ese paquete.

La costura entre GPU y CPU es ``fst_manifest.csv``, producido una vez por
``scripts/kaggle_fst_manifest.py`` (que importa este modulo) y consumido en
todos lados (``metrics.stratified_summary``, ``gates.fairness_gate``) como
filas de CSV testeables en CPU. Nada rio abajo del manifiesto importa este
modulo.
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

# Las imagenes de ISIC 2018 Task 1 varian de tamano; cada par imagen/mascara
# se redimensiona a esta resolucion cuadrada antes de entrar a la red, y la
# mascara predicha se redimensiona de vuelta al tamano original en
# ``infer_mask``.
INPUT_RESOLUTION = 256

# Probabilidad sigmoide por encima de este umbral = "lesion" en la mascara binaria.
MASK_THRESHOLD = 0.5

_BASE_CHANNELS = 32


class DoubleConv(nn.Module):
    """Dos bloques de conv 3x3 + BatchNorm + ReLU - la pieza basica del U-Net."""

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
    """U-Net chico para segmentacion binaria de lesiones.

    Cuatro etapas de encoder/decoder (32/64/128/256 canales) - suficientemente
    profundo para segmentar lesiones dermoscopicas a ``INPUT_RESOLUTION``, y
    suficientemente chico para entrenar en un solo Kaggle T4 dentro de un
    presupuesto de smoke test de pocos minutos.
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
    """Pares imagen/mascara segun el layout del challenge ISIC 2018 Task 1.

    ``tschandl/isic2018-challenge-task1-data-segmentation`` (el mirror de
    Kaggle confirmado para este dataset) trae imagenes JPEG y mascaras PNG
    con el mismo nombre base mas el sufijo oficial ``_segmentation`` en la
    mascara - ``ISIC_0000000.jpg`` / ``ISIC_0000000_segmentation.png``.
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
    seed: int = 42,
) -> UNet:
    """Entrena un ``UNet`` nuevo con perdida BCE-with-logits y optimizador Adam.

    Corre una sola vez en Kaggle; quien llama es responsable de guardar el
    resultado con :func:`save_checkpoint` si tiene que sobrevivir a la sesion.

    ``fst_manifest.csv`` alimenta ``gates.fairness_gate`` - una decision
    PASS/FAIL de fairness. Sin una semilla explicita, la inicializacion de
    pesos y la seleccion de kernels de cuDNN son no deterministas, asi que
    repetir este mismo entrenamiento (por ejemplo, recuperandose de un
    timeout de Kaggle, o re-validando un resultado) podria dar un U-Net
    distinto y un resultado distinto en el gate de fairness para los mismos
    datos de entrada. ``seed`` se usa aca; quien llama tambien es
    responsable de sembrar el shuffling de ``train_loader`` (ver el
    ``torch.Generator`` con semilla de ``scripts/kaggle_fst_manifest.py``).
    """
    torch.manual_seed(seed)
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
    """Corre ``model`` sobre una imagen sRGB, float ``(H, W, 3)`` en ``[0, 1]``.

    Devuelve una mascara bool ``(H, W)``, ``True`` = lesion - el mismo
    contrato que espera el parametro ``mask`` de
    ``fitzpatrick.estimate_fst``: se redimensiona a ``INPUT_RESOLUTION``
    para el forward pass, y el mapa de probabilidad predicho se
    redimensiona de vuelta al tamano original de la imagen antes de
    aplicar el umbral, asi la mascara devuelta siempre coincide con
    ``img.shape[:2]``.
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
