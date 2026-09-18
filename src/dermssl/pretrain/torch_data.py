"""Conjuntos Torch para preentrenamiento SSL sin etiquetas.

The ``torch_`` prefix marks the boundary of the tested core: modules with the
prefix require torch and are exercised on the cluster (and by the smoke run),
everything else in this package is pure numpy and covered by the unit suite.

Both datasets yield IMAGES ONLY. There is no code path that could attach a
label: ``config.collect_images`` never returns metadata tables, and the
__getitem__ signatures return tensors, not (tensor, target) pairs.

Normalization uses the ImageNet pixel statistics. This is required for the
DINOv2 arm (the released weights expect that normalization) and harmless for
the SimSiam arm. Pixel means/stds carry no label information - the audit
concern in docs/15 was about label supervision, not about constants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from dermssl.pretrain import color_constancy, lesion_guided

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _geometric(resolution: int, scale: tuple[float, float]) -> T.Compose:
    # La dermatoscopia no tiene orientación canónica: ambos volteos son invariancias seguras.
    return T.Compose(
        [
            T.RandomResizedCrop(resolution, scale=scale, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    )


def _photometric() -> T.Compose:
    return T.Compose(
        [
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        ]
    )


_TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _finish_view(
    view: Image.Image,
    rng: np.random.Generator,
    cc_probs: tuple[tuple[str, float], ...],
) -> torch.Tensor:
    """Aumento fotométrico, muestreo independiente de constancia de color y tensor."""
    view = _photometric()(view)
    arr = np.asarray(view, dtype=np.float64) / 255.0
    arr = color_constancy.apply_stochastic(arr, rng, cc_probs)
    view = Image.fromarray((arr * 255.0).astype(np.uint8))
    return _TO_TENSOR(view)


class SimSiamPairDataset(Dataset):
    """Dos vistas aumentadas de forma independiente por imagen."""

    def __init__(
        self,
        paths: list[Path] | list[tuple[Path, Path]],
        resolution: int,
        cc_probs: tuple[tuple[str, float], ...],
        seed: int,
    ):
        self.paths = paths
        self.geometric = _geometric(resolution, scale=(0.2, 1.0))
        self.cc_probs = cc_probs
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, idx))
        entry = self.paths[idx]
        if isinstance(entry, tuple):
            original, transformed = entry
            return (
                _finish_view(self.geometric(Image.open(original).convert("RGB")), rng, self.cc_probs),
                _finish_view(
                    self.geometric(Image.open(transformed).convert("RGB")),
                    rng,
                    self.cc_probs,
                ),
            )
        img = Image.open(entry).convert("RGB")
        return tuple(
            _finish_view(self.geometric(img), rng, self.cc_probs) for _ in range(2)
        )


class DinoMultiCropDataset(Dataset):
    """Dos crops globales y N locales, con sesgo local al borde de lesión."""

    def __init__(
        self,
        paths: list[Path],
        resolution: int,
        local_resolution: int,
        n_local: int,
        cc_probs: tuple[tuple[str, float], ...],
        lesion_guided_crops: bool,
        seed: int,
    ):
        self.paths = paths
        self.global_geo = _geometric(resolution, scale=(0.4, 1.0))
        self.local_geo = _geometric(local_resolution, scale=(0.05, 0.4))
        self.local_resolution = local_resolution
        self.n_local = n_local
        self.cc_probs = cc_probs
        self.lesion_guided_crops = lesion_guided_crops
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def _local_views(
        self, img: Image.Image, rng: np.random.Generator
    ) -> tuple[list[Image.Image], str]:
        if not self.lesion_guided_crops:
            return [self.local_geo(img) for _ in range(self.n_local)], "uniform"

        gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
        centers, note = lesion_guided.guided_crop_centers(gray, rng, self.n_local)
        # Window twice the local resolution keeps context around the border,
        # then the crop is resized down to the ViT-compatible local size.
        window = min(self.local_resolution * 2, min(img.size))
        views = []
        for center in centers:
            ys, xs = lesion_guided.crop_slices(
                (int(center[0]), int(center[1])), window, (img.height, img.width)
            )
            patch = img.crop((xs.start, ys.start, xs.stop, ys.stop))
            views.append(
                patch.resize((self.local_resolution, self.local_resolution))
            )
        return views, note

    def __getitem__(self, idx: int) -> dict[str, list[torch.Tensor] | int]:
        rng = np.random.default_rng((self.seed, idx))
        img = Image.open(self.paths[idx]).convert("RGB")
        globals_ = [self.global_geo(img) for _ in range(2)]
        locals_, note = self._local_views(img, rng)
        crops = [_finish_view(v, rng, self.cc_probs) for v in globals_ + locals_]
        return {"crops": crops, "guided": int(note == "border_ring")}


def dino_collate(batch: list[dict]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Stack crops per position: -> list of (B, C, H, W), plus guided flags."""
    n_crops = len(batch[0]["crops"])
    stacked = [
        torch.stack([item["crops"][i] for item in batch]) for i in range(n_crops)
    ]
    guided = torch.tensor([item["guided"] for item in batch], dtype=torch.float32)
    return stacked, guided
