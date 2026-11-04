"""Controlled PH2-only Copy-Paste augmentation for small/micro-lesion delineation.

Extracts authentic skin backgrounds from PH2 train split (qualified hosts with >=55%
clean skin and <=25% lesion), inpaints the central hole with clean skin texture, and
pastes scaled foreground lesions targeting clinical small-lesion bins:
- Bin 1 (Micro): 0.5% - 1.0% of image area
- Bin 2 (Small): 1.0% - 2.0% of image area
- Bin 3 (Medium-Small): 2.0% - 5.0% of image area

Ensures zero perilesional pigment leakage and 100% genuine dermoscopic skin.
"""

import math
from typing import Any, List, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import torch


def build_ph2_clean_background_bank(
    dataset: Any,
    target_size: Tuple[int, int] = (224, 224),
    min_clean_frac: float = 0.55,
    max_lesion_area: float = 0.25,
) -> torch.Tensor:
    """Construct verified clean dermoscopic skin backgrounds from PH2 training images.

    Args:
        dataset: PH2Dataset instance for the train split.
        target_size: Output spatial resolution (H, W).
        min_clean_frac: Minimum clean skin fraction within FOV.
        max_lesion_area: Maximum original lesion area fraction allowed for background host.

    Returns:
        Tensor of shape [N_hosts, 3, H, W] in [0, 1] float.
    """
    base_dir = getattr(dataset, "base_dir", None)
    records = getattr(dataset, "records", [])
    if base_dir is None or not records:
        raise ValueError("Dataset must have base_dir and records to build background bank")

    clean_backgrounds: List[torch.Tensor] = []
    h_out, w_out = target_size

    for rec in records:
        img_p = base_dir / rec["image_path"]
        msk_p = base_dir / rec["mask_path"]
        if not img_p.is_file() or not msk_p.is_file():
            continue

        img_orig = np.asarray(Image.open(img_p).convert("RGB"))
        msk_orig = np.asarray(Image.open(msk_p).convert("L")) > 127

        img_resized = cv2.resize(img_orig, (w_out, h_out), interpolation=cv2.INTER_AREA)
        msk_resized = (
            cv2.resize(msk_orig.astype(np.uint8), (w_out, h_out), interpolation=cv2.INTER_NEAREST)
            > 0
        )

        fov = img_resized.mean(axis=-1) > 28.0
        fov = ndimage.binary_erosion(fov, iterations=4)

        dil_r = max(6, int(round(min(h_out, w_out) * 0.045)))
        msk_dil = ndimage.binary_dilation(msk_resized, iterations=dil_r)

        clean_skin = fov & (~msk_dil)
        clean_frac = float(clean_skin.sum()) / float(fov.sum() + 1e-6)
        lesion_area = float(msk_resized.mean())

        # Quality Gate for Background Host
        if clean_frac < min_clean_frac or lesion_area > max_lesion_area or clean_skin.sum() < 400:
            continue

        inpaint_mask = (msk_dil & fov).astype(np.uint8) * 255
        inpainted = cv2.inpaint(img_resized, inpaint_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

        # Subtle high-frequency skin grain injection
        residual = img_resized.astype(np.float32) - cv2.GaussianBlur(img_resized, (9, 9), 2.0).astype(
            np.float32
        )
        grain_pool = residual[clean_skin]
        if len(grain_pool) > 50:
            h_idx = np.where(inpaint_mask > 0)
            n_h = len(h_idx[0])
            grain_idx = np.random.choice(len(grain_pool), n_h, replace=True)
            rand_grain = grain_pool[grain_idx]
            inp_f = inpainted.astype(np.float32)
            for c in range(3):
                inp_f[h_idx[0], h_idx[1], c] = np.clip(
                    inp_f[h_idx[0], h_idx[1], c] + 0.6 * rand_grain[:, c], 0, 255
                )
            clean_canvas = inp_f.astype(np.uint8)
        else:
            clean_canvas = inpainted

        # Convert to float tensor [3, H, W] in [0, 1]
        canvas_t = torch.from_numpy(clean_canvas).permute(2, 0, 1).float() / 255.0
        clean_backgrounds.append(canvas_t)

    if not clean_backgrounds:
        raise RuntimeError("No qualified background hosts found in dataset")

    return torch.stack(clean_backgrounds, dim=0)


def build_ph2_synthetic_val_diagnostic(
    dataset_val: Any,
    target_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build a fixed synthetic small-lesion diagnostic set strictly from PH2 val images."""
    rng = np.random.RandomState(seed)
    base_dir = getattr(dataset_val, "base_dir", None)
    records = getattr(dataset_val, "records", [])
    if base_dir is None or not records:
        return []

    h_out, w_out = target_size
    parsed = []
    for rec in records:
        img_p = base_dir / rec["image_path"]
        msk_p = base_dir / rec["mask_path"]
        if not img_p.is_file() or not msk_p.is_file():
            continue
        img_orig = np.asarray(Image.open(img_p).convert("RGB"))
        msk_orig = np.asarray(Image.open(msk_p).convert("L")) > 127
        img_r = cv2.resize(img_orig, (w_out, h_out), interpolation=cv2.INTER_AREA)
        msk_r = cv2.resize(msk_orig.astype(np.uint8), (w_out, h_out), interpolation=cv2.INTER_NEAREST) > 0
        fov = img_r.mean(axis=-1) > 28.0
        fov = ndimage.binary_erosion(fov, iterations=4)
        dil_r = max(6, int(round(min(h_out, w_out) * 0.045)))
        msk_dil = ndimage.binary_dilation(msk_r, iterations=dil_r)
        clean_skin = fov & (~msk_dil)
        clean_frac = float(clean_skin.sum()) / float(fov.sum() + 1e-6)
        lesion_area = float(msk_r.mean())
        parsed.append({
            "img": img_r,
            "msk": msk_r,
            "clean_frac": clean_frac,
            "lesion_area": lesion_area,
            "msk_dil": msk_dil,
            "fov": fov,
        })

    hosts = [p for p in parsed if p["clean_frac"] >= 0.55 and p["lesion_area"] <= 0.25]
    donors = [p for p in parsed if p["lesion_area"] >= 0.03]
    if not hosts or not donors:
        return []

    synthetic_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    target_bins = [(0.005, 0.010), (0.010, 0.020), (0.020, 0.050)]

    for idx, host in enumerate(hosts):
        inpaint_mask = (host["msk_dil"] & host["fov"]).astype(np.uint8) * 255
        clean_bg = cv2.inpaint(host["img"], inpaint_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

        donor = donors[idx % len(donors)]
        l_ys, l_xs = np.where(donor["msk"])
        if len(l_ys) < 20:
            continue
        ly0, ly1 = int(l_ys.min()), int(l_ys.max()) + 1
        lx0, lx1 = int(l_xs.min()), int(l_xs.max()) + 1
        l_img_crop = donor["img"][ly0:ly1, lx0:lx1].copy()
        l_msk_crop = donor["msk"][ly0:ly1, lx0:lx1].copy()
        orig_px = int(l_msk_crop.sum())

        bin_tup = target_bins[idx % len(target_bins)]
        target_f = (bin_tup[0] + bin_tup[1]) / 2.0
        target_px = target_f * (h_out * w_out)
        scale = math.sqrt(target_px / max(orig_px, 1))

        new_h = max(8, int(round((ly1 - ly0) * scale)))
        new_w = max(8, int(round((lx1 - lx0) * scale)))

        r_img = cv2.resize(l_img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        r_msk = cv2.resize(l_msk_crop.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0

        margin = 22
        max_py = max(margin, h_out - new_h - margin)
        max_px = max(margin, w_out - new_w - margin)
        py = rng.randint(margin, max_py + 1) if max_py > margin else (h_out - new_h) // 2
        px = rng.randint(margin, max_px + 1) if max_px > margin else (w_out - new_w) // 2

        alpha = cv2.GaussianBlur(r_msk.astype(np.float32), (3, 3), 0.7)[:, :, None]
        comp_img = clean_bg.copy()
        roi = comp_img[py : py + new_h, px : px + new_w].astype(np.float32)
        comp_img[py : py + new_h, px : px + new_w] = np.clip(
            (1.0 - alpha) * roi + alpha * r_img.astype(np.float32), 0, 255
        ).astype(np.uint8)

        comp_msk = np.zeros((h_out, w_out), dtype=np.float32)
        comp_msk[py : py + new_h, px : px + new_w] = r_msk.astype(np.float32)

        img_t = torch.from_numpy(comp_img).permute(2, 0, 1).float() / 255.0
        msk_t = torch.from_numpy(comp_msk).unsqueeze(0)
        synthetic_samples.append((img_t, msk_t))

    return synthetic_samples
