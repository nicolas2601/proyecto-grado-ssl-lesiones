"""Integration pipeline: DermLIP Bio-Refined Segmentation + Mask-Bounded CIRCLe.

Couples the refined semantic segmenter (hair removal, adaptive field FOV,
bio-admissible component selection, hole filling) with the CIE-Lab CIRCLe tone
transformer, strictly restricting skin tone adaptation to valid healthy perilesional
skin while preserving lesion morphology bit-by-bit and leaving dermatoscope vignettes intact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from dermssl.circle import (
    FST_SHIFT_L_STEP,
    MAX_FST_SHIFT,
    clamp_fst_shift,
    lab_to_srgb,
)
from dermssl.field_roi import field_validity_mask
from dermssl.fitzpatrick import srgb_to_lab
from dermssl.segmentation_methods.common.hair import remove_hair
from dermssl.segmentation_methods.dermlip_slic.trainer import DermLIPSegmentationUNet
from skimage.color import rgb2lab
from skimage.segmentation import slic

LOGGER = logging.getLogger(__name__)

MIN_HEALTHY_SKIN_FRACTION = 0.05
MIN_LESION_AREA_FRACTION = 0.005
MAX_LESION_AREA_FRACTION = 0.90


@dataclass(frozen=True)
class PipelineMaskResult:
    """Diagnostic outputs for a segmented frame."""

    lesion_mask: np.ndarray  # bool (H, W), True = lesion
    valid_fov_mask: np.ndarray  # bool (H, W), True = optical field
    healthy_skin_mask: np.ndarray  # bool (H, W), True = valid skin excluding lesion
    lesion_coverage: float
    healthy_skin_coverage: float
    qa_passed: bool
    fallback_reason: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def select_bio_admissible_component(
    pred_binary: np.ndarray,
    prob_map: np.ndarray,
    img_rgb: np.ndarray,
    center_prior: bool = True,
) -> np.ndarray:
    """Select the most biologically plausible lesion component.

    Penalizes synthetic calibration stickers (blue, cyan, green, neon yellow/orange),
    dampens raw pixel area via square-root, and rewards central high confidence.
    """
    labeled_mask, num_features = ndimage.label(pred_binary)
    if num_features <= 1:
        return pred_binary

    if not center_prior:
        sizes = ndimage.sum(pred_binary, labeled_mask, range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        return labeled_mask == largest_label

    h_p, w_p = pred_binary.shape
    cy, cx = h_p / 2.0, w_p / 2.0
    diag = np.sqrt(h_p**2 + w_p**2) / 2.0
    sizes = ndimage.sum(pred_binary, labeled_mask, range(1, num_features + 1))
    coms = ndimage.center_of_mass(pred_binary, labeled_mask, range(1, num_features + 1))

    hsv = None
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        import cv2

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    scores = []
    for i in range(num_features):
        comp_mask_i = labeled_mask == (i + 1)
        y_i, x_i = coms[i]
        d_norm = np.sqrt((y_i - cy) ** 2 + (x_i - cx) ** 2) / (diag + 1e-7)

        color_penalty = 1.0
        if hsv is not None:
            comp_rgb = img_rgb[comp_mask_i]
            comp_hsv = hsv[comp_mask_i]
            mean_r, mean_g, mean_b = comp_rgb.mean(axis=0)
            mean_s = comp_hsv[:, 1].mean()
            mean_v = comp_hsv[:, 2].mean()

            # Synthetic blue/cyan/teal/green stickers or ruler ink
            if mean_b > mean_r or mean_g > (mean_r * 1.05):
                color_penalty *= 0.0001
            # Neon saturated stickers (bright orange, yellow, magenta)
            elif mean_v > 190 and mean_s > 150:
                color_penalty *= 0.001

        mean_p = float(prob_map[comp_mask_i].mean())
        size_factor = np.sqrt(float(sizes[i]))
        center_factor = np.exp(-((d_norm / 0.35) ** 2))
        prob_factor = mean_p**2

        score = size_factor * center_factor * prob_factor * color_penalty
        scores.append(score)

    best_label = int(np.argmax(scores)) + 1
    return labeled_mask == best_label


def refine_diffuse_mask_with_slic(
    img_rgb: np.ndarray,
    prob_map: np.ndarray,
    valid_fov: np.ndarray,
    n_segments: int = 300,
    compactness: float = 12.0,
    saturation_threshold: float = 0.75,
) -> np.ndarray:
    """Refine diffuse/saturated probabilities using TiSage-inspired SLIC superpixel contrast.

    When a U-Net is globally saturated (> 75% coverage or low dynamic range),
    pixel-wise thresholding floods the field. SLIC superpixels group local coherent
    color in CIE-Lab. By measuring luminance and color contrast against the peripheral
    healthy skin, the pigmented core is decoupled from diffuse surrounding erythema.
    """
    h, w = img_rgb.shape[:2]
    raw_binary = (prob_map >= 0.5) & valid_fov
    if raw_binary.mean() < saturation_threshold:
        return raw_binary

    segments = slic(
        img_rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=valid_fov,
    )
    n_segs = int(segments.max())
    if n_segs <= 1:
        return raw_binary

    lab = rgb2lab(img_rgb)
    l_chan = lab[..., 0]

    border_mask = np.zeros((h, w), dtype=bool)
    m_h, m_w = int(h * 0.12), int(w * 0.12)
    border_mask[:m_h, :] = True
    border_mask[-m_h:, :] = True
    border_mask[:, :m_w] = True
    border_mask[:, -m_w:] = True
    bg_sample_mask = border_mask & valid_fov

    if bg_sample_mask.sum() > 50:
        bg_l = float(np.median(l_chan[bg_sample_mask]))
    else:
        bg_l = float(np.percentile(l_chan[valid_fov], 80))

    sp_scores = np.zeros(n_segs)
    cy, cx = h / 2.0, w / 2.0
    diag = np.hypot(h, w) / 2.0

    for s_idx in range(1, n_segs + 1):
        sp_mask = segments == s_idx
        if not sp_mask.any():
            continue

        sp_p = float(prob_map[sp_mask].mean())
        sp_l = float(l_chan[sp_mask].mean())

        contrast = max(0.0, bg_l - sp_l)
        y_c, x_c = ndimage.center_of_mass(sp_mask)
        d_norm = np.hypot(y_c - cy, x_c - cx) / (diag + 1e-7)
        spatial_prior = np.exp(-((d_norm / 0.40) ** 2))

        contrast_factor = 1.0 - np.exp(-max(0.0, contrast) / 10.0)
        score = sp_p * contrast_factor * spatial_prior
        sp_scores[s_idx - 1] = score

    pos_scores = sp_scores[sp_scores > 0]
    if len(pos_scores) > 0:
        th_cut = float(np.percentile(pos_scores, 45))
        active_segs = np.where(sp_scores >= th_cut)[0] + 1
        refined_binary = np.isin(segments, active_segs) & valid_fov
    else:
        refined_binary = raw_binary

    refined_binary = select_bio_admissible_component(
        refined_binary, prob_map, img_rgb, center_prior=True
    )
    if refined_binary.any():
        refined_binary = ndimage.binary_fill_holes(refined_binary)

    return refined_binary


class DermLIPCirclePipeline:
    """End-to-end pipeline coupling DermLIP-SLIC segmentation and bounded CIRCLe."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        component: str = "teacher",
        threshold: float = 0.5,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self.model: Optional[torch.nn.Module] = None

        if checkpoint_path is not None:
            self.load_weights(checkpoint_path, component=component)

    def load_weights(
        self, checkpoint_path: Union[str, Path], component: str = "teacher"
    ) -> None:
        """Load DermLIP weights from disk."""
        ckpt_p = Path(checkpoint_path).resolve()
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

        model = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=16)
        ckpt = torch.load(ckpt_p, map_location=self.device, weights_only=False)

        state_key = f"{component}_state_dict"
        state_dict = ckpt.get(state_key)
        if state_dict is None:
            raise ValueError(f"Component '{component}' not found in checkpoint {ckpt_p}")

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        apply_hair_removal: bool = True,
        apply_vignette_mask: bool = True,
    ) -> PipelineMaskResult:
        """Run robust segmentation on an RGB image, producing lesion and healthy skin masks."""
        h, w = image_rgb.shape[:2]

        if self.model is None:
            raise RuntimeError("Model weights must be loaded before segmenting")

        # 1. Hair removal pre-processing
        prep_img = image_rgb
        if apply_hair_removal:
            img_224 = np.array(
                Image.fromarray(image_rgb).resize((224, 224), Image.Resampling.BILINEAR)
            )
            clean_224, _ = remove_hair(img_224, shaped=False)
            norm_224 = clean_224.astype(np.float32) / 255.0
        else:
            img_224 = np.array(
                Image.fromarray(image_rgb).resize((224, 224), Image.Resampling.BILINEAR)
            )
            norm_224 = img_224.astype(np.float32) / 255.0

        # 2. U-Net inference
        tensor_in = (
            torch.from_numpy(np.transpose(norm_224, (2, 0, 1)))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.model(tensor_in)
            prob_224 = torch.sigmoid(logits).squeeze().cpu().numpy()

        prob_orig = (
            np.array(
                Image.fromarray((prob_224 * 255).astype(np.uint8)).resize(
                    (w, h), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )

        pred_binary = prob_orig >= self.threshold

        # 3. Field of view detection (dermatoscope vignette vs clinical full frame)
        if apply_vignette_mask:
            valid_fov, _ = field_validity_mask(image_rgb)
        else:
            valid_fov = np.ones((h, w), dtype=bool)

        pred_binary = pred_binary & valid_fov

        # 4. TiSage SLIC superpixel contrast refinement for diffuse/saturated lesions
        pred_binary = refine_diffuse_mask_with_slic(
            image_rgb, prob_orig, valid_fov
        )

        # 5. Biological connected component selection
        pred_binary = select_bio_admissible_component(
            pred_binary, prob_orig, image_rgb, center_prior=True
        )

        # 6. Fill specular/crust holes
        if pred_binary.any():
            pred_binary = ndimage.binary_fill_holes(pred_binary)

        # 6. Quality assurance checks
        lesion_coverage = float(pred_binary.mean())
        healthy_skin_mask = valid_fov & (~pred_binary)
        healthy_skin_coverage = float(healthy_skin_mask.mean())

        qa_passed = True
        fallback_reason = None

        if lesion_coverage < MIN_LESION_AREA_FRACTION:
            qa_passed = False
            fallback_reason = f"lesion_too_small:{lesion_coverage:.4f}"
        elif lesion_coverage > MAX_LESION_AREA_FRACTION:
            qa_passed = False
            fallback_reason = f"lesion_too_large:{lesion_coverage:.4f}"
        elif healthy_skin_coverage < MIN_HEALTHY_SKIN_FRACTION:
            qa_passed = False
            fallback_reason = f"insufficient_healthy_skin:{healthy_skin_coverage:.4f}"

        return PipelineMaskResult(
            lesion_mask=pred_binary,
            valid_fov_mask=valid_fov,
            healthy_skin_mask=healthy_skin_mask,
            lesion_coverage=lesion_coverage,
            healthy_skin_coverage=healthy_skin_coverage,
            qa_passed=qa_passed,
            fallback_reason=fallback_reason,
        )

    def apply_circle_shift(
        self,
        image_rgb: np.ndarray,
        mask_result: PipelineMaskResult,
        shift_buckets: int,
        *,
        max_shift: int = MAX_FST_SHIFT,
    ) -> Tuple[np.ndarray, bool]:
        """Apply CIRCLe L* shift strictly to healthy perilesional skin.

        Returns (transformed_rgb_uint8, was_transformed_bool).
        If mask_result fails QA, the image is returned unmodified (fallback identity).
        """
        if not mask_result.qa_passed or shift_buckets == 0:
            return image_rgb.copy(), False

        shift = clamp_fst_shift(shift_buckets, max_shift=max_shift)
        img_float = image_rgb.astype(float) / 255.0
        lab = srgb_to_lab(img_float)

        # Shift only valid healthy skin pixels!
        skin_mask = mask_result.healthy_skin_mask
        shifted_L = np.clip(lab[..., 0] - shift * FST_SHIFT_L_STEP, 0.0, 100.0)

        shifted_lab = lab.copy()
        shifted_lab[..., 0] = np.where(skin_mask, shifted_L, lab[..., 0])

        out_srgb = lab_to_srgb(shifted_lab)

        # Lesion and non-FOV vignette are preserved bit-for-bit
        out_float = np.where(skin_mask[..., None], out_srgb, img_float)
        out_uint8 = np.clip(np.round(out_float * 255.0), 0, 255).astype(np.uint8)

        return out_uint8, True
