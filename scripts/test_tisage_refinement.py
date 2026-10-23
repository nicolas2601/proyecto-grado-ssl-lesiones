"""TiSage-inspired SLIC contrast refinement for diffuse/saturated erythematous lesions."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.segmentation import slic

from dermssl.pipeline_segment_circle import (
    DermLIPCirclePipeline,
    select_bio_admissible_component,
)


def refine_saturated_mask_with_slic(
    img_rgb: np.ndarray,
    prob_map: np.ndarray,
    valid_fov: np.ndarray,
    n_segments: int = 300,
    compactness: float = 12.0,
) -> np.ndarray:
    """Refine diffuse/saturated probabilities using SLIC superpixel color contrast.

    When a U-Net is globally saturated (e.g. coverage > 75%), pixel-wise probabilities
    lack dynamic range. SLIC superpixels group local coherent color in CIE-Lab.
    By measuring the color/luminance contrast against the peripheral healthy skin,
    the pigmented core is cleanly decoupled from diffuse erythema.
    """
    h, w = img_rgb.shape[:2]

    # Check if raw thresholding is saturated (> 75% coverage)
    raw_binary = (prob_map >= 0.5) & valid_fov
    if raw_binary.mean() < 0.75:
        # Not saturated: standard pipeline is already accurate!
        return raw_binary

    # 1. Compute SLIC superpixels on valid field
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
    L_chan = lab[..., 0]

    # 2. Identify peripheral background skin reference
    # Outer 12% border margin within valid FOV
    border_mask = np.zeros((h, w), dtype=bool)
    m_h, m_w = int(h * 0.12), int(w * 0.12)
    border_mask[:m_h, :] = True
    border_mask[-m_h:, :] = True
    border_mask[:, :m_w] = True
    border_mask[:, -m_w:] = True
    bg_sample_mask = border_mask & valid_fov

    if bg_sample_mask.sum() > 50:
        bg_L = float(np.median(L_chan[bg_sample_mask]))
    else:
        bg_L = float(np.percentile(L_chan[valid_fov], 80))

    # 3. Score each superpixel by U-Net confidence + darkness/color contrast
    sp_scores = np.zeros(n_segs)
    cy, cx = h / 2.0, w / 2.0
    diag = np.hypot(h, w) / 2.0

    for s_idx in range(1, n_segs + 1):
        sp_mask = segments == s_idx
        if not sp_mask.any():
            continue

        sp_p = float(prob_map[sp_mask].mean())
        sp_L = float(L_chan[sp_mask].mean())

        # Darkness contrast: real lesions are darker (lower L*) than surrounding skin
        contrast = max(0.0, bg_L - sp_L)

        # Center prior
        y_c, x_c = ndimage.center_of_mass(sp_mask)
        d_norm = np.hypot(y_c - cy, x_c - cx) / (diag + 1e-7)
        spatial_prior = np.exp(-((d_norm / 0.40) ** 2))

        # TiSage-style fusion: confidence * contrast * spatial prior
        # contrast normalized by a typical 15 L* unit difference
        contrast_factor = 1.0 - np.exp(-max(0.0, contrast) / 10.0)
        score = sp_p * contrast_factor * spatial_prior
        sp_scores[s_idx - 1] = score

    # 4. Adaptive thresholding across superpixels
    pos_scores = sp_scores[sp_scores > 0]
    if len(pos_scores) > 0:
        # Cut at 45th percentile of active superpixel contrast
        th_cut = float(np.percentile(pos_scores, 45))
        active_segs = np.where(sp_scores >= th_cut)[0] + 1
        refined_binary = np.isin(segments, active_segs) & valid_fov
    else:
        refined_binary = raw_binary

    # 5. Clean up morphology and apply biological component selection
    refined_binary = select_bio_admissible_component(
        refined_binary, prob_map, img_rgb, center_prior=True
    )
    if refined_binary.any():
        refined_binary = ndimage.binary_fill_holes(refined_binary)

    return refined_binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/jovyan/work/dermlip_slic_v8/runs/dermlip-semi-supervised-v8-isic2018-2594-hair-010/selection/best_teacher_model.pt",
    )
    parser.add_argument(
        "--ham-dir",
        type=str,
        default="/home/jovyan/work/datos/ham10000/HAM10000_images_part_1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/tisage_refinement_comparison.png",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    pipeline = DermLIPCirclePipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        component="teacher",
    )

    ham_path = Path(args.ham_dir)
    # The 3 problematic images + the 2 successful ones
    test_ids = [
        "ISIC_0024320.jpg",
        "ISIC_0024328.jpg",
        "ISIC_0024362.jpg",
        "ISIC_0024368.jpg",
        "ISIC_0024371.jpg",
    ]

    fig, axes = plt.subplots(len(test_ids), 5, figsize=(18, 3.6 * len(test_ids)))

    for idx, fname in enumerate(test_ids):
        img_p = ham_path / fname
        if not img_p.exists():
            continue

        img_pil = Image.open(img_p).convert("RGB")
        img_rgb = np.array(img_pil)
        w, h = img_pil.size

        # 1. Standard pipeline result (raw U-Net threshold 0.5)
        raw_mask_res = pipeline.segment_image(img_rgb)

        # 2. Get U-Net probability map for TiSage refinement
        img_224 = np.array(
            img_pil.resize((224, 224), Image.Resampling.BILINEAR)
        )
        from dermssl.segmentation_methods.common.hair import remove_hair
        clean_224, _ = remove_hair(img_224, shaped=False)
        import torch
        t = torch.from_numpy(np.transpose(clean_224.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(args.device)
        with torch.no_grad():
            prob_224 = torch.sigmoid(pipeline.model(t)).squeeze().cpu().numpy()
        prob_orig = np.array(
            Image.fromarray((prob_224 * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        from dermssl.field_roi import field_validity_mask
        valid_fov, _ = field_validity_mask(img_rgb)

        # 3. Apply TiSage SLIC contrast refinement
        tisage_mask = refine_saturated_mask_with_slic(img_rgb, prob_orig, valid_fov)

        # Create updated PipelineMaskResult
        healthy_skin = valid_fov & (~tisage_mask)
        tisage_res = raw_mask_res.__class__(
            lesion_mask=tisage_mask,
            valid_fov_mask=valid_fov,
            healthy_skin_mask=healthy_skin,
            lesion_coverage=float(tisage_mask.mean()),
            healthy_skin_coverage=float(healthy_skin.mean()),
            qa_passed=(0.005 <= tisage_mask.mean() <= 0.90 and healthy_skin.mean() >= 0.05),
        )

        # 4. CIRCLe shift with the TiSage refined mask
        shifted_p1, _ = pipeline.apply_circle_shift(img_rgb, tisage_res, shift_buckets=1)
        shifted_p2, _ = pipeline.apply_circle_shift(img_rgb, tisage_res, shift_buckets=2)

        ax = axes[idx]

        # Col 0: Original
        ax[0].imshow(img_rgb)
        ax[0].set_title(f"{fname}", fontsize=11, fontweight="bold")

        # Col 1: Standard / Raw Mask
        ax[1].imshow(img_rgb)
        ax[1].contour(raw_mask_res.lesion_mask, levels=[0.5], colors="red", linewidths=1.5)
        ax[1].set_title(
            f"Sin TiSage ({raw_mask_res.lesion_coverage:.1%})",
            fontsize=11,
            color="red" if raw_mask_res.lesion_coverage > 0.9 else "green",
            fontweight="bold",
        )

        # Col 2: TiSage SLIC Refined Mask
        ax[2].imshow(img_rgb)
        ax[2].contour(tisage_mask, levels=[0.5], colors="lime", linewidths=1.5)
        skin_vis = img_rgb.copy().astype(float)
        skin_vis[healthy_skin, 0] = skin_vis[healthy_skin, 0] * 0.7
        skin_vis[healthy_skin, 2] = np.clip(skin_vis[healthy_skin, 2] * 1.3, 0, 255)
        ax[2].imshow(skin_vis.astype(np.uint8))
        ax[2].set_title(
            f"TiSage SLIC ({tisage_res.lesion_coverage:.1%}, Piel {tisage_res.healthy_skin_coverage:.1%})",
            fontsize=11,
            color="darkgreen",
            fontweight="bold",
        )

        # Col 3: CIRCLe Shift +1
        ax[3].imshow(shifted_p1)
        ax[3].set_title("CIRCLe Shift +1", fontsize=11, fontweight="bold", color="saddlebrown")

        # Col 4: CIRCLe Shift +2
        ax[4].imshow(shifted_p2)
        ax[4].set_title("CIRCLe Shift +2 (LatAm)", fontsize=11, fontweight="bold", color="maroon")

        for a in ax:
            a.axis("off")

    plt.suptitle(
        "TiSage Superpixel Contrast Refinement: Rescuing Diffuse Erythematous Lesions for CIRCLe",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Generated {args.output} successfully.")


if __name__ == "__main__":
    main()
