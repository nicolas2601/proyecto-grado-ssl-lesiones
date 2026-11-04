"""Rescue mechanisms for extreme segmentation edge cases.

Includes:
1. Micro-lesion coarse-to-fine zoom and retention (preventing small focal lesions from being discarded).
2. Multi-space color distance refinement (incorporating HSV Saturation and Lab a* for amelanotic/low-contrast lesions).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def rescue_micro_lesion_candidate(
    prob_map: np.ndarray,
    valid_fov: np.ndarray,
    min_prob_peak: float = 0.40,
    area_fraction_threshold: float = 0.02,
    min_pixels: int = 15,
) -> np.ndarray | None:
    """Detect and rescue isolated focal micro-lesions (< 2% of frame).

    When a lesion occupies < 2% of the field of view, standard connected component
    filters with area penalties or global Otsu thresholds extinguish it.
    This detector isolates compact, high-probability focal islands before noise culling.

    Returns:
        binary mask (H, W) if a valid focal micro-lesion is confirmed, else None.
    """
    h, w = prob_map.shape[:2]
    total_area = float(h * w)
    max_area_pixels = int(total_area * area_fraction_threshold)

    # Candidate focal spots with decent peak confidence
    focal_candidates = (prob_map >= min_prob_peak) & valid_fov
    if not focal_candidates.any():
        return None

    labeled, num_features = ndimage.label(focal_candidates)
    if num_features == 0:
        return None

    sizes = ndimage.sum(focal_candidates, labeled, range(1, num_features + 1))
    coms = ndimage.center_of_mass(focal_candidates, labeled, range(1, num_features + 1))

    cy, cx = h / 2.0, w / 2.0
    diag = np.hypot(h, w)

    best_mask = None
    best_score = -1.0

    for i in range(num_features):
        sz = sizes[i]
        # Must be strictly within micro-lesion bounds
        if min_pixels <= sz <= max_area_pixels:
            y_i, x_i = coms[i]
            d_norm = np.hypot(y_i - cy, x_i - cx) / (diag + 1e-7)

            comp_mask = labeled == (i + 1)
            mean_p = float(prob_map[comp_mask].mean())
            peak_p = float(prob_map[comp_mask].max())

            # Compactness / circularity check: micro-lesions are typically compact circular spots
            perimeter = float(ndimage.binary_dilation(comp_mask).sum() - comp_mask.sum())
            circularity = 4.0 * np.pi * sz / (max(perimeter, 1.0) ** 2)

            # Center-prior + high confidence + compactness
            center_factor = np.exp(-((d_norm / 0.40) ** 2))
            score = mean_p * peak_p * center_factor * min(circularity, 1.5)

            if score > best_score and mean_p >= 0.45:
                best_score = score
                best_mask = comp_mask

    return best_mask


def compute_multi_space_color_contrast(
    img_rgb: np.ndarray,
    foreground_mask: np.ndarray,
    background_mask: np.ndarray,
) -> float:
    """Compute combined CIE-Lab + HSV color contrast between foreground and background.

    For amelanotic/hypopigmented lesions, delta-L (luminance) is near zero, but
    erythematous vascularization elevates delta-a* (Lab green-red opponent) and
    HSV Saturation (S).

    Returns:
        Perceptual contrast metric >= 0.0.
    """
    import cv2  # noqa: PLC0415
    from skimage.color import rgb2lab  # noqa: PLC0415

    if not foreground_mask.any() or not background_mask.any():
        return 0.0

    lab = rgb2lab(img_rgb)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    fg_lab = lab[foreground_mask]
    bg_lab = lab[background_mask]

    delta_L = abs(float(fg_lab[:, 0].mean() - bg_lab[:, 0].mean()))
    delta_a = abs(float(fg_lab[:, 1].mean() - bg_lab[:, 1].mean()))  # Erythema / red axis
    delta_b = abs(float(fg_lab[:, 2].mean() - bg_lab[:, 2].mean()))  # Yellow / melanin axis

    delta_E_lab = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

    fg_hsv = hsv[foreground_mask]
    bg_hsv = hsv[background_mask]
    delta_S = abs(float(fg_hsv[:, 1].mean() - bg_hsv[:, 1].mean())) / 255.0

    # Multi-space weighted score
    return float(delta_E_lab + (delta_S * 20.0))
