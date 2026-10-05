"""The input contract for models trained on unit-range images.

This module exists because of a bug that produced no error at all. The
segmentation U-Net is trained with ``torchvision.to_tensor`` (torch_unet.py),
which yields float in [0, 1]. ``infer_mask_with_confidence`` therefore expects
[0, 1] and does NOT divide by 255 -- and a caller that hands it a raw uint8
array in [0, 255] gets saturated activations and masks that still look like
masks. Two of the three callers divided; one did not, and nothing anywhere
noticed.

Validating at the callee rather than fixing the one caller is deliberate: the
next caller will make the same assumption, and a contract that is only
documented is a contract that gets broken silently.

Deliberately torch-free so it can be imported and tested on a machine with no
GPU stack (see tests/test_torch_unet_static.py, which forbids torch imports in
non ``torch_*`` modules).
"""

from __future__ import annotations

import numpy as np


def assert_unit_float_image(img: np.ndarray, *, name: str = "img") -> None:
    """Raise unless ``img`` is floating point within [0, 1].

    The error names the fix, because the symptom of getting this wrong is not
    an exception but a plausible-looking wrong answer.
    """
    array = np.asarray(img)

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"{name} has shape {array.shape}, expected (H, W, 3). Grayscale (H, W) and "
            "RGBA (H, W, 4) both reach the network as the wrong number of channels; "
            "convert with PIL's .convert('RGB') first."
        )

    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"{name} has dtype {array.dtype}, expected floating point in [0, 1]. "
            "This model was trained on to_tensor() output; pass "
            "`np.asarray(image, dtype=np.float32) / 255.0`, not the raw uint8 array. "
            "Feeding it 0..255 saturates the network and returns confident nonsense."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} has shape {array.shape} and no pixels. An empty array passes every "
            "range check vacuously, so it would reach the network as a silent no-op."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or infinity. A range check alone would not catch it: "
            "every comparison against NaN is False, so it slips past min/max bounds and "
            "propagates silently through the network."
        )

    low = float(array.min())
    high = float(array.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f"{name} spans [{low:.4g}, {high:.4g}], outside the expected range [0, 1]. "
            "If the values look like 0..255, divide by 255 before calling."
        )
