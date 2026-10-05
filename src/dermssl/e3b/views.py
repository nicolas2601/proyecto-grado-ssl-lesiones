"""The two images F4 gives a frozen backbone (docs/33 E3), without the other seven conditions.

``e3_render.render`` builds nine conditions per image, which for eight backbones would
repeat the shifts, the ring and the 448 crop 8 times for nothing. F4 needs ``full`` (the
bank view) and ``crop_view`` (the margin box cut from that view); this module builds just
those with the SAME helpers, and ``tests/test_e3b_views.py`` pins the result to
``e3_render.render`` so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from dermssl import c1_extraction as ex
from dermssl import e3_render as render

FALLBACK_COLUMN = render.CONDITIONS.index("crop_view")  # its column in E3's ``fallbacks`` array


def render_pair(image: np.ndarray, mask: np.ndarray, resize_crop) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(full, crop_view, fell_back)``; a crop that cannot be built is the full view itself."""
    full = resize_crop(image, render.BASE)
    crop = render._view_crop(full, ex._baseline_geometry_mask(np.asarray(mask, dtype=bool)))
    return full, (full if crop is None else crop), crop is None
