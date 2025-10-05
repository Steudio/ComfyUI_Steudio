# _utils_.py

import torch
import torch.nn.functional as F

# Copy to py
# import os
# import sys
# sys.path.append(os.path.dirname(__file__))
# from _utils_ import _soft_despeckle, _binarize_and_despeckle

# ---------- Utility cleanup helpers ----------

def _soft_despeckle(m, tau=1/255, min_neighbors=2, passes=1):
    """
    Accepts (H,W), (1,H,W), or (N,1,H,W). Returns same rank/shape.
    Multiplies by a binary support map that removes isolated 1‑pixels and crumbs.
    """
    orig_ndim = m.ndim
    if orig_ndim == 2:
        # (H,W) -> (1,1,H,W)
        m4 = m.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        # (1,H,W) or (C,H,W) -> (1,1,H,W) using first channel if needed
        if m.shape[0] != 1:
            m4 = m[:1].unsqueeze(0)
        else:
            m4 = m.unsqueeze(0)
    elif orig_ndim == 4:
        # (N,C,H,W) -> ensure C==1
        m4 = m if m.shape[1] == 1 else m[:, :1, ...]
    else:
        raise ValueError(f"_soft_despeckle: unsupported ndim={orig_ndim}")

    b = (m4 > tau).float()
    k = torch.ones((1,1,3,3), device=m4.device, dtype=m4.dtype)
    for _ in range(passes):
        cnt = F.conv2d(b, k, padding=1)
        b = (cnt >= float(min_neighbors)).float()
    out4 = (m4 * b).clamp(0, 1)

    if orig_ndim == 2:
        return out4[0, 0]
    elif orig_ndim == 3:
        return out4[0]
    else:  # orig_ndim == 4
        return out4

def _binarize_and_despeckle(m, tau=1/255, min_neighbors=2):
    """Normalize, threshold, require neighbor support; returns (H,W) float binary."""
    if m.max() > 1.0:
        m = (m / 255.0).clamp(0, 1)
    b = (m > tau).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    k = torch.ones((1,1,3,3), device=b.device)
    count = F.conv2d(b, k, padding=1)
    support = (count >= float(min_neighbors)).float()
    b = b * support
    return b[0,0]

# Overlap ratios by description
OVERLAP_DICT = {
    "None": 0,
    "1/64 Tile": 0.015625,
    "1/32 Tile": 0.03125,
    "1/16 Tile": 0.0625,
    "1/8 Tile": 0.125,
    "1/4 Tile": 0.25,
    "1/2 Tile": 0.5,
}

# Tile ordering methods
TILE_ORDER_DICT = {
    "linear": 0,
    "spiral": 1
}

# Available scaling algorithms
SCALING_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos"
]

# Minimum scale threshold
MIN_SCALE_FACTOR_THRESHOLD = 1.0

# Ratio Presets
RATIO_PRESETS = {
    "auto": (1, 1),
    "1:1 ◻": (1, 1),
    "5:4 ▭": (5, 4),
    "4:3 ▭": (4, 3),
    "3:2 ▭": (3, 2),
    "16:9 ▭": (16, 9),
    "2:1 ▭": (2, 1),
    "21:9 ▭": (21, 9),
    "32:9 ▭": (32, 9),
    "": (1, 1),
    "4:5 ▯": (4, 5),
    "3:4 ▯": (3, 4),
    "2:3 ▯": (2, 3),
    "9:16 ▯": (9, 16),
    "1:2 ▯": (1, 2),
    "9:21 ▯": (9, 21),
    "9:32 ▯": (9, 32),
}