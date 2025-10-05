# mask_to_geometry.py

import math
import numpy as np
import torch
from PIL import Image, ImageDraw
import os, sys
sys.path.append(os.path.dirname(__file__))
from _config_utils_ import RATIO_PRESETS

class mask_to_geometry:
    MASK_VAL = 255

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "shape": (["rectangle", "square", "circle", "ellipse", "ratio"],),
                "mode": (["solid", "line"],),
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 100}),
                "ratio": (list(RATIO_PRESETS.keys()),),
            },
            "optional": {
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "ui")
    FUNCTION = "convert"
    CATEGORY = "Steudio/Utils"
    OUTPUT_IS_LIST = (False, False)
    OUTPUT_NODE = True

    @staticmethod
    def _round_up_64(val):
        return int(math.ceil(val / 64) * 64)

    def draw_shape_np(self, shape_type, canvas_size, coords, fill, thickness):
        img = Image.new("L", canvas_size, 0)
        draw = ImageDraw.Draw(img)
        if shape_type in ["rectangle", "square", "ratio"]:
            draw.rectangle(
                coords,
                fill=fill if thickness == -1 else None,
                outline=fill,
                width=thickness
            )
        elif shape_type in ["ellipse", "circle"]:
            draw.ellipse(
                coords,
                fill=fill if thickness == -1 else None,
                outline=fill,
                width=thickness
            )
        return np.array(img, dtype=np.uint8)

    def _center_and_clamp(self, cx, cy, ww, hh, w_img, h_img):
        """Center box at cx,cy but keep fully inside image bounds."""
        x1 = cx - ww // 2
        y1 = cy - hh // 2
        x2 = x1 + ww - 1
        y2 = y1 + hh - 1

        # shift horizontally if overflow
        if x1 < 0:
            x2 += -x1
            x1 = 0
        elif x2 >= w_img:
            shift = x2 - (w_img - 1)
            x1 -= shift
            x2 = w_img - 1

        # shift vertically if overflow
        if y1 < 0:
            y2 += -y1
            y1 = 0
        elif y2 >= h_img:
            shift = y2 - (h_img - 1)
            y1 -= shift
            y2 = h_img - 1

        return max(0, x1), max(0, y1), min(w_img - 1, x2), min(h_img - 1, y2)

    def convert(self, mask, shape, mode, line_thickness, ratio, mask2=None):
        # --- Helper to normalize mask into uint8 2D array
        def to_uint8(m):
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            else:
                m = np.array(m)
            if m.ndim == 3:
                if m.shape[0] == 1:
                    m = m[0]
                else:
                    m = m[..., 0]
            if m.max() <= 1.0:
                m = (m * self.MASK_VAL).clip(0, self.MASK_VAL).astype(np.uint8)
            else:
                m = m.astype(np.uint8)
            return m

        mask_np = to_uint8(mask)

        if mask2 is not None:
            mask2_np = to_uint8(mask2)
            mask_np = np.maximum(mask_np, mask2_np)

        ys, xs = np.nonzero(mask_np)
        h, w = mask_np.shape
        out_mask = np.zeros((h, w), dtype=np.uint8)

        if len(xs) == 0 or len(ys) == 0:
            empty = torch.from_numpy(out_mask.astype(np.float32) / self.MASK_VAL).unsqueeze(0)
            return (empty, "Empty mask")

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        center = ((x_min + x_max)//2, (y_min + y_max)//2)

        fill_val = self.MASK_VAL
        thickness_val = line_thickness if mode == "line" else -1

        width = height = 0
        x1 = y1 = x2 = y2 = 0

        if shape == "rectangle":
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            width = self._round_up_64(width)
            height = self._round_up_64(height)
            x1, y1, x2, y2 = self._center_and_clamp(center[0], center[1], width, height, w, h)
            out_mask = self.draw_shape_np("rectangle", (w, h), [(x1, y1), (x2, y2)], fill_val, thickness_val)

        elif shape == "square":
            side = max(x_max - x_min + 1, y_max - y_min + 1)
            side = self._round_up_64(side)
            x1, y1, x2, y2 = self._center_and_clamp(center[0], center[1], side, side, w, h)
            out_mask = self.draw_shape_np("square", (w, h), [(x1, y1), (x2, y2)], fill_val, thickness_val)
            width = height = side

        elif shape == "circle":
            diameter = max(x_max - x_min + 1, y_max - y_min + 1)
            diameter = self._round_up_64(diameter)
            x1, y1, x2, y2 = self._center_and_clamp(center[0], center[1], diameter, diameter, w, h)
            out_mask = self.draw_shape_np("circle", (w, h), [(x1, y1), (x2, y2)], fill_val, thickness_val)
            width = height = diameter

        elif shape == "ellipse":
            width = self._round_up_64(x_max - x_min + 1)
            height = self._round_up_64(y_max - y_min + 1)
            x1, y1, x2, y2 = self._center_and_clamp(center[0], center[1], width, height, w, h)
            out_mask = self.draw_shape_np("ellipse", (w, h), [(x1, y1), (x2, y2)], fill_val, thickness_val)

        elif shape == "ratio":
            if ratio not in RATIO_PRESETS:
                ratio = next((r for r in RATIO_PRESETS if r), "1:1 ◻")
            rw, rh = RATIO_PRESETS[ratio]
            target_ratio = rw / rh
            bw, bh = x_max - x_min + 1, y_max - y_min + 1
            box_ratio = bw / bh if bh != 0 else float('inf')

            if ratio == "1:1 ◻":
                side = max(bw, bh)
                width_req, height_req = side, side
            else:
                if target_ratio > box_ratio:
                    width_req = int(math.ceil(bh * target_ratio))
                    height_req = bh
                else:
                    width_req = bw
                    height_req = int(math.ceil(bw / target_ratio))

            # round both dimensions to multiples of 64
            width_req = self._round_up_64(width_req)
            height_req = self._round_up_64(height_req)

            x1, y1, x2, y2 = self._center_and_clamp(center[0], center[1], width_req, height_req, w, h)
            out_mask = self.draw_shape_np("rectangle", (w, h), [(x1, y1), (x2, y2)], fill_val, thickness_val)
            width, height = width_req, height_req

        # --- Metadata
        gcd_wh = math.gcd(width, height) if width > 0 and height > 0 else 1
        sw, sh = width // gcd_wh, height // gcd_wh
        total_px = width * height
        mpix_str = "{:,} pixels".format(total_px)

        # find closest named ratio
        closest, mdiff = None, float('inf')
        trw = trh = None
        for name, (arw, arh) in RATIO_PRESETS.items():
            diff = abs((sw / sh) - (arw / arh))
            if diff < mdiff:
                mdiff = diff
                closest = name
                trw, trh = arw, arh

        precision = round((trw / trh) - (sw / sh), 2) if trw and trh else 0.0

        ui = (
            f"Width: {width} pixels\n"
            f"Height: {height} pixels\n"
            f"x: {x1} pixels\n"
            f"y: {y1} pixels\n"
            f"Aspect Ratio: {sw}:{sh} ({closest})\n"
            f"Megapixels: {mpix_str}\n"
            f"Precision: {precision}"
        )

        tm = torch.from_numpy(out_mask.astype(np.float32) / self.MASK_VAL).unsqueeze(0)
        return (tm, ui)


NODE_CLASS_MAPPINGS = {
    "mask_to_geometry": mask_to_geometry,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mask_to_geometry": "Mask to Geometry",
}