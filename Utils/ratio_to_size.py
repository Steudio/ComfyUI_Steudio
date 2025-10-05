# Ratio_to_Size.py

import os, sys, math
sys.path.append(os.path.dirname(__file__))
from _config_utils_ import RATIO_PRESETS

class Ratio_to_Size:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (list(RATIO_PRESETS.keys()),),
                "megapixel": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 16.00, "step": 0.01}), # Allow 0 for auto megapixel mode
                "priority": (["ratio", "megapixel"], {"default": "megapixel"}),
                "divisible": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
            },
            "optional": {
                "image": ("IMAGE",),    # Optional for auto modes
            }
        }

    RETURN_TYPES = ("INT", "INT", "UI",)
    RETURN_NAMES = ("width", "height", "ui",)
    FUNCTION = "calculate_dimensions"
    CATEGORY = "Steudio/Utils"

    def calculate_dimensions(self, ratio, megapixel, divisible, priority, image=None):

        # --- AUTO RATIO MODE ---
        if ratio == "auto":
            if image is not None:
                _, height, width, _ = image.shape
                gcd = math.gcd(width, height)
                simplified_width = width // gcd
                simplified_height = height // gcd

                closest_ratio = None
                min_difference = float('inf')
                for name, (rw, rh) in RATIO_PRESETS.items():
                    if name in ("auto", ""):
                        continue
                    difference = abs(simplified_width / simplified_height - rw / rh)
                    if difference < min_difference:
                        min_difference = difference
                        closest_ratio = name
                        aspect_width, aspect_height = rw, rh

                ratio_label = closest_ratio
            else:
                aspect_width, aspect_height = 1, 1
                ratio_label = "1:1 ◻"  # fallback if no image
        else:
            aspect_width, aspect_height = RATIO_PRESETS.get(ratio, (1, 1))
            ratio_label = ratio

        # --- AUTO MEGAPIXEL MODE ---
        if megapixel == 0:
            if image is not None:
                _, img_h, img_w, _ = image.shape
                megapixel = (img_w * img_h) / (1024 * 1024)
            else:
                megapixel = 1.0  # fallback if no image provided

        # --- Dimension calculation ---
        target_ratio = aspect_width / aspect_height
        target_pixels = int(megapixel * 1024 * 1024)

        # Initial estimate
        width = int((target_pixels * target_ratio) ** 0.5)
        height = int(width / target_ratio)

        # Snap to multiples of `divisible`
        width = max(divisible, (width // divisible) * divisible)
        height = max(divisible, (height // divisible) * divisible)

        # Try small adjustments based on priority
        best_w, best_h = width, height
        best_ratio_diff = abs((width / height) - target_ratio)
        best_pixel_diff = abs((width * height) - target_pixels) / target_pixels

        for dw in (-divisible, 0, divisible):
            for dh in (-divisible, 0, divisible):
                w = max(divisible, width + dw)
                h = max(divisible, height + dh)
                ratio_diff = abs((w / h) - target_ratio)
                pixel_diff = abs((w * h) - target_pixels) / target_pixels

                if priority == "ratio":
                    # Ratio first, then pixel tolerance
                    if ratio_diff < best_ratio_diff and pixel_diff <= 0.05:
                        best_w, best_h = w, h
                        best_ratio_diff = ratio_diff
                        best_pixel_diff = pixel_diff

                elif priority == "megapixel":
                    # Pixel count first, then ratio tolerance
                    if pixel_diff < best_pixel_diff and ratio_diff <= 0.05:
                        best_w, best_h = w, h
                        best_pixel_diff = pixel_diff
                        best_ratio_diff = ratio_diff

        width, height = best_w, best_h

        f_megapixel = "{:,}".format(width * height)
        f_precision = round((aspect_width / aspect_height) - (width / height), 4)

        ui = (
            f"Priority: {priority}\n"
            f"Ratio: {ratio_label}\n"
            f"Width: {width}\n"
            f"Height: {height}\n"
            f"{f_megapixel} pixels\n"
            f"Precision: {f_precision}\n"
            f"Divisible by: {divisible}\n"
        )

        return int(width), int(height), ui
    

NODE_CLASS_MAPPINGS = {
    "Ratio to Size": Ratio_to_Size,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Ratio to Size": "Ratio to Size",
}