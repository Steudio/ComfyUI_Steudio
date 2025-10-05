# tiles_visualizer.py

import os
import sys
import colorsys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(__file__))
from _utils_ import create_tile_coordinates


class Tiles_Visualizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
                "input_opacity": ("INT", {"default": 75, "min": 0, "max": 100}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "label_size": ("INT", {"default": 50, "min": 10, "max": 200}),
                "tile": ("INT", {"default": 0, "min": 0, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_tiles"
    CATEGORY = "Steudio/Divide and Conquer"

    def screen_blend(self, base, overlay):
        base_np = np.asarray(base, dtype=np.float32) / 255.0
        overlay_np = np.asarray(overlay, dtype=np.float32) / 255.0

        result_rgb = 1.0 - (1.0 - base_np[..., :3]) * (1.0 - overlay_np[..., :3])
        result_rgb = np.clip(result_rgb, 0, 1) * 255

        alpha = np.maximum(base_np[..., 3], overlay_np[..., 3]) * 255
        blended = np.dstack((result_rgb, alpha)).astype(np.uint8)
        return Image.fromarray(blended, mode="RGBA")

    def visualize_tiles(self, input_image, dac_data, input_opacity, line_thickness, label_size, tile):
        if isinstance(dac_data, list):
            dac_data = dac_data[0]

        up_w, up_h = dac_data["upscaled_width"], dac_data["upscaled_height"]
        tile_w, tile_h = dac_data["tile_width"], dac_data["tile_height"]

        # Convert tensor to PIL RGBA
        img_np = input_image.squeeze(0).cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        base_img = Image.fromarray(img_np).resize((up_w, up_h), Image.LANCZOS).convert("RGBA")

        # Dim image based on opacity
        bg = Image.new("RGBA", base_img.size, (0, 0, 0, 255))
        base_img = Image.alpha_composite(bg, base_img)
        np_base = np.array(base_img, dtype=np.float32)
        np_base[..., :3] *= input_opacity / 100.0
        np_base[..., 3] = 255
        base_img = Image.fromarray(np_base.clip(0, 255).astype(np.uint8), mode="RGBA")

        tiles_only, _ = create_tile_coordinates(
            up_w, up_h, tile_w, tile_h,
            dac_data.get("overlap_x", 0), dac_data.get("overlap_y", 0),
            dac_data["grid_x"], dac_data["grid_y"],
            dac_data.get("tile_order", 0)
        )
        tiles = [(i + 1, x, y) for i, (x, y) in enumerate(tiles_only)]

        try:
            font_path = "arial.ttf"
            font = ImageFont.truetype(font_path, label_size)
        except IOError:
            font_path = "DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, label_size)

        def unique_color(n):
            h = (n * 37) % 360
            s, v = 0.9, 0.95
            return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h / 360, s, v)) + (255,)

        # Draw overlay
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for n, x, y in tiles:
            if tile != 0 and tile != n:
                continue
            line_color = unique_color(n)
            label_text = f"{n} ({x},{y})"
            draw.rectangle([x, y, x + tile_w, y + tile_h],
                           outline=line_color,
                           width=line_thickness)
            draw.text((x + 10, y + 10), label_text, font=font, fill=(255, 255, 255, 255))

        # Apply screen blend
        base_img = self.screen_blend(base_img, overlay)
        tensor = torch.from_numpy(np.array(base_img, dtype=np.float32) / 255.0).unsqueeze(0)
        return (tensor,)


NODE_CLASS_MAPPINGS = {"Tiles Visualizer": Tiles_Visualizer,}
NODE_DISPLAY_NAME_MAPPINGS = {"Tiles Visualizer": "Tiles Visualizer",}