# algorithm.py
# Inspired by https://github.com/kinfolk0117/ComfyUI_SimpleTiles
# Upscaling code from https://github.com/comfyanonymous/ComfyUI

import os
import sys
import math
import torch
import comfy.utils
from comfy import model_management

sys.path.append(os.path.dirname(__file__))
from _config_upscale_ import OVERLAP_DICT, TILE_ORDER_DICT, SCALING_METHODS, MIN_SCALE_FACTOR_THRESHOLD
from _utils_ import calculate_overlap


class DaC_Algorithm:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024,}),
                "tile_height": ("INT", {"default": 1024,}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/32 Tile",}),
                "min_scale_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 16.0, "step": 0.01}),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "spiral",}),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),  
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),  # Now optional
                "use_upscale_with_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "DAC_DATA", "STRING")
    RETURN_NAMES = ("IMAGE", "dac_data", "ui")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"
    DESCRIPTION = """
Calculate the best dimensions and optionally upscale an image
while maintaining minimum tile overlap and scale factor constraints.
Steudio
"""

    def execute(self, image, scaling_method, tile_width, tile_height, min_overlap, min_scale_factor, tile_order, upscale_model=None, use_upscale_with_model=True):

        tile_order = TILE_ORDER_DICT.get(tile_order, 0)  # Default to 0 if the key is not found
        _, height, width, _ = image.shape

        # --- Auto overlap branch ---
        if min_overlap == "Auto":
            # If image smaller than tile, upscale to minimum tile size
            target_width = max(width, tile_width)
            target_height = max(height, tile_height)

            upscale_ratio_x = target_width / width
            upscale_ratio_y = target_height / height
            upscale_ratio = max(upscale_ratio_x, upscale_ratio_y)

            upscaled_width = int(width * upscale_ratio)
            upscaled_height = int(height * upscale_ratio)

            grid_x = math.ceil(upscaled_width / tile_width)
            grid_y = math.ceil(upscaled_height / tile_height)

            overlap_x = 0 if grid_x == 1 else round((tile_width * grid_x - upscaled_width) / (grid_x - 1))
            overlap_y = 0 if grid_y == 1 else round((tile_height * grid_y - upscaled_height) / (grid_y - 1))
        else:
            overlap = OVERLAP_DICT.get(min_overlap, 0)  # Default to 0 if the key is not found

            # Calculate initial overlaps
            overlap_x = calculate_overlap(tile_width, overlap)
            overlap_y = calculate_overlap(tile_height, overlap)

            # Ensure min_scale_factor
            min_scale_factor = max(min_scale_factor, MIN_SCALE_FACTOR_THRESHOLD)

            if width <= height:
                # Calculate initial upscaled width based on min_scale_factor
                multiply_factor = math.ceil(min_scale_factor * width / tile_width)
                while True:
                    upscaled_width = tile_width * multiply_factor
                    grid_x = math.ceil(upscaled_width / tile_width)
                    upscaled_width = (tile_width * grid_x) - (overlap_x * (grid_x - 1))
                    upscale_ratio = upscaled_width / width
                    if upscale_ratio >= min_scale_factor:
                        break
                    multiply_factor += 1
                upscaled_height = int(height * upscale_ratio)
                grid_y = math.ceil((upscaled_height - overlap_y) / (tile_height - overlap_y))
                overlap_y = round((tile_height * grid_y - upscaled_height) / (grid_y - 1))
            else:
                multiply_factor = math.ceil(min_scale_factor * height / tile_height)
                while True:
                    upscaled_height = tile_height * multiply_factor
                    grid_y = math.ceil(upscaled_height / tile_height)
                    upscaled_height = (tile_height * grid_y) - (overlap_y * (grid_y - 1))
                    upscale_ratio = upscaled_height / height
                    if upscale_ratio >= min_scale_factor:
                        break
                    multiply_factor += 1
                upscaled_width = int(width * upscale_ratio)
                grid_x = math.ceil((upscaled_width - overlap_x) / (tile_width - overlap_x))
                overlap_x = round((tile_width * grid_x - upscaled_width) / (grid_x - 1))

        effective_upscale = round(upscaled_width / width, 2)
        upscaled_image_size = f"{upscaled_width}x{upscaled_height}"
        original_image = f"{width}x{height}"
        grid_n_xy = f"{grid_x}x{grid_y}"
        Tiles_Q = grid_x * grid_y

        dac_data = {
            'upscaled_width': upscaled_width,
            'upscaled_height': upscaled_height,
            'tile_width': tile_width,
            'tile_height': tile_height,
            'overlap_x': overlap_x,
            'overlap_y': overlap_y,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'tile_order': tile_order,
        }

        if use_upscale_with_model and upscale_model:
            # Upscale the image with model
            device = model_management.get_torch_device()
            memory_required = model_management.module_size(upscale_model.model)
            memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
            memory_required += image.nelement() * image.element_size()
            model_management.free_memory(memory_required, device)

            upscale_model.to(device)
            in_img = image.movedim(-1, -3).to(device)

            tile = 512
            overlap_value = 32

            oom = True
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap_value)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap_value, upscale_amount=upscale_model.scale, pbar=pbar)
                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e

            upscale_model.to("cpu")
            Upscaled_with_Model = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

            samples = Upscaled_with_Model.movedim(-1, 1)
        else:
            samples = image.movedim(-1, 1)  # Use original image

        if upscaled_width == 0:
            upscaled_width = max(1, round(samples.shape[3] * upscaled_height / samples.shape[2]))
        elif upscaled_height == 0:
            upscaled_height = max(1, round(samples.shape[2] * upscaled_width / samples.shape[3]))

        upscaled_image = comfy.utils.common_upscale(samples, upscaled_width, upscaled_height, scaling_method, crop=0).movedim(1, -1)

        algo_ui = f"Divide and Conquer Algorithm:\nOriginal Image Size: {original_image}\nUpscaled Image Size: {upscaled_image_size}\nGrid: {grid_n_xy} ({Tiles_Q} tiles)\nOverlap_x: {overlap_x} pixels\nOverlap_y: {overlap_y} pixels\nEffective_upscale: {effective_upscale}"

        return (upscaled_image, dac_data, algo_ui)

NODE_CLASS_MAPPINGS = {"Divide and Conquer Algorithm": DaC_Algorithm,}
NODE_DISPLAY_NAME_MAPPINGS = {"Divide and Conquer Algorithm": "Divide and Conquer Algorithm",}