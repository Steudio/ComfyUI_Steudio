# combine_tiles_advanced.py

import os
import sys
import math
import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.append(os.path.dirname(__file__))
from _utils_ import create_tile_coordinates, generate_matrix_ui, generate_tile_mask_np


class Combine_Tiles_Advanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
                "tile": ("INT", {"default": 0, "min": 0, "step": 1}),  # can be int or list of ints
            },
            "optional": {
                "background": ("IMAGE",),  # optional background image
            }
        }

    # image (RGB), tiles+alpha (list of RGBA), image+alpha (RGBA), ui
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "UI")
    RETURN_NAMES = ("image", "tiles+alpha", "image+alpha", "ui")
    INPUT_IS_LIST = (True, True, False, False)   # images + dac_data as lists
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"

    def execute(self, images, dac_data, tile, background=None):
        # Unwrap list inputs
        if isinstance(dac_data, list):
            dac_data = dac_data[0]
        if isinstance(tile, list) and len(tile) == 1:
            tile = tile[0]  # e.g., [3] -> 3

        # Allow tile to be a list/tuple of ints
        if isinstance(tile, (list, tuple)):
            tile_numbers = list(tile)
        elif tile == 0:
            tile_numbers = 0
        else:
            tile_numbers = [tile]

        images = torch.stack(images).squeeze(1)  # (N, H, W, 3)
        device = images.device
        dtype = images.dtype

        # Parameters
        upscaled_width = dac_data['upscaled_width']
        upscaled_height = dac_data['upscaled_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        tile_height, tile_width = images.shape[1:3]
        f_overlap_x = overlap_x // 4
        f_overlap_y = overlap_y // 4
        blend_x = math.sqrt(overlap_x)
        blend_y = math.sqrt(overlap_y)

        tile_coordinates, matrix = create_tile_coordinates(
            upscaled_width, upscaled_height, tile_width, tile_height,
            overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        # Background setup
        if background is not None:
            if isinstance(background, list):
                background = background[0]
            # Ensure shape matches (1,H,W,3)
            if background.shape[1:3] != (upscaled_height, upscaled_width):
                raise ValueError("Background image must match upscaled dimensions")
            bg_rgb = background.to(device=device, dtype=dtype)
        else:
            bg_rgb = torch.zeros((1, upscaled_height, upscaled_width, 3), dtype=dtype, device=device)

        # Prepare outputs
        output_rgb_visual = bg_rgb.clone()  # start from background
        output_rgb_straight = bg_rgb.clone()
        output_alpha_union = torch.zeros((upscaled_height, upscaled_width), dtype=torch.float32, device=device)

        tile_masks = []
        tile_images = []

        for img_i, (x, y) in enumerate(tile_coordinates if tile_numbers == 0 else [tile_coordinates[t-1] for t in tile_numbers]):
            if img_i >= images.shape[0]:
                raise IndexError(f"No image provided for tile index {img_i} in this batch.")
            image_tile = images[img_i]  # (H, W, 3)

            # Generate mask
            mask_np = generate_tile_mask_np(
                x, y, tile_width, tile_height,
                upscaled_width, upscaled_height,
                f_overlap_x, f_overlap_y
            )
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            blur = ImageFilter.BoxBlur if overlap_x <= 64 or overlap_y <= 64 else ImageFilter.GaussianBlur
            mask_img = mask_img.filter(blur(radius=max(blend_x, blend_y)))
            mask_np = np.array(mask_img, dtype=np.float32) / 255.0

            mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)

            # Store tile mask
            full_tile_mask = torch.zeros((upscaled_height, upscaled_width), dtype=torch.float32, device=device)
            full_tile_mask[y:y + tile_height, x:x + tile_width] = mask_t[y:y + tile_height, x:x + tile_width] if mask_t.shape == full_tile_mask.shape else mask_t
            tile_masks.append(full_tile_mask.unsqueeze(0))

            # Store tile image on full canvas
            tile_canvas = torch.zeros((1, upscaled_height, upscaled_width, 3), dtype=dtype, device=device)
            tile_canvas[:, y:y + tile_height, x:x + tile_width, :] = image_tile
            tile_images.append(tile_canvas)

            # Visual composite (for "image")
            region_rgb = output_rgb_visual[:, y:y + tile_height, x:x + tile_width, :]
            mask_region = full_tile_mask[y:y + tile_height, x:x + tile_width].unsqueeze(0).unsqueeze(-1).to(dtype)
            output_rgb_visual[:, y:y + tile_height, x:x + tile_width, :] = region_rgb * (1.0 - mask_region) + image_tile.unsqueeze(0) * mask_region

            # Straight RGBA export
            output_rgb_straight[:, y:y + tile_height, x:x + tile_width, :] = image_tile
            region_a = output_alpha_union[y:y + tile_height, x:x + tile_width]
            mask_local = full_tile_mask[y:y + tile_height, x:x + tile_width]
            region_a = 1.0 - (1.0 - region_a) * (1.0 - mask_local)
            output_alpha_union[y:y + tile_height, x:x + tile_width] = region_a

        # Build tiles+alpha list (straight per tile)
        tile_masks_tensor = torch.stack(tile_masks)                 # (N, 1, H, W)
        tile_images_tensor = torch.stack(tile_images)               # (N, 1, H, W, 3)
        tile_images_squeezed = tile_images_tensor.squeeze(1)        # (N, H, W, 3)
        tile_masks_squeezed = tile_masks_tensor.squeeze(1).unsqueeze(-1)  # (N, H, W, 1)
        tile_images_with_alpha_tensor = torch.cat(
            [tile_images_squeezed, tile_masks_squeezed], dim=-1
        ).unsqueeze(1)  # (N, 1, H, W, 4)

        # Final outputs
        image_rgb = output_rgb_visual
        alpha_ = output_alpha_union.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        image_plus_alpha = torch.cat([output_rgb_straight, alpha_.to(dtype)], dim=-1)

        matrix_ui = generate_matrix_ui(matrix)

        return image_rgb, tile_images_with_alpha_tensor, image_plus_alpha, matrix_ui


NODE_CLASS_MAPPINGS = {"Combine Tiles Advanced": Combine_Tiles_Advanced}
NODE_DISPLAY_NAME_MAPPINGS = {"Combine Tiles Advanced": "Combine Tiles Advanced"}