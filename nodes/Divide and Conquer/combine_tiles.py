# combine_tiles.py

import os
import sys
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.append(os.path.dirname(__file__))
from _utils_ import create_tile_coordinates, generate_matrix_ui, generate_tile_mask_np


class Combine_Tiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "UI")
    RETURN_NAMES = ("image", "ui",) 
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"
    
    def execute(self, images, dac_data):

        # Ensure `dac_data` is not a list
        if isinstance(dac_data, list):
            dac_data = dac_data[0]

        # Combine images into a single tensor
        out = []
        for i in range(len(images)):
            img = images[i]
            out.append(img)
        images = torch.stack(out).squeeze(1)

        # Import from dac_data
        upscaled_width = dac_data['upscaled_width']
        upscaled_height = dac_data['upscaled_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        # Import from Images
        tile_height, tile_width = images.shape[1:3]

        # Overlap / factor
        f_overlap_x = overlap_x // 4
        f_overlap_y = overlap_y // 4

        # Blend factor
        blend_x = math.sqrt(overlap_x)
        blend_y = math.sqrt(overlap_y)


        tile_coordinates, matrix = create_tile_coordinates(
            upscaled_width, upscaled_height, tile_width, tile_height,
            overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        output = torch.zeros((1, upscaled_height, upscaled_width, 3), dtype=images.dtype)

        for idx, (x, y) in enumerate(tile_coordinates):
            image_tile = images[idx]

            # Generate mask using NumPy
            mask_np = generate_tile_mask_np(
                x, y, tile_width, tile_height,
                upscaled_width, upscaled_height,
                f_overlap_x, f_overlap_y
            )

            # Apply blur
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            blur = ImageFilter.BoxBlur if overlap_x <= 64 or overlap_y <= 64 else ImageFilter.GaussianBlur
            mask_img = mask_img.filter(blur(radius=(blend_x, blend_y)))
            mask_np = np.array(mask_img, dtype=np.float32) / 255.0

            # Store tile mask
            full_tile_mask = torch.zeros((upscaled_height, upscaled_width), dtype=torch.float32)
            full_tile_mask[y:y + tile_height, x:x + tile_width] = torch.tensor(mask_np)

            # Blend into output
            mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(-1)
            output[:, y:y + tile_height, x:x + tile_width, :] *= (1 - mask_tensor)
            output[:, y:y + tile_height, x:x + tile_width, :] += image_tile * mask_tensor

        # Prepare a UI text representation of the tile order matrix.
        matrix_ui = generate_matrix_ui(matrix)

        return output, matrix_ui 
    
NODE_CLASS_MAPPINGS = {"Combine Tiles": Combine_Tiles,}
NODE_DISPLAY_NAME_MAPPINGS = {"Combine Tiles": "Combine Tiles",}
