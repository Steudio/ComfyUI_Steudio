# Inspired by https://github.com/kinfolk0117/ComfyUI_SimpleTiles
# Upscaling code from https://github.com/comfyanonymous/ComfyUI
# Created by Steudio

import sys
import os
import torch
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import comfy.utils
from comfy import model_management

OVERLAP_DICT = {
    "None": 0,
    "1/64 Tile": 0.015625,
    "1/32 Tile": 0.03125,
    "1/16 Tile": 0.0625,
    "1/8 Tile": 0.125,
    "1/4 Tile": 0.25,
    "1/2 Tile": 0.5,
}

TILE_ORDER_DICT = {
    "linear": 0,
    "spiral": 1
}

SCALING_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos"
]

MIN_SCALE_FACTOR_THRESHOLD = 1.0

def calculate_overlap(tile_size, overlap_fraction):
    return int(overlap_fraction * tile_size)

def create_tile_coordinates(image_width, image_height, tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y, tile_order):
    tiles = []
    num_columns = grid_x
    num_rows = grid_y
    matrix = [['' for _ in range(num_columns)] for _ in range(num_rows)]

    # Generate tiles in grid layout
    for row in range(grid_y):
        y = row * (tile_height - overlap_y)
        if row == grid_y - 1:
            y = image_height - tile_height
        for col in range(grid_x):
            x = col * (tile_width - overlap_x)
            if col == grid_x - 1:
                x = image_width - tile_width
            tiles.append((x, y))
    
    if tile_order == 1:  # Spiral order
        # Rearrange tiles in an outward clockwise spiral pattern starting from the center
        spiral_tiles = []
        visited = set()
        x, y = num_columns // 2, num_rows // 2
        dx, dy = 1, 0  # Start moving right
        layer = 1

        while len(spiral_tiles) < len(tiles):
            for _ in range(2):
                for _ in range(layer):
                    if 0 <= x < num_columns and 0 <= y < num_rows and (x, y) not in visited:
                        index = y * num_columns + x
                        if index < len(tiles):
                            spiral_tiles.append(tiles[index])
                            visited.add((x, y))
                    x += dx
                    y += dy
                dx, dy = -dy, dx  # Rotate direction clockwise
            layer += 1

        spiral_tiles.reverse()
        tiles = spiral_tiles

    # Rebuild matrix to match tile order
    for i, (x, y) in enumerate(tiles):
        row, col = y // (tile_height - overlap_y), x // (tile_width - overlap_x)
        matrix[row][col] = f"{i + 1} ({x},{y})"

    return tiles, matrix



class DaC_Algorithm:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024,}),
                "tile_height": ("INT", {"default": 1024,}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/32 Tile",}),
                "min_scale_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 8.0}),
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

        overlap = OVERLAP_DICT.get(min_overlap, 0)  # Default to 0 if the key is not found
        tile_order = TILE_ORDER_DICT.get(tile_order, 0)  # Default to 0 if the key is not found

        _, height, width, _ = image.shape

        # Calculate initial overlaps
        overlap_x = calculate_overlap(tile_width, overlap)
        overlap_y = calculate_overlap(tile_height, overlap)

        # Ensure min_scale_factor is at least 1.01 to avoid divide by zero error
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


class Divide_Image_Select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
                "tile": ("INT", { "default": 0, "min": 0, "step": 1, }),
            },
        }


    RETURN_TYPES = ("IMAGE", "UI")
    RETURN_NAMES = ("TILE(S)", "ui",)
    OUTPUT_IS_LIST = (True,False)
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"
    DESCRIPTION = """
tile 0 = All tiles
tile # = Tile #
"""

    def execute(self, image, tile, dac_data,):
        # # Ensure `ui` is not a list
        # if isinstance(ui, list):
        #     ui = ui[0]

        image_height = image.shape[1]
        image_width = image.shape[2]



        tile_width = dac_data['tile_width']
        tile_height = dac_data['tile_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        tile_coordinates, matrix = create_tile_coordinates(
            image_width, image_height, tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        iteration = 1

        image_tiles = []
        for tile_coordinate in tile_coordinates:
            iteration += 1

            image_tile = image[
                :,
                tile_coordinate[1] : tile_coordinate[1] + tile_height,
                tile_coordinate[0] : tile_coordinate[0] + tile_width,
                :,
            ]

            image_tiles.append(image_tile)

        all_tiles = torch.cat(image_tiles, dim=0)
        selected_tile = image_tiles[tile - 1]

        if tile == 0:
            tile_or_tiles = all_tiles
        else:
            tile_or_tiles = selected_tile

        matrix_ui = "Divide and Conquer Matrix:\n" + '\n'.join([' '.join(row) for row in matrix])


        return ([tile_or_tiles[i].unsqueeze(0) for i in range(tile_or_tiles.shape[0])], matrix_ui)


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
        
        overlap_factor = 4
        # blur_factor = 20

        # Import from dac_data
        upscaled_width = dac_data['upscaled_width']
        upscaled_height = dac_data['upscaled_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        # Import from Images
        tile_width = images.shape[2]
        tile_height = images.shape[1]

        # Overlap / factor
        f_overlap_x = overlap_x //overlap_factor
        f_overlap_y = overlap_y //overlap_factor

        # Blend factor
        blend_x = math.sqrt(overlap_x)
        blend_y = math.sqrt(overlap_y)


        tile_coordinates, matrix = create_tile_coordinates(
            upscaled_width, upscaled_height, tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        original_shape = (1, upscaled_height, upscaled_width, 3)
        output = torch.zeros(original_shape, dtype=images.dtype)

        index = 0
        iteration = 1
        for tile_coordinate in tile_coordinates:
            image_tile = images[index]
            x = tile_coordinate[0]
            y = tile_coordinate[1]

            iteration += 1

            # Create mask for the tile
            mask = Image.new("L", (tile_width, tile_height), 0)
            draw = ImageDraw.Draw(mask)

            # Do not apply gaussian to tile at the edge of the image
            # 1234 Detect corners top/left top/right bottom/left bottom/right and grid >1
            if x == 0 and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([x, y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x == upscaled_width - tile_width and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, y, tile_width, tile_height - f_overlap_y], fill=255)
            elif x == 0 and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([x, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            elif x == upscaled_width - tile_width and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height], fill=255)
            # 5678 Detect corners 3 edges and grid =1
            elif x == 0 and y == 0 and upscaled_height == tile_height:
                draw.rectangle([x, y, tile_width - f_overlap_x, tile_height], fill=255)
            elif x == upscaled_width - tile_width and y == 0 and upscaled_height == tile_height:
                draw.rectangle([f_overlap_x, y, tile_width, tile_height], fill=255)
            elif x == 0 and y == 0 and upscaled_width == tile_width:
                draw.rectangle([x, y, tile_width, tile_height - f_overlap_y], fill=255)
            elif x == 0 and y == upscaled_height - tile_height and upscaled_width == tile_width:
                draw.rectangle([x, f_overlap_y, tile_width, tile_height], fill=255)
            # 9 12 Detect top or bottom edges
            elif x != 0 and x !=upscaled_width - tile_width and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x != 0 and x !=upscaled_width - tile_width and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            # 10 11 Detect left or right edges
            elif x == 0 and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([x, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x == upscaled_width - tile_width and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height - f_overlap_y], fill=255)
            # 13 Detect top and bottom edges
            elif x != 0 and x !=upscaled_width - tile_width and y == 0 and upscaled_height == tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, y, tile_width - f_overlap_x, tile_height], fill=255)
            # 14 Detect left and right edges
            elif x == 0 and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width == tile_width:
                draw.rectangle([x, f_overlap_y, tile_width, tile_height - f_overlap_y], fill=255)
            # 15 Detect not touching any edges
            elif x != 0 and x !=upscaled_width - tile_width and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)

            # Use a box blur if overlap is getting too narrow
            if overlap_x <= 64 or overlap_y <= 64:
                mask = mask.filter(ImageFilter.BoxBlur(radius=(blend_x, blend_y)))
            else:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=(blend_x, blend_y)))

            mask_np = np.array(mask) / 255.0
            mask_tensor = torch.tensor(mask_np, dtype=images.dtype).unsqueeze(0).unsqueeze(-1)

            output[:, y : y + tile_height, x : x + tile_width, :] *= (1 - mask_tensor)
            output[:, y : y + tile_height, x : x + tile_width, :] += image_tile * mask_tensor

            index += 1

        matrix_ui = "Divide and Conquer Matrix:\n" + '\n'.join([' '.join(row) for row in matrix])


        return output, matrix_ui 
    

NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Divide Image and Select Tile": Divide_Image_Select,
    "Combine Tiles": Combine_Tiles,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Divide Image and Select Tile": "Divide Image and Select Tile",
    "Combine Tiles": "Combine Tiles",
}
