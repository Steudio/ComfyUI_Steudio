# Inspired by https://github.com/kinfolk0117/ComfyUI_SimpleTiles
# Created by Steudio

import sys
import os
import torch
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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
            matrix[row][col] = f"({x},{y})"


    print("\n" * 2)
    print("Divide and Conquer Matrix:")
    for row in matrix:
        print(' '.join(row))
    print("\n" * 2)


    if tile_order == 1:  # Spiral order
        # Rearrange tiles in an outward clockwise spiral pattern starting from the center
        spiral_tiles = []
        center_x, center_y = num_columns // 2, num_rows // 2
        x, y = center_x, center_y
        dx, dy = 1, 0  # Start moving right
        layer, steps = 1, 0

        while len(spiral_tiles) < len(tiles):
            for _ in range(2):
                for _ in range(layer):
                    if 0 <= x < num_columns and 0 <= y < num_rows:
                        index = y * num_columns + x
                        if index < len(tiles):
                            spiral_tiles.append(tiles[index])
                        steps += 1
                    x += dx
                    y += dy
                dx, dy = -dy, dx  # Rotate direction clockwise
            layer += 1

        # Reverse the tiles
        spiral_tiles.reverse()
        tiles = spiral_tiles

    return tiles


class DaC_Algorithm:
    @classmethod
    def INPUT_TYPES(s):
        Overlap_list = [
                "None",
                    '1/64 Tile',
                    '1/32 Tile',
                    '1/16 Tile',
                    '1/8 Tile',
                    '1/4 Tile',
                    '1/2 Tile',
                    ]
        tile_order_list = [
                "linear", 'spiral',]
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024,}),
                "tile_height": ("INT", {"default": 1024,}),
                "overlap": (Overlap_list, {"default": "1/32 Tile",}),
                "multiply_factor": ("INT", {"default": 3, "min":2, "max":9,}),
                "tile_order": (tile_order_list, {"default": "spiral",}),
            }
        }


    RETURN_TYPES = ("INT", "INT", "DAC_DATA",)
    RETURN_NAMES = ("image_width", "image_height", "dac_data",)
    OUTPUT_NODE = True
    FUNCTION = "Calc"
    CATEGORY = "Steudio/Divide and Conquer"

    def Calc(self, image, tile_width, tile_height, overlap, multiply_factor, tile_order,):

        if tile_order == "linear":
            tile_order = 0
        elif tile_order == "spiral":
            tile_order = 1


        if overlap == "None":
            overlap = 0
        elif overlap == "1/2 Tile":
            overlap = 0.5
        elif overlap == "1/4 Tile":
            overlap = 0.25
        elif overlap == "1/8 Tile":
            overlap = 0.125
        elif overlap == "1/16 Tile":
            overlap = 0.0625
        elif overlap == "1/32 Tile":
            overlap = 0.03125
        elif overlap == "1/64 Tile":
            overlap = 0.015625

        _, height, width, _ = image.shape

        # Calculate initial overlaps
        overlap_x = int(overlap * tile_width)
        overlap_y = int(overlap * tile_height)

        if width <= height:
            # Calculate initial upscaled width first
            upscaled_width = width * multiply_factor
            # Determine grid_x
            grid_x = math.ceil(upscaled_width / tile_width)
            # Recalculate upscaled width
            upscaled_width = (tile_width * grid_x) - (overlap_x * (grid_x - 1))
            # Calculate upscale ratio
            upscale_ratio = upscaled_width / width
            # Determine upscaled height
            upscaled_height = int(height * upscale_ratio)
            # Determine grid_y
            grid_y = math.ceil((upscaled_height - overlap_y) / (tile_height - overlap_y))
            # Recalculate overlap_y
            overlap_y = round((tile_height * grid_y - upscaled_height) / (grid_y - 1))
        else:
            # Calculate initial upscaled height first
            upscaled_height = height * multiply_factor
            # Determine grid_y
            grid_y = math.ceil(upscaled_height / tile_height)
            # Recalculate upscaled height
            upscaled_height = (tile_height * grid_y) - (overlap_y * (grid_y - 1))
            # Calculate upscale ratio
            upscale_ratio = upscaled_height / height
            # Determine upscaled width
            upscaled_width = int(width * upscale_ratio)
            # Determine grid_x
            grid_x = math.ceil((upscaled_width - overlap_x) / (tile_width - overlap_x))
            # Recalculate overlap_x
            overlap_x = round((tile_width * grid_x - upscaled_width) / (grid_x - 1))


        effective_upscale = round(upscaled_width / width, 2)
        upscaled_image = f"{upscaled_width}x{upscaled_height}"
        original_image = f"{width}x{height}"
        grid_n_xy = f"{grid_x}x{grid_y}"
        # overlap_xy = f"x{overlap_x} y{overlap_y}"


        dac_data = {'upscaled_width': upscaled_width,
                    'upscaled_height': upscaled_height,
                    'tile_width': tile_width,
                    'tile_height': tile_height,
                    'overlap_x': overlap_x,
                    'overlap_y': overlap_y,
                    'grid_x': grid_x,
                    'grid_y': grid_y,
                    'tile_order': tile_order,
                    }

        ui_data = {
            'Original Image Size':original_image,
            'Upscaled Image Size':upscaled_image,
            'Grid':grid_n_xy,
            'overlap_x': overlap_x,
            'overlap_y': overlap_y,
            'effective_upscale': effective_upscale,
            }

        print()
        for key, value in ui_data.items():
            print(f"{key}: {value}")


        return [upscaled_width, upscaled_height, dac_data]

class Divide_Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"

    def execute(self, image, dac_data,):
        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_width = dac_data['tile_width']
        tile_height = dac_data['tile_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        tile_coordinates = create_tile_coordinates(
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

        tiles_tensor = torch.stack(image_tiles).squeeze(1)

        return (tiles_tensor,)

class Combine_Tiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
                #"overlap_factor": ("INT",{"default": 4, "min": 3, "max": 6}),  # Added blur blend parameter
                #"blur_factor": ("INT",{"default": 20, "min": 15, "max": 25}),  # Added blur blend parameter
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"

    def execute(self, images, dac_data,):  # Added index parameter

        overlap_factor = 4
        blur_factor = 20

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
        blend_x = overlap_x //blur_factor
        blend_y = overlap_y //blur_factor

        tile_coordinates = create_tile_coordinates(
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
                draw.rectangle([f_overlap_x, y, tile_width, tile_height], fill=255)
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

            mask = mask.filter(ImageFilter.GaussianBlur(radius=(blend_x, blend_y)))

            mask_np = np.array(mask) / 255.0
            mask_tensor = torch.tensor(mask_np, dtype=images.dtype).unsqueeze(0).unsqueeze(-1)

            output[:, y : y + tile_height, x : x + tile_width, :] *= (1 - mask_tensor)
            output[:, y : y + tile_height, x : x + tile_width, :] += image_tile * mask_tensor

            index += 1
        return [output]




NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Divide Image": Divide_Image,
    "Combine Tiles": Combine_Tiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Divide Image": "Divide Image",
    "Combine Tiles": "Combine Tiles",
}


# blur_factor
# Purpose: Controls the amount of Gaussian blur applied to the mask used for blending the tiles.
# Effect: A higher blur_factor results in a smoother transition between tiles, reducing visible seams. Conversely, a lower blur_factor will make the transitions sharper, which might reveal more seams.

# overlap_factor
# Purpose: Determines the extent of the overlap between adjacent tiles.
# Effect: A higher overlap_factor reduces the effective overlap area, which can help in reducing redundancy and potential artifacts. A lower overlap_factor increases the overlap area, which can help in creating a more seamless blend but might introduce more redundancy.


# Optimal Values
# blur_factor: 15-25
# Reason: This range provides a good balance between smooth transitions and maintaining detail. It helps to blend the tiles effectively without overly blurring the image.
# overlap_factor: 3-6
# Reason: This range ensures sufficient overlap to avoid visible seams while minimizing redundancy. It helps to create a seamless blend without excessive overlap that could lead to artifacts.