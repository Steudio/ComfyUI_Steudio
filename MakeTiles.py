# Original nodes from https://github.com/kinfolk0117/ComfyUI_SimpleTiles
# Modified by Steudio

import sys
import os

import torch
import math
import numpy as np

from PIL import Image, ImageDraw, ImageFilter



sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


def generate_tiles(image_width, image_height, tile_width, tile_height, overlap_x, overlap_y):
    tiles = []
    matrix = [['' for _ in range((image_width + tile_width - overlap_x - 1) // (tile_width - overlap_x))] for _ in range((image_height + tile_height - overlap_y - 1) // (tile_height - overlap_y))]

    y = 0
    row = 0
    while y < image_height:
        next_y = y + tile_height - overlap_y
        if y + tile_height >= image_height:
            y = max(image_height - tile_height, 0)
            next_y = image_height

        x = 0
        col = 0
        while x < image_width:
            next_x = x + tile_width - overlap_x
            if x + tile_width >= image_width:
                x = max(image_width - tile_width, 0)
                next_x = image_width

            tiles.append((x, y))
            matrix[row][col] = f"({x},{y})"

            if next_x >= image_width:
                break
            x = next_x
            col += 1

        if next_y >= image_height:
            break
        y = next_y
        row += 1

    # Define the order of processing
    center_x = image_width / 2
    center_y = image_height / 2
    corners = [(0, 0), (image_width - tile_width, 0), (0, image_height - tile_height), (image_width - tile_width, image_height - tile_height)]
    middle = (center_x - tile_width / 2, center_y - tile_height / 2)

    def tile_priority(tile):
        x, y = tile
        if tile in corners:
            return 0
        elif y == center_y - tile_height / 2 and x != center_x - tile_width / 2:
            return 1
        elif x == center_x - tile_width / 2 and y != center_y - tile_height / 2:
            return 2
        elif tile == middle:
            return 4  # Ensure the center tile is processed last
        else:
            return 3

    tiles.sort(key=tile_priority)

    # Print the matrix with empty space before and after
    print("\n" * 2)
    for row in matrix:
        print(' '.join(row))
    print("\n" * 2)

    return tiles


class Make_Tiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
                #"tile_width": ("INT", {"default": 512, "min": 1, "max": 10000}),
                #"tile_height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                #"overlap_x": ("INT", {"default": 0, "min": 0, "max": 10000}),
                #"overlap_y": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Steudio"

    def process(self, image, tile_calc,):
        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_width = tile_calc['tile_width']
        tile_height = tile_calc['tile_height']
        overlap_x = tile_calc['overlap_x']
        overlap_y = tile_calc['overlap_y']
        grid_n = tile_calc['grid_n']

        tile_coordinates = generate_tiles(
            image_width, image_height, tile_width, tile_height, overlap_x, overlap_y,
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


class Unmake_Tiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
                "overlap_factor": ("INT",{"default": 4, "min": 1, "max": 100}),  # Added blur blend parameter
                "blur_factor": ("INT",{"default": 20, "min": 1, "max": 100}),  # Added blur blend parameter
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Steudio"

    def process(self, images, tile_calc, overlap_factor, blur_factor,):  # Added index parameter


        # Import from Tile_calc
        upscaled_width = tile_calc['upscaled_width']
        upscaled_height = tile_calc['upscaled_height']
        overlap_x = tile_calc['overlap_x']
        overlap_y = tile_calc['overlap_y']
        grid_n = tile_calc['grid_n']

        # Import from Images
        tile_width = images.shape[2]
        tile_height = images.shape[1]

        # Overlap / factor
        f_overlap_x = overlap_x //overlap_factor
        f_overlap_y = overlap_y //overlap_factor

        # Blend factor
        blend_x = overlap_x //blur_factor
        blend_y = overlap_y //blur_factor

        tile_coordinates = generate_tiles(
            upscaled_width, upscaled_height, tile_width, tile_height, overlap_x, overlap_y
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
            # Detect corners
            if x == 0 and y == 0:
                draw.rectangle([x, y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x == upscaled_width - tile_width and y == 0:
                draw.rectangle([f_overlap_x, y, tile_width, tile_height - f_overlap_y], fill=255)
            elif x == 0 and y == upscaled_height - tile_height:
                draw.rectangle([x, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            elif x == upscaled_width - tile_width and y == upscaled_height - tile_height:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height], fill=255)
            # Detect top and bottom edges
            elif x == 0 and y !=0 and y != upscaled_height - tile_height:
                draw.rectangle([x, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x == upscaled_width - tile_width and y !=0 and y != upscaled_height - tile_height:
                 draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height - f_overlap_y], fill=255)
            # Detect left and right edges
            elif x != 0 and x !=upscaled_width - tile_width and y == 0:
                draw.rectangle([f_overlap_x, y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x != 0 and x !=upscaled_width - tile_width and y == upscaled_height - tile_height:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            # Detect not touching any edges
            elif x != 0 and x !=upscaled_width - tile_width and y !=0 and y != upscaled_height - tile_height:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)


            mask = mask.filter(ImageFilter.GaussianBlur(radius=(blend_x, blend_y)))

            mask_np = np.array(mask) / 255.0
            mask_tensor = torch.tensor(mask_np, dtype=images.dtype).unsqueeze(0).unsqueeze(-1)

            output[:, y : y + tile_height, x : x + tile_width, :] *= (1 - mask_tensor)
            output[:, y : y + tile_height, x : x + tile_width, :] += image_tile * mask_tensor

            index += 1
        return [output]

    
class Make_Tile_Calc:
    @classmethod
    def INPUT_TYPES(s):        
        Overlap_list = ["None",
                    '1/64 Tile',
                    '1/32 Tile',
                    '1/16 Tile',
                    '1/8 Tile',
                    '1/4 Tile',
                    '1/2 Tile',
                    ]
        Grid_list = ["2x2",
                    "3x3",
                    "4x4",
                    "5x5",
                    "6x6",
                    "7x7",
                    "8x8",
                    "9x9",
                    ]
        return {
            "required": {
                "tile_width": ("INT", {"forceInput": True}),
                "tile_height": ("INT", {"forceInput": True}),
                "overlap": (Overlap_list,),
                "grid_n": (Grid_list,),
            }
        }
    

    RETURN_TYPES = ("INT", "INT", "TILE_CALC",)
    RETURN_NAMES = ("upscaled_width", "upscaled_height", "tile_calc",)
    FUNCTION = "calc"
    CATEGORY = "Steudio"

    def calc(self, tile_width, tile_height, overlap, grid_n,):


        if grid_n == "2x2":
            grid_n = 2
        elif grid_n == "3x3":
            grid_n = 3
        elif grid_n == "4x4":
            grid_n = 4
        elif grid_n == "5x5":
            grid_n = 5
        elif grid_n == "6x6":
            grid_n = 6
        elif grid_n == "7x7":
            grid_n = 7
        elif grid_n == "8x8":
            grid_n = 8
        elif grid_n == "9x9":
            grid_n = 9

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


        overlap_x = int(overlap * tile_width)
        overlap_y = int(overlap * tile_height)

        upscaled_width = tile_width * grid_n - overlap_x * (grid_n - 1)
        upscaled_height = tile_height * grid_n - overlap_y * (grid_n - 1)

        effective_upscale = upscaled_width / tile_width


        tile_calc = {'upscaled_width': upscaled_width,
                    'upscaled_height': upscaled_height,
                    'tile_width': tile_width,
                    'tile_height': tile_height,
                    'overlap_x': overlap_x,
                    'overlap_y': overlap_y,
                    'effective_upscale': effective_upscale,
                    'grid_n':grid_n,
                    }

    
        return [upscaled_width, upscaled_height, tile_calc]


NODE_CLASS_MAPPINGS = {
    "Make Tiles": Make_Tiles,
    "Unmake Tiles": Unmake_Tiles,
    "Make Tile Calc": Make_Tile_Calc,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Make Tiles": "Make_Tiles",
    "Unmake Tiles": "Unmake_Tiles",
    "Make Tile Calc": "Make_Tile_Calc",
}
