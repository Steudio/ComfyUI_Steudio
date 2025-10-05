# divide_image_select.py

import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from _utils_ import create_tile_coordinates, generate_matrix_ui

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

        # Prepare a UI text representation of the tile order matrix.
        matrix_ui = generate_matrix_ui(matrix)


        return ([tile_or_tiles[i].unsqueeze(0) for i in range(tile_or_tiles.shape[0])], matrix_ui)


NODE_CLASS_MAPPINGS = {"Divide Image and Select Tile": Divide_Image_Select,}
NODE_DISPLAY_NAME_MAPPINGS = {"Divide Image and Select Tile": "Divide Image and Select Tile",}
