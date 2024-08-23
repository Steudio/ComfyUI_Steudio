# Original nodes from https://github.com/kinfolk0117/ComfyUI_SimpleTiles modified to suit my needs
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


def order_by_center_last(tiles, image_width, image_height, tile_width, tile_height):
    # for 3x3: custom_order = [0, 2, 6, 8, 1, 3, 5, 7, 4] # First 4 corners, then the sides, then the center
    # order the tiles so they are add based on absolute distance from the center of the tile to the center of the image
    # this is done so that the center of the image is the last tile to be added, so that the center of the image is the most refined

    # get the center of the image
    center_x = image_width // 2
    center_y = image_height // 2

    # sort the tiles by distance from the center
    tiles = sorted(
        tiles,
        key=lambda tile: abs(tile[0] + tile_width // 2 - center_x)
        + abs(tile[1] + tile_height // 2 - center_y),
    )

    # reverse the order so that the center is last
    tiles = tiles[::-1]

    return tiles


def generate_tiles(
    image_width, image_height, tile_width, tile_height, overlap, offset=0
):
    tiles = []

    y = 0
    while y < image_height:
        if y == 0:
            next_y = y + tile_height - overlap + offset
        else:
            next_y = y + tile_height - overlap

        if y + tile_height >= image_height:
            y = max(image_height - tile_height, 0)
            next_y = image_height

        x = 0
        while x < image_width:
            if x == 0:
                next_x = x + tile_width - overlap + offset
            else:
                next_x = x + tile_width - overlap
            if x + tile_width >= image_width:
                x = max(image_width - tile_width, 0)
                next_x = image_width

            tiles.append((x, y))

            if next_x > image_width:
                break
            x = next_x

        if next_y > image_height:
            break
        y = next_y

    return order_by_center_last(tiles, image_width, image_height, tile_width, tile_height)


class DynamicTileSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "tile_height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "overlap": ("INT", {"default": 128, "min": 1, "max": 10000}),
                "offset": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_CALC")
    FUNCTION = "DYNAMIC_TILES"
    CATEGORY = "Steudio"

    def process(self, image, tile_width, tile_height, overlap, offset):
        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_coordinates = generate_tiles(
            image_width, image_height, tile_width, tile_height, overlap, offset
        )

        print("Tile coordinates: {}".format(tile_coordinates))

        iteration = 1

        image_tiles = []
        for tile_coordinate in tile_coordinates:
            print("Processing tile {} of {}".format(iteration, len(tile_coordinates)))
            print("Tile coordinate: {}".format(tile_coordinate))
            iteration += 1

            image_tile = image[
                :,
                tile_coordinate[1] : tile_coordinate[1] + tile_height,
                tile_coordinate[0] : tile_coordinate[0] + tile_width,
                :,
            ]

            image_tiles.append(image_tile)

        tiles_tensor = torch.stack(image_tiles).squeeze(1)
        tile_calc = (overlap, image_height, image_width, offset)

        return (tiles_tensor, tile_calc)


class DynamicTileMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "blend": ("INT", {"default": 64, "min": 0, "max": 4096}),
                "tile_calc": ("TILE_CALC",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "DYNAMIC_TILES"
    CATEGORY = "Steudio"

    def process(self, images, blend, tile_calc):
        overlap, final_height, final_width, offset = tile_calc
        tile_height = images.shape[1]
        tile_width = images.shape[2]
        print("Tile height: {}".format(tile_height))
        print("Tile width: {}".format(tile_width))
        print("Final height: {}".format(final_height))
        print("Final width: {}".format(final_width))
        print("Overlap: {}".format(overlap))

        tile_coordinates = generate_tiles(
            final_width, final_height, tile_width, tile_height, overlap, offset
        )

        print("Tile coordinates: {}".format(tile_coordinates))
        original_shape = (1, final_height, final_width, 3)
        count = torch.zeros(original_shape, dtype=images.dtype)
        output = torch.zeros(original_shape, dtype=images.dtype)

        index = 0
        iteration = 1
        for tile_coordinate in tile_coordinates:
            image_tile = images[index]
            x = tile_coordinate[0]
            y = tile_coordinate[1]

            print("Processing tile {} of {}".format(iteration, len(tile_coordinates)))
            print("Tile coordinate: {}".format(tile_coordinate))
            iteration += 1

            channels = images.shape[3]
            weight_matrix = torch.ones((tile_height, tile_width, channels))

            # blend border
            for i in range(blend):
                weight = float(i) / blend
                weight_matrix[i, :, :] *= weight  # Top rows
                weight_matrix[-(i + 1), :, :] *= weight  # Bottom rows
                weight_matrix[:, i, :] *= weight  # Left columns
                weight_matrix[:, -(i + 1), :] *= weight  # Right columns

            # We only want to blend with already processed pixels, so we keep
            # track if it has been processed.
            old_tile = output[:, y : y + tile_height, x : x + tile_width, :]
            old_tile_count = count[:, y : y + tile_height, x : x + tile_width, :]

            weight_matrix = (
                weight_matrix * (old_tile_count != 0).float()
                + (old_tile_count == 0).float()
            )

            image_tile = image_tile * weight_matrix + old_tile * (1 - weight_matrix)

            output[:, y : y + tile_height, x : x + tile_width, :] = image_tile
            count[:, y : y + tile_height, x : x + tile_width, :] = 1

            index += 1
        return [output]


NODE_CLASS_MAPPINGS = {
    "DynamicTileSplit": DynamicTileSplit,
    "DynamicTileMerge": DynamicTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicTileSplit": "DynamicTileSplit",
    "DynamicTileMerge": "DynamicTileMerge",
}
