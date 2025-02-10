# Inspired by https://github.com/kinfolk0117/ComfyUI_SimpleTiles
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

MIN_SCALE_FACTOR_THRESHOLD = 1.01

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


class DaC_Algorithm_No_Upscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024,}),
                "tile_height": ("INT", {"default": 1024,}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/32 Tile",}),
                "min_scale_factor": ("FLOAT", {"default": 3.0, "min":1.0, "max":8.0, }),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "spiral",}),
            }
        }


    RETURN_TYPES = ("INT", "INT", "DAC_DATA",)
    RETURN_NAMES = ("image_width", "image_height", "dac_data",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer/Advanced"
    DESCRIPTION = """
Calculate the best dimensions for upscaling an image
while maintaining minimum tile overlap and scale factor constraints.
Steudio
"""

    def execute(self, image, tile_width, tile_height, min_overlap, min_scale_factor, tile_order):

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
                upscaled_width = width * multiply_factor
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
                upscaled_height = height * multiply_factor
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
        upscaled_image = f"{upscaled_width}x{upscaled_height}"
        original_image = f"{width}x{height}"
        grid_n_xy = f"{grid_x}x{grid_y}"
        Tiles_Q = grid_x * grid_y

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

        #Display results to panel
        print("\n" * 2)
        print("Divide and Conquer Algorithm:")
        print('Original Image Size:', original_image)
        print('Upscaled Image Size:', upscaled_image)
        print(f"Grid: {grid_n_xy} ({Tiles_Q} tiles)")
        print(f"Overlap_x: {overlap_x} pixels")
        print(f"Overlap_y: {overlap_y} pixels")
        print('Effective_upscale:', effective_upscale)
        print("\n" * 2)


        return [upscaled_width, upscaled_height, dac_data]

class DaC_Algorithm:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),           
                "tile_width": ("INT", {"default": 1024,}),
                "tile_height": ("INT", {"default": 1024,}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/32 Tile",}),
                "min_scale_factor": ("FLOAT", {"default": 3.0, "min":1.0, "max":8.0, }),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "spiral",}),
            },
        }

    RETURN_TYPES = ("IMAGE", "DAC_DATA",)
    RETURN_NAMES = ("IMAGE", "dac_data",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"
    DESCRIPTION = """
Calculate the best dimensions and upscale an image
while maintaining minimum tile overlap and scale factor constraints.
Steudio
"""

    def execute(self, image, upscale_model, scaling_method, tile_width, tile_height, min_overlap, min_scale_factor, tile_order):

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
                upscaled_width = width * multiply_factor
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
                upscaled_height = height * multiply_factor
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
        upscaled_image = f"{upscaled_width}x{upscaled_height}"
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

        #Display results to panel
        print("\n" * 2)
        print("Divide and Conquer Algorithm:")
        print('Original Image Size:', original_image)
        print('Upscaled Image Size:', upscaled_image)
        print(f"Grid: {grid_n_xy} ({Tiles_Q} tiles)")
        print(f"Overlap_x: {overlap_x} pixels")
        print(f"Overlap_y: {overlap_y} pixels")
        print('Effective_upscale:', effective_upscale)
        print("\n" * 2)

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

        # Scale the image to match with algorithm results
        if upscaled_width == 0 and upscaled_height == 0:
            upscaled_image = Upscaled_with_Model
        else:
            samples = Upscaled_with_Model.movedim(-1, 1)

            if upscaled_width == 0:
                upscaled_width = max(1, round(samples.shape[3] * upscaled_height / samples.shape[2]))
            elif upscaled_height == 0:
                upscaled_height = max(1, round(samples.shape[2] * upscaled_width / samples.shape[3]))

            upscaled_image = comfy.utils.common_upscale(samples, upscaled_width, upscaled_height, scaling_method, crop=0)
            upscaled_image = upscaled_image.movedim(1, -1)

        return (upscaled_image, dac_data)

class Divide_Image_Select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
                "position": ("INT", { "default": 0, "min": 0, "step": 1, }),
                # "position": ("INT", { "default": 0, "min": 0, "step": 1, "forceInput": True }),
            },
        }


    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("SELECTED TILE", "ALL TILES",)
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"

    def execute(self, image, position, dac_data,):
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

        all_tiles = torch.cat(image_tiles, dim=0)
        selected_tile = image_tiles[position]

        return (selected_tile, [all_tiles[i].unsqueeze(0) for i in range(all_tiles.shape[0])])

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
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer/Advanced"

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

        all_tiles = torch.cat(image_tiles, dim=0)

        return ([all_tiles[i].unsqueeze(0) for i in range(all_tiles.shape[0])],)


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
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"

    def execute(self, images, dac_data,):

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


        # # Calculate the square root of overlap_x
        # blend_x = overlap_x //blur_factor
        # blend_y = overlap_y //blur_factor

        # Blend factor
        blend_x = math.sqrt(overlap_x)
        blend_y = math.sqrt(overlap_y)


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
    

# Original LoadImagesFromFolderKJ from https://github.com/kijai/ComfyUI-KJNodes
class Load_Images_into_List:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "Load_Images_into_List"
    CATEGORY = "Steudio/Utils"

    def Load_Images_into_List(self, directory,):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        images = []

        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            images.append(image)

        images = torch.cat(images, dim=0)
        return ([images[i].unsqueeze(0) for i in range(images.shape[0])],)



NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Divide and Conquer Algorithm (No Upscale)": DaC_Algorithm_No_Upscale,
    "Divide Image": Divide_Image,
    "Combine Tiles": Combine_Tiles,
    "Divide Image and Select Tile": Divide_Image_Select,
    "Load Images into List": Load_Images_into_List,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Divide and Conquer Algorithm (No Upscale)": "Divide and Conquer Algorithm (No Upscale)",
    "Divide Image": "Divide Image",
    "Combine Tiles": "Combine Tiles",
    "Divide Image and Select Tile": "Divide Image and Select Tile",
    "Load Images into List": "Load Images into List",
}
