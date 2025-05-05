# Ratio_to_Size utilize code from Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# AnyType code from Pythongosssss https://github.com/pythongosssss/
# Sequence_Generator utilize code from Cubiq    https://github.com/cubiq/ComfyUI_essentials
# Original LoadImagesFromFolderKJ from https://github.com/kijai/ComfyUI-KJNodes
# Original display_any code from https://github.com/rgthree/rgthree-comfy
# Created by Steudio

import comfy.samplers
import math
import os
import torch
import math
from PIL import Image, ImageOps
import numpy as np


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

_any_ = AnyType("*")

RATIO = {
    "1:1 ◻": (1, 1),
    "5:4 ▭": (5, 4),
    "4:3 ▭": (4, 3),
    "3:2 ▭": (3, 2),
    "16:9 ▭": (16, 9),
    "2:1 ▭": (2, 1),
    "21:9 ▭": (21, 9),
    "32:9 ▭": (32, 9),
    "": (1, 1),
    "4:5 ▯": (4, 5),
    "3:4 ▯": (3, 4),
    "2:3 ▯": (2, 3),
    "9:16 ▯": (9, 16),
    "1:2 ▯": (1, 2),
    "9:21 ▯": (9, 21),
    "9:32 ▯": (9, 32),
}

class Ratio_Calculator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):  
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = (_any_,)
    RETURN_NAMES = ("ratio",)
    FUNCTION = "calc"
    OUTPUT_NODE = True
    CATEGORY = "Steudio/Utils"

    def calc(self, image):      

        # Get dimensions of the image
        _, height, width, _ = image.shape

        # Find the greatest common divisor (GCD)
        gcd = math.gcd(width, height)
        
        # Simplify the dimensions
        simplified_width = width // gcd
        simplified_height = height // gcd

        # Find the closest ratio
        closest_ratio = None
        min_difference = float('inf')
        for name, (rw, rh) in RATIO.items():
            difference = abs(simplified_width / simplified_height - rw / rh)
            if difference < min_difference:
                min_difference = difference
                closest_ratio = name

        # return closest_ratio,
        return {"ui": {"text": closest_ratio}, "result": (closest_ratio,)}

class Ratio_to_Size:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (list(RATIO.keys()),),
                "Megapixel": ("FLOAT", {"default": 1.05, "min": 0.10, "max": 3.00, "step": 0.01 }),
                "Precision": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.00, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "UI",)
    RETURN_NAMES = ("width", "height", "ui",)
    FUNCTION = "calculate_dimensions"
    CATEGORY = "Steudio/Utils"

    def calculate_dimensions(self, ratio, Megapixel, Precision):

        # Retrieve aspect width and height from the resolutions dictionary
        aspect_width, aspect_height = RATIO.get(ratio, (1, 1))  # Default to (1, 1) if ratio not found

        # Convert megapixels to total pixels
        total_pixels = int(Megapixel * 1_000_000)

        # Calculate approximate starting dimensions
        width = int((total_pixels * (aspect_width / aspect_height)) ** 0.5)
        height = int(width * (aspect_height / aspect_width))

        # Adjust width and height to multiples of 64 while respecting precision
        while True:
            # Ensure dimensions are multiples of 64
            width = (width // 64) * 64
            height = (height // 64) * 64

            # Check precision
            if abs((width / height) - (aspect_width / aspect_height)) <= Precision:
                break

            # Try reducing dimensions
            if width > 64 and height > 64:
                if (width / height) > (aspect_width / aspect_height):
                    width -= 64
                else:
                    height -= 64
            else:
                break

        f_megapixel = "{:,}".format(width * height) 
        f_precision = round((aspect_width / aspect_height) - (width / height), 2)

        ui = f"Ratio: {ratio}\nWidth: {width}\nHeight: {height}\nMegapixel: {f_megapixel}\nPrecision: {f_precision}\n"


        return int(width), int(height), ui


class Seed_Shifter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_": ("INT", { "default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1 }),
                "seed_shifter": ("INT", {"default": 0, "min": 0}),
                "batch": ("INT", {"default": 1, "min": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seeds",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "shift_seeds"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
A simple and effective way to generate a “batch” of images with reproducible seed.
Steudio
"""
    def shift_seeds(self, seed_, seed_shifter, batch):
        seeds = [(seed_ + seed_shifter + i) for i in range(batch)]
        return seeds,

class Sequence_Generator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "0...1+0.1"}),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", )
    OUTPUT_IS_LIST = (True,True)
    OUTPUT_NODE = True
    FUNCTION = "Execute"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
x...y+z | Generates a sequence of numbers from x to y with a step of z.
x...y#z | Generates z evenly spaced numbers between x and y.
  x,y,z | Generates a list of x, y, z.
    """

    def Execute(self, gen):
        elements = gen.split(',')
        result = []

        def parse_number(s):
            try:
                return float(s)
            except ValueError:
                return 0.0

        for element in elements:
            element = element.strip()

            if '...' in element:
                if '#' in element:
                    start, rest = element.split('...')
                    end, num_items = rest.split('#')
                    start = parse_number(start)
                    end = parse_number(end)
                    num_items = int(parse_number(num_items))
                    if num_items == 1:
                        result.append(round(start, 2))
                    else:
                        step = (end - start) / (num_items - 1)
                        for i in range(num_items):
                            result.append(round(start + i * step, 2))
                else:
                    start, rest = element.split('...')
                    end, step = rest.split('+')
                    start = parse_number(start)
                    end = parse_number(end)
                    step = abs(parse_number(step))
                    current = start
                    if start > end:
                        step = -step
                    while (step > 0 and current <= end) or (step < 0 and current >= end):
                        result.append(round(current, 2))
                        current += step
            else:
                result.append(round(parse_number(element), 2))

        seq_int = list(map(int, result))
        seq_float = list(map(float, [f"{num:.2f}"for num in result if isinstance(num, float)]))
        seq_int_float = f"{len(seq_int)} INT: {seq_int}\n{len(seq_float)} FLOAT: {seq_float}"

        return {"ui": {"text": (seq_int_float)}, "result": (seq_int, seq_float)}


class Simple_Config:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):               
        return {
            "required": {
                "steps": ("INT", {"default": 24, "min": 1, "max": 99}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            }
        }
    RETURN_TYPES = ("INT",comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS )
    RETURN_NAMES = ("STEPS", "SAMPLER", "SCHEDULER")
    

    FUNCTION = "config"
    CATEGORY = "Steudio/Utils"

    def config(self, steps, sampler, scheduler,):           
        return(steps, sampler, scheduler,)
    


# Original display_any code from https://github.com/rgthree/rgthree-comfy
class Display_UI:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name, missing-function-docstring
        return {
            "required": {
                "ui": (_any_, {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "Steudio/Utils"

    def main(self, ui=None):
        value = 'None'
        if isinstance(ui, str):
            value = ui
        elif isinstance(ui, (int, float, bool)):
            value = str(ui)

        return {"ui": {"text": (value,)}}



# Original LoadImagesFromFolderKJ code from https://github.com/kijai/ComfyUI-KJNodes
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

    def Load_Images_into_List(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        # List all files in the directory
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        if not dir_files:
            raise FileNotFoundError(f"No valid image files found in directory '{directory}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        images = []

        for image_path in dir_files:
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image)[None,]
                images.append(image_tensor)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        if not images:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        # Concatenate images into a single tensor
        images = torch.cat(images, dim=0)
        return ([images[i].unsqueeze(0) for i in range(images.shape[0])],)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Ratio Calculator": Ratio_Calculator,
    "Ratio to Size": Ratio_to_Size,
    "Seed Shifter": Seed_Shifter,
    "Sequence Generator": Sequence_Generator,
    "Simple Config": Simple_Config,
    "Load Images into List": Load_Images_into_List,
    "Display UI": Display_UI,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Ratio Calculator": "Ratio Calculator",
    "Ratio to Size": "Ratio to Size",
    "Seed Shifter": "Seed Shifter",
    "Sequence Generator": "Sequence Generator",
    "Simple Config": "Simple Config",
    "Load Images into List": "Load Images into List",
    "Display UI": "Display UI",
}