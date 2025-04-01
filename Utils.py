# Make_size utilize code from Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# Range_List utilize code from Cubiq    https://github.com/cubiq/ComfyUI_essentials
# Modified by Steudio

class Make_Size:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
    
        Resolutions_list = ["custom",
                         "SD1.5 - 1:1 ◻ 512x512",
                         "SD1.5 - 2:3 ▯ 512x768",
                         "SD1.5 - 3:4 ▯ 512x682",
                         "SD1.5 - 3:2 ▭ 768x512",
                         "SD1.5 - 4:3 ▭ 682x512",
                         "SD1.5 - 16:9 ▭ 910x512",
                         "SD1.5 - 1.85:1 ▭ 952x512",
                         "SD1.5 - 2:1 ▭ 1024x512",
                         "SDXL - 1:1 ◻ 1024x1024",
                         "SDXL - 3:4 ▯ 896x1152",
                         "SDXL - 5:8 ▯ 832x1216",
                         "SDXL - 9:16 ▯ 768x1344",
                         "SDXL - 9:21 ▯ 640x1536",
                         "SDXL - 4:3 ▭ 1152x896",
                         "SDXL - 3:2 ▭ 1216x832",
                         "SDXL - 16:9 ▭ 1344x768",
                         "SDXL ! 2:1 ▭ 1408x704",
                         "SDXL - 21:9 ▭ 1536x640",
                         "SDXL ! 32:9 ▭ 1792x512"]
               
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "resolutions": (Resolutions_list,),
                "swap_dimensions": (["Off", "On"],),
            }
        }
    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("width", "height", )
    

    FUNCTION = "Resolutions"
    CATEGORY = "Steudio/Utils"

    def Resolutions(self, width, height, resolutions, swap_dimensions):
        
        # SD1.5
        if resolutions == "SD1.5 - 1:1 ◻ 512x512":
            width, height = 512, 512
        elif resolutions == "SD1.5 - 2:3 ▯ 512x768":
            width, height = 512, 768
        elif resolutions == "SD1.5 - 16:9 ▭ 910x512":
            width, height = 910, 512
        elif resolutions == "SD1.5 - 3:4 ▯ 512x682":
            width, height = 512, 682
        elif resolutions == "SD1.5 - 3:2 ▭ 768x512":
            width, height = 768, 512    
        elif resolutions == "SD1.5 - 4:3 ▭ 682x512":
            width, height = 682, 512
        elif resolutions == "SD1.5 - 1.85:1 ▭ 952x512":            
            width, height = 952, 512
        elif resolutions == "SD1.5 - 2:1 ▭ 1024x512":
            width, height = 1024, 512
        elif resolutions == "SD1.5 - 2.39:1 ▭ 1224x512":
            width, height = 1224, 512 
        # SDXL   
        if resolutions == "SDXL - 1:1 ◻ 1024x1024":
            width, height = 1024, 1024
        elif resolutions == "SDXL - 3:4 ▯ 896x1152":
            width, height = 896, 1152
        elif resolutions == "SDXL - 5:8 ▯ 832x1216":
            width, height = 832, 1216
        elif resolutions == "SDXL - 9:16 ▯ 768x1344":
            width, height = 768, 1344
        elif resolutions == "SDXL - 9:21 ▯ 640x1536":
            width, height = 640, 1536
        elif resolutions == "SDXL - 4:3 ▭ 1152x896":
            width, height = 1152, 896
        elif resolutions == "SDXL - 3:2 ▭ 1216x832":
            width, height = 1216, 832
        elif resolutions == "SDXL - 16:9 ▭ 1344x768":
            width, height = 1344, 768
        elif resolutions == "SDXL ! 2:1 ▭ 1408x704":
            width, height = 1408, 704
        elif resolutions == "SDXL - 21:9 ▭ 1536x640":
            width, height = 1536, 640
        elif resolutions == "SDXL ! 32:9 ▭ 1792x512":
            width, height = 1792, 512                    
        if swap_dimensions == "On":
            width, height = height, width
        
        width = int(width)
        height = int(height)
           
        return(width, height, )
    
class Flux_Size:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        Resolutions_list = [
            "1:1 ◻ 512x512 | 0.3",
            "1:1 ◻ 768x768 | 0.6",
            "1:1 ◻ 1024x1024 | 1.0",
            "1:1 ◻ 1152x1152 | 1.3",
            "5:4 ▭ 640x512 | 0.3",
            "5:4 ▭ 960x768 | 0.7",
            "5:4 ▭ 1280x1024 | 1.3",
            "4:3 ▭ 768x576 | 0.4",
            "4:3 ▭ 1024x768 | 0.8",
            "4:3 ▭ 1280x960 | 1.2",
            "4:3 ▭ 1536x1152 | 1.8",
            "3:2 ▭ 576x384 | 0.2",
            "3:2 ▭ 768x512 | 0.4",
            "3:2 ▭ 960x640 | 0.6",
            "3:2 ▭ 1152x768 | 0.9",
            "3:2 ▭ 1344x896 | 1.2",
            "3:2 ▭ 1536x1024 | 1.6",
            "16:9 ▭ 1024x576 | 0.6",
            "16!9 ▭ 1344x768 | 1.0",
            "16!9 ▭ 1472x832 | 1.2",
            "16!9 ▭ 1600x896 | 1.4",
            "2:1 ▭ 768x384 | 0.3",
            "2:1 ▭ 1024x512 | 0.5",
            "2:1 ▭ 1280x640 | 0.8",
            "2:1 ▭ 1536x768 | 1.2",
            "21:9 ▭ 896x384 | 0.3",
            "21:9 ▭ 1344x576 | 0.8",
            "21:9 ▭ 1792x768 | 1.4",
            "32!9 ▭ 896x256 | 0.2",
            "32!9 ▭ 1152x320 | 0.4",
            "32!9 ▭ 1344x384 | 0.5",
            "32!9 ▭ 1600x448 | 0.7",
            "32!9 ▭ 1792x512 | 0.9",
            "32:9 ▭ 2048x576 | 1.2",
        ]

        return {
            "required": {
                "res": (Resolutions_list,),
                "Orientation": (["▭", "▯"],),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)

    FUNCTION = "res"
    CATEGORY = "Steudio/Utils"

    def res(self, res, Orientation):
        # Updated dictionary mapping resolutions to width and height
        resolutions = {
            "1:1 ◻ 512x512 | 0.3": (512, 512),
            "1:1 ◻ 768x768 | 0.6": (768, 768),
            "1:1 ◻ 1024x1024 | 1.0": (1024, 1024),
            "1:1 ◻ 1152x1152 | 1.3": (1152, 1152),
            "5:4 ▭ 640x512 | 0.3": (640, 512),
            "5:4 ▭ 960x768 | 0.7": (960, 768),
            "5:4 ▭ 1280x1024 | 1.3": (1280, 1024),
            "4:3 ▭ 768x576 | 0.4": (768, 576),
            "4:3 ▭ 1024x768 | 0.8": (1024, 768),
            "4:3 ▭ 1280x960 | 1.2": (1280, 960),
            "4:3 ▭ 1536x1152 | 1.8": (1536, 1152),
            "3:2 ▭ 576x384 | 0.2": (576, 384),
            "3:2 ▭ 768x512 | 0.4": (768, 512),
            "3:2 ▭ 960x640 | 0.6": (960, 640),
            "3:2 ▭ 1152x768 | 0.9": (1152, 768),
            "3:2 ▭ 1344x896 | 1.2": (1344, 896),
            "3:2 ▭ 1536x1024 | 1.6": (1536, 1024),
            "16:9 ▭ 1024x576 | 0.6": (1024, 576),
            "16!9 ▭ 1344x768 | 1.0": (1344, 768),
            "16!9 ▭ 1472x832 | 1.2": (1472, 832),
            "16!9 ▭ 1600x896 | 1.4": (1600, 896),
            "2:1 ▭ 768x384 | 0.3": (768, 384),
            "2:1 ▭ 1024x512 | 0.5": (1024, 512),
            "2:1 ▭ 1280x640 | 0.8": (1280, 640),
            "2:1 ▭ 1536x768 | 1.2": (1536, 768),
            "21:9 ▭ 896x384 | 0.3": (896, 384),
            "21:9 ▭ 1344x576 | 0.8": (1344, 576),
            "21:9 ▭ 1792x768 | 1.4": (1792, 768),
            "32!9 ▭ 896x256 | 0.2": (896, 256),
            "32!9 ▭ 1152x320 | 0.4": (1152, 320),
            "32!9 ▭ 1344x384 | 0.5": (1344, 384),
            "32!9 ▭ 1600x448 | 0.7": (1600, 448),
            "32!9 ▭ 1792x512 | 0.9": (1792, 512),
            "32:9 ▭ 2048x576 | 1.2": (2048, 576),
        }

        # Retrieve width and height from the updated dictionary
        width, height = resolutions.get(res, (0, 0))  # Default to (0, 0) if resolution not found

        # Adjust for orientation
        if Orientation == "▯":
            width, height = height, width

        return int(width), int(height),

class Aspect_Ratio_Size:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        Resolutions_list = [
            "1:1 ◻",
            "5:4 ▭",
            "4:3 ▭",
            "3:2 ▭",
            "16:9 ▭",
            "2:1 ▭",
            "21:9 ▭",
            "32:9 ▭",
        ]

        return {
            "required": {
                "ratio": (Resolutions_list,),
                "Orientation": (["▭", "▯"],),
                "Megapixel": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 3.00, "step": 0.01 }),
                "Precision": ("FLOAT", {"default": 0.00, "min": 0.30, "max": 1.00, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "DATA",)
    RETURN_NAMES = ("width", "height", "data",)
    FUNCTION = "calculate_dimensions"
    CATEGORY = "Steudio/Utils"

    def calculate_dimensions(self, ratio, Orientation, Megapixel, Precision):
        # Dictionary mapping resolutions to width and height
        resolutions = {
            "1:1 ◻": (1, 1),
            "5:4 ▭": (5, 4),
            "4:3 ▭": (4, 3),
            "3:2 ▭": (3, 2),
            "16:9 ▭": (16, 9),
            "2:1 ▭": (2, 1),
            "21:9 ▭": (21, 9),
            "32:9 ▭": (32, 9),
        }

        # Retrieve aspect width and height from the resolutions dictionary
        aspect_width, aspect_height = resolutions.get(ratio, (0, 0))  # Default to (0, 0) if ratio not found

        # Convert megapixels to total pixels
        total_pixels = int(Megapixel * 1_000_000)

        # Calculate approximate starting dimensions
        width = int((total_pixels * (aspect_width / aspect_height)) ** 0.5)
        height = int(width * (aspect_height / aspect_width))

        # Adjust width and height to multiples of 64 while respecting precision
        found_solution = False
        while True:
            # Ensure dimensions are multiples of 64
            width = (width // 64) * 64
            height = (height // 64) * 64

            # Check precision
            if abs((width / height) - (aspect_width / aspect_height)) <= Precision:
                found_solution = True
                break

            # Try reducing dimensions
            if width > 64 and height > 64:
                if (width / height) > (aspect_width / aspect_height):
                    width -= 64
                else:
                    height -= 64
            else:
                # No solution found
                break

        f_megapixel = "{:,}".format(width * height) 
        f_precision = (aspect_width / aspect_height) - (width / height)

        # Adjust for orientation
        if Orientation == "▯":
            width, height = height, width

        data = {'width': width,
            'height': height,
            'Megapixel': f_megapixel,
            'Precision': f_precision,
            }



        return int(width), int(height), data


class Seed_Shifter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"forceInput":True}),
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
    def shift_seeds(self, seed, seed_shifter, batch):
        seeds = [(seed + seed_shifter + i) for i in range(batch)]
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

        return (list(map(int, result)), list(map(float, [f"{num:.2f}"for num in result if isinstance(num, float)])), )





# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Make Size": Make_Size,
    "Flux Size": Flux_Size,
    "Aspect Ratio Size": Aspect_Ratio_Size,
    "Seed Shifter": Seed_Shifter,
    "Sequence Generator": Sequence_Generator,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Make Size": "Make Size",
    "Flux Size": "Flux Size",
    "Aspect Ratio Size": "Aspect Ratio Size",
    "Seed Shifter": "Seed Shifter",
    "Sequence Generator": "Sequence Generator",
}