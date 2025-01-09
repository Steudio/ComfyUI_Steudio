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

class Range_List:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Integer", "Float"], {"default": "Integer"}),
                "any": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("List",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "Execute"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
x...y+z | Generates a sequence of numbers from x to y with a step of z.
x...y#z | Generates z evenly spaced numbers between x and y.
x,y,z | Generates a list of x, y, z.
    """

    def Execute(self, mode, any):
        elements = any.split(',')
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

            if mode == "Integer":
                result = list(map(int, result))
            elif mode == "Float":
                result = list(map(float, [f"{num:.2f}"for num in result if isinstance(num, float)]))

        return result,





# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Make Size": Make_Size,
    "Seed Shifter": Seed_Shifter,
    "Range List": Range_List,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Make Size": "Make Size",
    "Seed Shifter": "Seed Shifter",
    "Range List": "Range List",
}