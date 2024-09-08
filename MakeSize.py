# Original nodes from Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# Modified by Steudio

import torch

class Make_Size_Latent:
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
    RETURN_TYPES = ("INT", "INT", "LATENT", )
    RETURN_NAMES = ("width", "height", "empty_latent", )
    

    FUNCTION = "Resolutions"
    CATEGORY = "Steudio"

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
        
        latent = torch.zeros([1, 4, height // 8, width // 8])
           
        return(width, height, {"samples":latent}, )


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
    CATEGORY = "Steudio"

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



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Make Size Latent": Make_Size_Latent,
    "Make Size": Make_Size,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Make Size Latent": "Make_Size_Latent",
    "Make Size": "Make_Size",
}