from .DaC import DaC_Algorithm, DaC_Algorithm_No_Upscale, Divide_Image, Combine_Tiles, Divide_Image_Select, Load_Images_into_List
from .Utils import Make_Size, Flux_Size, Aspect_Ratio_Size, Seed_Shifter, Sequence_Generator


NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Divide and Conquer Algorithm (No Upscale)": DaC_Algorithm_No_Upscale,
    "Divide Image": Divide_Image,
    "Combine Tiles": Combine_Tiles,
    "Divide Image and Select Tile": Divide_Image_Select,
    "Make Size": Make_Size,
    "Flux Size": Flux_Size,
    "Aspect Ratio Size": Aspect_Ratio_Size,
    "Seed Shifter": Seed_Shifter,
    "Sequence Generator": Sequence_Generator,
    "Load Images into List": Load_Images_into_List,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Divide and Conquer Algorithm (No Upscale)": "Divide and Conquer Algorithm (No Upscale)",
    "Divide Image": "Divide Image",
    "Combine Tiles": "Combine Tiles",
    "Divide Image and Select Tile": "Divide Image and Select Tile",
    "Make Size": "Make_Size",
    "Flux Size": "Flux Size",
    "Aspect Ratio Size": "Aspect Ratio Size",
    "Seed Shifter": "Seed Shifter",
    "Sequence Generator": "Sequence Generator",
    "Load Images into List": "Load Images into List",
}