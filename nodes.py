from ComfyUI_Steudio.DaC import DaC_Algorithm, DaC_Algorithm_No_Upscale, Divide_Image, Combine_Tiles, Divide_Image_Select
from ComfyUI_Steudio.MakeSize import Make_Size


NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Divide and Conquer Algorithm (No Upscale)": DaC_Algorithm_No_Upscale,
    "Divide Image": Divide_Image,
    "Combine Tiles": Combine_Tiles,
    "Divide Image and Select Tile": Divide_Image_Select,
    "Make Size": Make_Size,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Divide and Conquer Algorithm (No Upscale)": "Divide and Conquer Algorithm (No Upscale)",
    "Divide Image": "Divide Image",
    "Combine Tiles": "Combine Tiles",
    "Divide Image and Select Tile": "Divide Image and Select Tile",
    "Make Size": "Make_Size",
}