#from ComfyUI_Steudio.standard import DaC_Algorithm, Divide_Image, Combine_Tiles, Make_Size
from ComfyUI_Steudio.DaC import DaC_Algorithm, Divide_Image, Combine_Tiles
from ComfyUI_Steudio.MakeSize import Make_Size

NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Divide Image": Divide_Image,
    "Combine Tiles": Combine_Tiles,
    "Make Size": Make_Size,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Divide Image": "Divide Image",
    "Combine Tiles": "Combine Tiles",
    "Make Size": "Make_Size",
}