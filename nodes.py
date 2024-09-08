#from ComfyUI_Steudio.standard import Make_Tiles, Unmake_Tiles, Make_Tiles_Math
from ComfyUI_Steudio.MakeTiles import Make_Tiles, Unmake_Tiles, Make_Tiles_Math
from ComfyUI_Steudio.MakeSize import Make_Size_Latent
from ComfyUI_Steudio.MakeSize import Make_Size

NODE_CLASS_MAPPINGS = {
    "Make Tiles Math": Make_Tiles_Math,
    "Make Tiles": Make_Tiles,
    "Unmake Tiles": Unmake_Tiles,
    "Make Size": Make_Size,
    "Make Size Latent": Make_Size_Latent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Make Tiles Math": "Make_Tiles_Math",
    "Make Tiles": "Make_Tiles",
    "Unmake Tiles": "Unmake_Tiles",
    "Make Size": "Make_Size",
    "Make Size Latent": "Make_Size_Latent",
}