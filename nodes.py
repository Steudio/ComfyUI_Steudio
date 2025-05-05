from .DaC import DaC_Algorithm, Combine_Tiles, Divide_Image_Select
from .Utils import Ratio_to_Size, Seed_Shifter, Sequence_Generator, Simple_Config, Display_UI, Ratio_Calculator, Load_Images_into_List


WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Combine Tiles": Combine_Tiles,
    "Divide Image and Select Tile": Divide_Image_Select,
    "Ratio Calculator": Ratio_Calculator,
    "Ratio to Size": Ratio_to_Size,
    "Seed Shifter": Seed_Shifter,
    "Sequence Generator": Sequence_Generator,
    "Load Images into List": Load_Images_into_List,
    "Simple Config": Simple_Config,
    "Display UI": Display_UI,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Combine Tiles": "Combine Tiles",
    "Divide Image and Select Tile": "Divide Image and Select Tile",
    "Ratio Calculator": "Ratio Calculator",
    "Ratio to Size": "Ratio to Size",
    "Seed Shifter": "Seed Shifter",
    "Sequence Generator": "Sequence Generator",
    "Load Images into List": "Load Images into List",
    "Simple Config": "Simple Config",
    "Display UI": "Display UI",
}