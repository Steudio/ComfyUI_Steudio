# display_ui.py 
# Original display_any code from https://github.com/rgthree/rgthree-comfy
# AnyType code from Pythongosssss https://github.com/pythongosssss/

import os
import sys
import torch
import math
import numpy as np

sys.path.append(os.path.dirname(__file__))
from _config_utils_ import _any_

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


NODE_CLASS_MAPPINGS = {
    "Display UI": Display_UI,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Display UI": "Display UI",
}
