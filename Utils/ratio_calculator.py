import math
import os, sys
sys.path.append(os.path.dirname(__file__))
from _config_utils_ import RATIO_PRESETS, _any_

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

        # Calculate megapixel
        f_megapixel = "{:,} pixels".format(width * height)

        # Find the closest ratio (skip "auto" placeholder)
        closest_ratio = None
        min_difference = float('inf')
        for name, (rw, rh) in RATIO_PRESETS.items():
            if name in ("auto", ""):
                continue
            difference = abs(simplified_width / simplified_height - rw / rh)
            if difference < min_difference:
                min_difference = difference
                closest_ratio = name
                closest_rw, closest_rh = rw, rh

        # Precision difference
        f_precision = round((closest_rw / closest_rh) - (simplified_width / simplified_height), 2)

        # ✅ UI now shows the matched preset name, never "auto"
        return {
            "ui": {
                "text": f"{closest_ratio}\nWidth: {width}\nHeight: {height}\n{f_megapixel}\nPrecision: {f_precision}"
            },
            "result": (closest_ratio,)
        }
    

NODE_CLASS_MAPPINGS = {
    "Ratio Calculator": Ratio_Calculator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Ratio Calculator": "Ratio Calculator",
}