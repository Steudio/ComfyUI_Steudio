# config.py
# AnyType code from Pythongosssss https://github.com/pythongosssss/
#
# Copy to py
# sys.path.append(os.path.dirname(__file__))
# from _config_upscale_ import OVERLAP_DICT, TILE_ORDER_DICT, SCALING_METHODS, MIN_SCALE_FACTOR_THRESHOLD

# Overlap ratios by description
OVERLAP_DICT = {
    "None": 0,
    "1/64 Tile": 0.015625,
    "1/32 Tile": 0.03125,
    "1/16 Tile": 0.0625,
    "1/8 Tile": 0.125,
    "1/4 Tile": 0.25,
    "1/2 Tile": 0.5,
}

# Tile ordering methods
TILE_ORDER_DICT = {
    "linear": 0,
    "spiral": 1
}

# Available scaling algorithms
SCALING_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos"
]

# Minimum scale threshold
MIN_SCALE_FACTOR_THRESHOLD = 1.0

# Ratio Presets
RATIO_PRESETS = {
    "auto": (1, 1),
    "1:1 ◻": (1, 1),
    "5:4 ▭": (5, 4),
    "4:3 ▭": (4, 3),
    "3:2 ▭": (3, 2),
    "16:9 ▭": (16, 9),
    "2:1 ▭": (2, 1),
    "21:9 ▭": (21, 9),
    "32:9 ▭": (32, 9),
    "": (1, 1),
    "4:5 ▯": (4, 5),
    "3:4 ▯": (3, 4),
    "2:3 ▯": (2, 3),
    "9:16 ▯": (9, 16),
    "1:2 ▯": (1, 2),
    "9:21 ▯": (9, 21),
    "9:32 ▯": (9, 32),
}

# AnyType code from Pythongosssss https://github.com/pythongosssss/
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

_any_ = AnyType("*")