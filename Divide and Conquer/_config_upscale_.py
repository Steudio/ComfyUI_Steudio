# _config_upscale_.py

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

# ANY
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

_any_ = AnyType("*")