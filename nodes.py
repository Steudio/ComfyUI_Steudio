from ComfyUI_SimpleTiles.standard import TileSplit, TileMerge, TileCalc
from ComfyUI_SimpleTiles.dynamic import DynamicTileSplit, DynamicTileMerge

NODE_CLASS_MAPPINGS = {
    "TileSplit": TileSplit,
    "TileMerge": TileMerge,
    "TileCalc": TileCalc,
    "DynamicTileSplit": DynamicTileSplit,
    "DynamicTileMerge": DynamicTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileSplit": "TileSplit (Legacy)",
    "TileMerge": "TileMerge (Legacy)",
    "TileCalc": "TileCalc (Legacy)",
    "DynamicTileSplit": "TileSplit (Dynamic)",
    "DynamicTileMerge": "TileMerge (Dynamic)",
}
