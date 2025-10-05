# seed_shifter.py

class Seed_Shifter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_": ("INT", { "default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1 }),
                "seed_shifter": ("INT", {"default": 0, "min": 0}),
                "batch": ("INT", {"default": 1, "min": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seeds",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "shift_seeds"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
A simple and effective way to generate a “batch” of images with reproducible seed.
Steudio
"""
    def shift_seeds(self, seed_, seed_shifter, batch):
        seeds = [(seed_ + seed_shifter + i) for i in range(batch)]
        return seeds,


NODE_CLASS_MAPPINGS = {"Seed Shifter": Seed_Shifter,}
NODE_DISPLAY_NAME_MAPPINGS = {"Seed Shifter": "Seed Shifter",}
