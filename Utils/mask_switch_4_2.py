class mask_switch_4_2:
    # Central dictionary of mask keys (values can be anything, we only use the keys for the dropdown)
    MASK_PRESETS = {
        "inpaint": None,
        "outpaint": None,
        "context": None,
        "white": None
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Dropdowns populated from MASK_PRESETS keys
                "mask_A": (list(cls.MASK_PRESETS.keys()),),
                "mask_B": (list(cls.MASK_PRESETS.keys()),),
            },
            "optional": {
                # Optional inputs named after the keys
                key: ("MASK",) for key in cls.MASK_PRESETS.keys()
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask_A", "mask_B")
    FUNCTION = "select"
    CATEGORY = "Steudio/Utils"
    OUTPUT_IS_LIST = (False, False)
    OUTPUT_NODE = False

    def select(self, mask_A, mask_B, **kwargs):
        def fetch_mask(name):
            mask = kwargs.get(name)
            if mask is None:
                raise ValueError(f"⚠️ Mask input '{name}' was not provided.")
            return mask

        output_A = fetch_mask(mask_A)
        output_B = fetch_mask(mask_B)
        return (output_A, output_B)


NODE_CLASS_MAPPINGS = {
    "mask_switch_4_2": mask_switch_4_2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mask_switch_4_2": "Mask Switch 4>2",
}