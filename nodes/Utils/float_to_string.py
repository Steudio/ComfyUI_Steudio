class FloatToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "Steudio/Utils"

    def convert(self, value):
        # Format to 2 decimal places
        return (f"{value:.2f}",)


class StringToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "1.00"}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = "Steudio/Utils"

    def convert(self, text):
        try:
            f = float(text)
            # Clamp to [0.0, 1.0]
            f = max(0.0, min(1.0, f))
            return (f,)
        except ValueError:
            return (0.0,)


NODE_CLASS_MAPPINGS = {
    "Float To String": FloatToString,
    "String To Float": StringToFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Float To String": "Float to String",
    "String To Float": "String to Float",
}