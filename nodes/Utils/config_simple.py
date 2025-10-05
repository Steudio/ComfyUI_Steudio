import comfy.samplers

class Config_Simple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 24, "min": 1, "max": 99}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 99}),
            }
        }

    RETURN_TYPES = ("INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "FLOAT")
    RETURN_NAMES = ("STEPS", "SAMPLER", "SCHEDULER", "CFG")
    FUNCTION = "config"
    CATEGORY = "Steudio/Utils"

    def config(self, steps, sampler, scheduler, cfg):
        return (steps, sampler, scheduler, cfg)

NODE_CLASS_MAPPINGS = {"Config Simple": Config_Simple}
NODE_DISPLAY_NAME_MAPPINGS = {"Config Simple": "Config Simple"}
