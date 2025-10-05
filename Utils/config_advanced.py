import comfy.samplers

# Define your preset combos
COMBO_PRESETS = {
    "euler | simple": ("euler", "simple"),
    "res_multistep | beta": ("res_multistep", "beta"),
    "deis | beta": ("deis", "beta"),
    "seeds_2 | beta": ("seeds_2", "beta"),
    "ddim | ddim_uniform": ("ddim", "ddim_uniform"),
}

# Define quality presets and steps per combo
# You can set different steps for each combo/quality pair
STEPS_PRESETS = {
    "Fast (4)": {
        "euler | simple": 4, # slow but fine
        "res_multistep | beta": 4,
        "deis | beta": 4, # unusable
        "seeds_2 | beta": 4, # slow but fine
        "ddim | ddim_uniform": 4, # fine
    },
    "Fast (8)": {
        "euler | simple": 8, # slow but fine
        "res_multistep | beta": 8,
        "deis | beta": 8, # unusable
        "seeds_2 | beta": 8, # slow but fine
        "ddim | ddim_uniform": 8, # fine
    },
    "Balanced (16-24)": {
        "euler | simple": 16,
        "res_multistep | beta": 20,
        "deis | beta": 20,
        "seeds_2 | beta": 18,
        "ddim | ddim_uniform": 24,
    },
    "Enhanced (22-30)": {
        "euler | simple": 22,
        "res_multistep | beta": 26,
        "deis | beta": 26,
        "seeds_2 | beta": 24,
        "ddim | ddim_uniform": 30,
    },
    "High Quality (28-36)": {
        "euler | simple": 28,
        "res_multistep | beta": 32,
        "deis | beta": 32,
        "seeds_2 | beta": 30,
        "ddim | ddim_uniform": 36,
    },
}

class Config_Advanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combo": (list(COMBO_PRESETS.keys()),{"default": "euler | simple"}),
                "quality": (list(STEPS_PRESETS.keys()),{"default": "Balanced (16-24)"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 99}),
            }
        }

    RETURN_TYPES = ("INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "FLOAT", "STRING",)
    RETURN_NAMES = ("STEPS", "SAMPLER", "SCHEDULER", "CFG", "ui")
    FUNCTION = "config"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
Balanced (16-24):
    "euler | simple": 16,
    "res_multistep | beta": 20,
    "deis | beta": 20,
    "seeds_2 | beta": 18,
    "ddim | ddim_uniform": 24,

Enhanced (22-30)":
    "euler | simple": 22,
    "res_multistep | beta": 26,
    "deis | beta": 26,
    "seeds_2 | beta": 24,
    "ddim | ddim_uniform": 30,

High Quality (28-36)":
    "euler | simple": 28,
    "res_multistep | beta": 32,
    "deis | beta": 32,
    "seeds_2 | beta": 30,
    "ddim | ddim_uniform": 36,
    """

    def config(self, combo, quality, cfg):
        sampler, scheduler = COMBO_PRESETS[combo]
        steps = STEPS_PRESETS[quality][combo]
        ui = f"{steps} steps\n"
        return (steps, sampler, scheduler, cfg, ui)


NODE_CLASS_MAPPINGS = {"Config Advanced": Config_Advanced}
NODE_DISPLAY_NAME_MAPPINGS = {"Config Advanced": "Config Advanced"}



# KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
#                   "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
#                   "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
#                   "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
#                   "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece"]

# SCHEDULER_HANDLERS = {
#     "simple": SchedulerHandler(simple_scheduler),
#     "sgm_uniform": SchedulerHandler(partial(normal_scheduler, sgm=True)),
#     "karras": SchedulerHandler(k_diffusion_sampling.get_sigmas_karras, use_ms=False),
#     "exponential": SchedulerHandler(k_diffusion_sampling.get_sigmas_exponential, use_ms=False),
#     "ddim_uniform": SchedulerHandler(ddim_scheduler),
#     "beta": SchedulerHandler(beta_scheduler),
#     "normal": SchedulerHandler(normal_scheduler),
#     "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
#     "kl_optimal": SchedulerHandler(kl_optimal_scheduler, use_ms=False),