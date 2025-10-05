import numpy as np
import torch
import torch.nn.functional as F

class mask_splitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":      ("MASK",),
                "threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("inpaint_mask", "context_mask")
    FUNCTION = "split"
    CATEGORY = "Steudio/Utils"
    OUTPUT_IS_LIST = (False, False)
    OUTPUT_NODE = True

    @staticmethod
    def _to_tensor(mask):
        if isinstance(mask, torch.Tensor):
            t = mask.detach().float().cpu()
        else:
            arr = np.array(mask, copy=False).astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3 and (arr.shape[0] == 1 or arr.shape[-1] == 1):
                arr = arr.squeeze(0) if arr.shape[0] == 1 else arr[..., 0]
            t = torch.from_numpy(arr)
        return t.unsqueeze(0) if t.ndim == 2 else t

    def split(self, mask, threshold):
        # Normalize to [0,1] float tensor [1,H,W]
        m = self._to_tensor(mask).clamp(0.0, 1.0)

        # inpaint_mask: 1 where m >= threshold, else 0
        inpaint = (m >= threshold).float()

        # context_mask: 1 where 0 < m < threshold, else 0
        context = ((m > 0.0) & (m < threshold)).float()

        return inpaint, context

NODE_CLASS_MAPPINGS = {
    "mask_splitter": mask_splitter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "mask_splitter": "Mask Splitter",
}