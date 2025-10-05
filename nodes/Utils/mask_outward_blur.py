import numpy as np
import torch
import math
import torch.nn.functional as F

class mask_outward_blur:
    MASK_VAL = 255

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "radius_px": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "convert"
    CATEGORY = "Steudio/Utils"
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = True

    @staticmethod
    def _to_uint8_2d(mask, mask_val=255):
        if isinstance(mask, torch.Tensor):
            m = mask.detach().cpu().numpy()
        else:
            m = np.array(mask)
        if m.ndim == 3:
            if m.shape[0] == 1:
                m = m[0]
            elif m.shape[-1] == 1:
                m = m[..., 0]
        if m.dtype != np.uint8:
            if m.max() <= 1.0:
                m = (m * mask_val).clip(0, mask_val).astype(np.uint8)
            else:
                m = m.clip(0, mask_val).astype(np.uint8)
        return m

    @staticmethod
    def _odd_kernel_from_radius(radius_px: int) -> int:
        k = max(1, int(radius_px) * 2 + 1)
        return k if (k % 2 == 1) else (k + 1)

    @staticmethod
    def _gaussian_min_expansion(radius_px, sigma=None, tol_dn=1):
        """Minimal expansion so Gaussian blur drop < tol_dn DN at edge."""
        if sigma is None:
            sigma = 0.3*((radius_px*2-1)*0.5 - 1) + 0.8
        coords = np.arange(-radius_px, radius_px+1)
        w = np.exp(-0.5*(coords/sigma)**2)
        w /= w.sum()
        target_ratio = (255 - tol_dn) / 255.0
        for E in range(radius_px+1):
            mask_1 = (np.abs(coords) <= E).astype(float)
            if (w * mask_1).sum() >= target_ratio:
                return E
        return radius_px

    @staticmethod
    def _dilate_torch(mask_tensor, radius):
        """Binary dilation with circular kernel in PyTorch."""
        if radius <= 0:
            return mask_tensor
        # Build circular structuring element
        y, x = torch.meshgrid(torch.arange(-radius, radius+1),
                              torch.arange(-radius, radius+1),
                              indexing='ij')
        se = ((x**2 + y**2) <= radius**2).float()[None, None]
        se = se.to(mask_tensor.device)
        # Use 2D conv for dilation (max filter)
        pad = radius
        mask_f = (mask_tensor > 0).float()
        # Conv2d with weights as structuring element
        out = F.conv2d(mask_f, se, padding=pad)
        return (out > 0).float()

    @staticmethod
    def _gaussian_blur_torch(img, radius_px):
        """Gaussian blur in PyTorch matching OpenCV border replicate."""
        if radius_px <= 0:
            return img
        k = mask_outward_blur._odd_kernel_from_radius(radius_px)
        sigma = 0.3*((k-1)*0.5 - 1) + 0.8
        coords = torch.arange(k, device=img.device) - (k-1)/2
        g = torch.exp(-0.5*(coords/sigma)**2)
        g = g / g.sum()
        g_col = g.view(1, 1, -1, 1)
        g_row = g.view(1, 1, 1, -1)
        # Border replicate: pad with 'replicate'
        img = F.pad(img, (radius_px, radius_px, radius_px, radius_px), mode='replicate')
        out = F.conv2d(img, g_col, groups=img.shape[1])
        out = F.conv2d(out, g_row, groups=img.shape[1])
        return out

    def convert(self, mask, radius_px):
        m8 = self._to_uint8_2d(mask, self.MASK_VAL)
        h, w = m8.shape

        if radius_px <= 0:
            return (torch.from_numpy(m8.astype(np.float32)/self.MASK_VAL).unsqueeze(0),)

        if not np.any(m8):
            return (torch.zeros((1, h, w), dtype=torch.float32),)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        silhouette = torch.from_numpy((m8 > 0).astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

        pre_expansion = self._gaussian_min_expansion(radius_px, tol_dn=1)
        expanded = self._dilate_torch(silhouette, pre_expansion)

        blurred = self._gaussian_blur_torch(expanded, radius_px)

        out_f32 = blurred.squeeze(0).clamp(0, 1).cpu()
        return (out_f32,)


NODE_CLASS_MAPPINGS = {
    "mask_outward_blur": mask_outward_blur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mask_outward_blur": "Mask Outward Blur",
}