# dominate.py

import torch
import torch.nn.functional as F

class Merge_Isolated:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edited_image": ("IMAGE",),   # BHWC float32 [0,1], any size
                "iad_data": ("IAD_DATA",),     # dict or list of dicts with canvas_w/h, bbox_x/y/w/h, image, uncropped_mask
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Steudio/Isolate and Dominate"

    # -------- helpers --------
    def _to_bchw_img(self, img: torch.Tensor) -> torch.Tensor:
        """Convert IMAGE to BCHW float32 in [0,1]."""
        if img.ndim == 4:
            return img.permute(0, 3, 1, 2).to(torch.float32) if img.shape[-1] in (1, 3, 4) else img.to(torch.float32)
        if img.ndim == 3:
            if img.shape[-1] in (1, 3, 4):
                return img.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            elif img.shape[0] in (1, 3, 4):
                return img.unsqueeze(0).to(torch.float32)
            else:
                return img.unsqueeze(0).unsqueeze(0).to(torch.float32)
        if img.ndim == 2:
            return img.unsqueeze(0).unsqueeze(0).to(torch.float32)
        raise ValueError(f"Unsupported IMAGE shape {tuple(img.shape)}")

    def _to_bchw_mask(self, msk: torch.Tensor) -> torch.Tensor:
        """Convert MASK to BCHW float32 in [0,1], single channel."""
        if msk.ndim == 4:
            if msk.shape[-1] == 1:
                out = msk.permute(0, 3, 1, 2)
            elif msk.shape[1] == 1:
                out = msk
            elif msk.shape[-1] in (3, 4):
                out = msk[..., :1].permute(0, 3, 1, 2)
            else:
                out = msk[:, :1, ...]
        elif msk.ndim == 3:
            out = msk[:1, ...].unsqueeze(0) if msk.shape[0] in (1, 3, 4) else msk[..., :1].permute(2, 0, 1).unsqueeze(0)
        elif msk.ndim == 2:
            out = msk.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported MASK shape {tuple(msk.shape)}")
        return out.to(torch.float32).clamp(0.0, 1.0)[:, 0:1]

    def _broadcast_to(self, x: torch.Tensor, b: int) -> torch.Tensor:
        if x.shape[0] == b:
            return x
        if x.shape[0] == 1:
            return x.expand(b, *x.shape[1:])
        raise ValueError(f"Cannot broadcast batch {x.shape[0]} to {b}")

    def _align_channels(self, src: torch.Tensor, target_c: int) -> torch.Tensor:
        """Simple channel align: repeat if 1->3/4, slice if 3/4->1, else keep/trim."""
        c = src.shape[1]
        if c == target_c:
            return src
        if c == 1 and target_c in (3, 4):
            return src.repeat(1, target_c, 1, 1)
        if c in (3, 4) and target_c == 1:
            return src[:, :1, ...]
        if c > target_c:
            return src[:, :target_c, ...]
        pad = torch.zeros((src.shape[0], target_c - c, src.shape[2], src.shape[3]),
                          dtype=src.dtype, device=src.device)
        return torch.cat([src, pad], dim=1)

    # -------- main --------
    def merge(self, edited_image, iad_data):
        # Normalize iad_data to per-sample dicts
        if isinstance(iad_data, dict):
            iad_list = [iad_data]
        else:
            iad_list = list(iad_data)

        B = len(iad_list)

        # Validate and load base image + mask from iad_data
        def require_int(d, k):
            if k not in d:
                raise KeyError(f"iad_data missing key '{k}'")
            return int(d[k])

        def require_tensor(d, k):
            if k not in d:
                raise KeyError(f"iad_data missing tensor '{k}'")
            return d[k]

        cw0 = require_int(iad_list[0], 'canvas_w')
        ch0 = require_int(iad_list[0], 'canvas_h')
        if cw0 <= 0 or ch0 <= 0:
            raise ValueError("iad_data.canvas_w and canvas_h must be positive")

        # Load and broadcast edited_image
        crop_bchw = self._to_bchw_img(edited_image)
        crop_bchw = self._broadcast_to(crop_bchw, B)

        # Prepare output tensor
        out_C = None
        out = torch.zeros((B, 0, ch0, cw0), dtype=torch.float32)  # placeholder init

        for i, d in enumerate(iad_list):
            # Verify canvas size consistency
            cw = require_int(d, 'canvas_w'); ch = require_int(d, 'canvas_h')
            if (cw, ch) != (cw0, ch0):
                raise ValueError(f"Inconsistent canvas in batch {i}: {(cw, ch)} != {(cw0, ch0)}")

            # Extract base image and mask from iad_data
            base_img = require_tensor(d, 'image')
            uncropped_mask = require_tensor(d, 'uncropped_mask')

            base_bchw = self._to_bchw_img(base_img)
            mask_bchw = self._to_bchw_mask(uncropped_mask)

            if out_C is None:
                out_C = base_bchw.shape[1] if base_bchw.shape[1] in (1, 3, 4) \
                        else max(base_bchw.shape[1], crop_bchw.shape[1])
                out = torch.zeros((B, out_C, ch0, cw0), dtype=torch.float32, device=base_bchw.device)

            # Resize base to canvas
            base_i = self._align_channels(base_bchw, out_C)
            if base_i.shape[-2:] != (ch0, cw0):
                base_i = F.interpolate(base_i, size=(ch0, cw0), mode="bilinear", align_corners=False)

            out[i:i+1] = base_i

            # Skip if mask has no active pixels
            mask_bchw = mask_bchw.to(base_i.device)
            if mask_bchw.shape[2:] != (ch0, cw0):
                mask_bchw = F.interpolate(mask_bchw, size=(ch0, cw0), mode="nearest")
            mask_hw = mask_bchw[0, 0].clamp(0.0, 1.0)
            if not (mask_hw > 0).any():
                continue

            # Get bbox
            bx = require_int(d, 'bbox_x')
            by = require_int(d, 'bbox_y')
            bw = require_int(d, 'bbox_w')
            bh = require_int(d, 'bbox_h')
            if bw <= 0 or bh <= 0:
                continue

            # Clip to canvas
            x0 = max(0, bx); y0 = max(0, by)
            x1 = min(cw0, bx + bw); y1 = min(ch0, by + bh)
            if x1 <= x0 or y1 <= y0:
                continue

            clip_w = x1 - x0
            clip_h = y1 - y0

            # Prepare crop
            crop_i = self._align_channels(crop_bchw[i:i+1], out_C)
            if (bh, bw) != crop_i.shape[-2:]:
                crop_i = F.interpolate(crop_i, size=(bh, bw), mode="bilinear", align_corners=False)

            px0 = max(0, -bx)
            py0 = max(0, -by)
            crop_slice = crop_i[:, :, py0:py0+clip_h, px0:px0+clip_w]

            # Mask slice
            local_mask = mask_bchw[:, :, y0:y1, x0:x1]
            if local_mask.shape[-2:] != (clip_h, clip_w):
                local_mask = F.interpolate(local_mask, size=(clip_h, clip_w), mode="nearest")

            # Blend
            roi = out[i:i+1, :, y0:y1, x0:x1]
            out[i:i+1, :, y0:y1, x0:x1] = crop_slice * local_mask + roi * (1.0 - local_mask)

        # Return as BHWC, clamped to [0,1]
        return (out.permute(0, 2, 3, 1).clamp(0.0, 1.0),)
    
NODE_CLASS_MAPPINGS = {"Merge_Isolated": Merge_Isolated}
NODE_DISPLAY_NAME_MAPPINGS = {"Merge_Isolated": "Merge Isolated"}
