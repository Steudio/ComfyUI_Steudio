import torch
import torch.nn.functional as F

class Scale_Image_Mask_old:
    MASK_VAL = 255  # internal value for "padding fill"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 1}),
                "height": ("INT", {"default": 1024, "min": 1}),
                "mode": (["fit", "fill", "pad/crop"], {"default": "fit"}),
                "anchor": ([
                    "◻", "▣", "⊡", "⌜", "⎴", "⌝",
                    "[", "]", "⌞", "⎵", "⌟"
                ], {"default": "◻"}),
                "zoom_out": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "inpaint", "canvas")
    FUNCTION = "scale"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
Outputs:
- IMAGE   : transformed image
- inpaint : transformed mask with original values preserved, padding as black
- canvas  : binary map of padded regions only

Anchor positions:
◻  | Center (default)
▣  | Dynamic mask center
⊡  | Force mask center
⌜  | Top-left
⎴  | Top-center
⌝  | Top-right
[   | Middle-left
]   | Middle-right
⌞  | Bottom-left
⎵  | Bottom-center
⌟  | Bottom-right
"""

    def scale(self, image: torch.Tensor, width: int, height: int,
              mode: str, anchor: str, zoom_out: float,
              mask: torch.Tensor = None):

        anchor_map = {
            "◻": (0.5, 0.5), "▣": (0.5, 0.5), "⊡": (0.5, 0.5),
            "⌜": (0.0, 0.0), "⎴": (0.0, 0.5), "⌝": (0.0, 1.0),
            "[":  (0.5, 0.0), "]":  (0.5, 1.0),
            "⌞": (1.0, 0.0), "⎵": (1.0, 0.5), "⌟": (1.0, 1.0),
        }
        ay, ax = anchor_map.get(anchor, (0.5, 0.5))

        def _safe_crop(x, tgt_h, tgt_w, crop_ay, crop_ax):
            C, H, W = x.shape
            crop_h = max(H - tgt_h, 0)
            crop_w = max(W - tgt_w, 0)
            top = int(crop_h * crop_ay)
            bottom = crop_h - top
            left = int(crop_w * crop_ax)
            right = crop_w - left
            return x[:, top:H-bottom, left:W-right]

        def _safe_pad(x, tgt_h, tgt_w, pad_ay, pad_ax, v=0):
            C, H, W = x.shape
            pad_h = max(tgt_h - H, 0)
            pad_w = max(tgt_w - W, 0)
            top = int(pad_h * pad_ay)
            bottom = pad_h - top
            left = int(pad_w * pad_ax)
            right = pad_w - left
            return F.pad(x, (left, right, top, bottom), mode="constant", value=v)

        def _crop_centered(x, tgt_h, tgt_w, cy, cx):
            C, H, W = x.shape
            top = max(int(cy - tgt_h // 2), 0)
            left = max(int(cx - tgt_w // 2), 0)
            bottom = min(top + tgt_h, H)
            right = min(left + tgt_w, W)
            return x[:, top:bottom, left:right]

        squeezed = False
        if image.ndim == 4 and image.shape[0] == 1:
            squeezed = True
            image = image[0]
            if mask is not None:
                mask = mask[0]

        img_H, img_W = image.shape[:2]
        if mask is None:
            mask = torch.zeros((img_H, img_W), device=image.device)
        elif mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.shape[0] == 1 and mask.shape[1:] == (img_H, img_W):
                mask = mask[0]
            elif mask.shape[0] == 1 and mask.shape[1] == 1 and mask.shape[2] == 1:
                mask = torch.zeros((img_H, img_W), device=image.device)
            elif mask.shape[1:] == (img_H, img_W):
                mask = mask[0] if mask.shape[0] == 1 else mask
            else:
                mask = torch.zeros((img_H, img_W), device=image.device)
        elif mask.ndim == 2:
            if mask.shape != (img_H, img_W):
                mask = torch.zeros((img_H, img_W), device=image.device)
        else:
            mask = torch.zeros((img_H, img_W), device=image.device)

        if mask.ndim == 3:
            mask = mask[..., 0]

        if torch.is_floating_point(mask) and mask.max() > 0:
            if mask.max() <= 1.0:
                mask = mask.clamp(0.0, 1.0).float()
            else:
                mask = (mask / float(self.MASK_VAL)).clamp(0.0, 1.0).float()
            preserve_float = True
        else:
            preserve_float = False
            if mask.max() <= 1.0:
                mask = (mask * self.MASK_VAL).clamp(0, self.MASK_VAL).to(torch.uint8)
            else:
                mask = mask.to(torch.uint8)

        img_chw = image.permute(2, 0, 1)
        msk_chw = mask.unsqueeze(0)
        cov_chw = torch.ones_like(msk_chw)

        def _compute_mask_center(m):
            m2 = m.squeeze(0)
            nz = m2.nonzero(as_tuple=False)
            if nz.numel() > 0:
                ys, xs = nz[:, 0], nz[:, 1]
                return int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2)
            _, H_, W_ = m.shape
            return H_ // 2, W_ // 2

        # Mask‑centric anchors
        if anchor in ("▣", "⊡") and msk_chw is not None:
            cy, cx = _compute_mask_center(msk_chw)
            _, H0, W0 = msk_chw.shape
            if anchor == "▣" and msk_chw.sum() > 0:
                ay, ax = cy / H0, cx / W0
            elif anchor == "⊡":
                ay, ax = 0.5, 0.5
                tgt_h, tgt_w = height, width
                if (cx - tgt_w // 2) >= 0 and ((W0 - cx) - tgt_w // 2) >= 0:
                    ax = cx / W0
                if (cy - tgt_h // 2) >= 0 and ((H0 - cy) - tgt_h // 2) >= 0:
                    ay = cy / H0

        # --- Phase 1: aspect ratio match ---
        if mode in ("fit", "fill"):
            _, H, W = img_chw.shape
            target_ratio = width / height
            current_ratio = W / H

            if mode == "fit":
                if current_ratio < target_ratio:
                    new_W = int(round(H * target_ratio))
                    img_chw = _safe_pad(img_chw, H, new_W, ay, ax, v=0)
                    msk_chw = _safe_pad(msk_chw, H, new_W, ay, ax,
                                        v=0.0 if preserve_float else self.MASK_VAL)
                    cov_chw = _safe_pad(cov_chw, H, new_W, ay, ax, v=0)
                elif current_ratio > target_ratio:
                    new_H = int(round(W / target_ratio))
                    img_chw = _safe_pad(img_chw, new_H, W, ay, ax, v=0)
                    msk_chw = _safe_pad(msk_chw, new_H, W, ay, ax,
                                        v=0.0 if preserve_float else self.MASK_VAL)
                    cov_chw = _safe_pad(cov_chw, new_H, W, ay, ax, v=0)

            elif mode == "fill":
                _, H0, W0 = img_chw.shape
                target_ratio = width / height
                current_ratio = W0 / H0

                if anchor == "⊡":
                    cy, cx = _compute_mask_center(msk_chw)

                if current_ratio > target_ratio:
                    # crop width, keep full height
                    crop_W = int(round(H0 * target_ratio))
                    if anchor == "⊡":
                        half_w = crop_W // 2
                        if (cx - half_w) >= 0 and ((W0 - cx) - half_w) >= 0:
                            ax = cx / W0
                    img_chw = _safe_crop(img_chw, H0, crop_W, ay, ax)
                    msk_chw = _safe_crop(msk_chw, H0, crop_W, ay, ax)
                    cov_chw = _safe_crop(cov_chw, H0, crop_W, ay, ax)

                elif current_ratio < target_ratio:
                    # crop height, keep full width
                    crop_H = int(round(W0 / target_ratio))
                    if anchor == "⊡":
                        half_h = crop_H // 2
                        if (cy - half_h) >= 0 and ((H0 - cy) - half_h) >= 0:
                            ay = cy / H0
                    img_chw = _safe_crop(img_chw, crop_H, W0, ay, ax)
                    msk_chw = _safe_crop(msk_chw, crop_H, W0, ay, ax)
                    cov_chw = _safe_crop(cov_chw, crop_H, W0, ay, ax)

        elif mode == "pad/crop":
            _, H, W = img_chw.shape
            img_chw = _safe_pad(img_chw, height, width, 0.5, 0.5, v=0)
            msk_chw = _safe_pad(msk_chw, height, width, 0.5, 0.5,
                                v=0.0 if preserve_float else self.MASK_VAL)
            cov_chw = _safe_pad(cov_chw, height, width, 0.5, 0.5, v=0)

            if anchor in ["▣", "⊡"]:
                cy, cx = _compute_mask_center(msk_chw)
                if anchor == "⊡":
                    pad_top = max(0, height // 2 - cy)
                    pad_bottom = max(0, (height - height // 2) - (msk_chw.shape[1] - cy))
                    pad_left = max(0, width // 2 - cx)
                    pad_right = max(0, (width - width // 2) - (msk_chw.shape[2] - cx))
                    img_chw = F.pad(img_chw, (pad_left, pad_right, pad_top, pad_bottom),
                                    mode="constant", value=0)
                    msk_chw = F.pad(msk_chw, (pad_left, pad_right, pad_top, pad_bottom),
                                    mode="constant",
                                    value=0.0 if preserve_float else self.MASK_VAL)
                    cov_chw = F.pad(cov_chw, (pad_left, pad_right, pad_top, pad_bottom),
                                    mode="constant", value=0)
                    cy += pad_top
                    cx += pad_left

                img_chw = _crop_centered(img_chw, height, width, cy, cx)
                msk_chw = _crop_centered(msk_chw, height, width, cy, cx)
                cov_chw = _crop_centered(cov_chw, height, width, cy, cx)
            else:
                img_chw = _safe_crop(img_chw, height, width, ay, ax)
                msk_chw = _safe_crop(msk_chw, height, width, ay, ax)
                cov_chw = _safe_crop(cov_chw, height, width, ay, ax)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --- Phase 1.5: Apply zoom (no padding in crop) ---
        _, H_adj, W_adj = img_chw.shape
        if zoom_out != 1.0:
            if zoom_out < 1.0:
                new_h = max(1, int(round(H_adj * zoom_out)))
                new_w = max(1, int(round(W_adj * zoom_out)))
                img_chw = F.interpolate(img_chw.unsqueeze(0), (new_h, new_w),
                                        mode="bilinear", align_corners=False)[0]
                if preserve_float:
                    msk_chw = F.interpolate(msk_chw.unsqueeze(0), (new_h, new_w),
                                            mode="nearest")[0].clamp(0.0, 1.0)
                else:
                    msk_chw = F.interpolate(msk_chw.unsqueeze(0).float(), (new_h, new_w),
                                            mode="nearest")[0].to(torch.uint8)
                cov_chw = F.interpolate(cov_chw.unsqueeze(0).float(), (new_h, new_w),
                                        mode="nearest")[0].to(torch.uint8)

                if mode != "fill":
                    img_chw = _safe_pad(img_chw, H_adj, W_adj, ay, ax, v=0)
                    msk_chw = _safe_pad(msk_chw, H_adj, W_adj, ay, ax,
                                        v=0.0 if preserve_float else self.MASK_VAL)
                    cov_chw = _safe_pad(cov_chw, H_adj, W_adj, ay, ax, v=0)
            else:
                crop_h = max(1, int(round(H_adj / zoom_out)))
                crop_w = max(1, int(round(W_adj / zoom_out)))
                img_chw = _safe_crop(img_chw, crop_h, crop_w, ay, ax)
                msk_chw = _safe_crop(msk_chw, crop_h, crop_w, ay, ax)
                cov_chw = _safe_crop(cov_chw, crop_h, crop_w, ay, ax)

        # --- Phase 2: final scale (no padding in fill) ---
        img_chw = F.interpolate(img_chw.unsqueeze(0), (height, width),
                                mode="bilinear", align_corners=False)[0]
        if preserve_float:
            msk_chw = F.interpolate(msk_chw.unsqueeze(0), (height, width),
                                     mode="nearest")[0].clamp(0.0, 1.0)
        else:
            msk_chw = F.interpolate(msk_chw.unsqueeze(0).float(), (height, width),
                                     mode="nearest")[0].to(torch.uint8)
        cov_chw = F.interpolate(cov_chw.unsqueeze(0).float(), (height, width),
                                 mode="nearest")[0].to(torch.uint8)

        # Outputs
        img_out = img_chw.permute(1, 2, 0)
        inpaint_tensor = msk_chw * (cov_chw > 0).to(msk_chw.dtype)
        canvas_tensor = (1.0 - (cov_chw > 0).float()) if preserve_float \
                        else (1 - (cov_chw > 0).to(msk_chw.dtype))

        if squeezed:
            img_out = img_out.unsqueeze(0)

        return (img_out, inpaint_tensor, canvas_tensor)


NODE_CLASS_MAPPINGS = {"Scale_Image_Mask_old": Scale_Image_Mask_old}
NODE_DISPLAY_NAME_MAPPINGS = {"Scale_Image_Mask_old": "Scale Image & Mask old"}