import math
import os, sys
import torch
import torch.nn.functional as F

# Bring in ratio presets from your config
sys.path.append(os.path.dirname(__file__))
from _iad_utils_ import RATIO_PRESETS

class Isolate:
    MASK_VAL = 255.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "inpaint_mask": ("MASK",),
                "ratio":        (list(RATIO_PRESETS.keys()),),
                "Megapixel":    ("FLOAT", {"default": 1.00, "min": 0.10, "max": 8.00, "step": 0.01}),
            },
            "optional": {
                "context_mask": ("MASK",),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "MASK", "IAD_DATA", "STRING")
    RETURN_NAMES  = ("cropped_image", "cropped_mask", "uncropped_mask", "iad_data", "ui")
    FUNCTION      = "crop_by_mask"
    CATEGORY = "Steudio/Isolate and Dominate"

    def _normalize_mask(self, m: torch.Tensor) -> torch.Tensor:
        m = m.to(torch.float32)
        maxv = float(m.max().item()) if m.numel() > 0 else 0.0
        if maxv <= 1.0 + 1e-6:
            return m
        return m / self.MASK_VAL

    def _prepare_mask_for_bbox(self, m: torch.Tensor, target_shape) -> torch.Tensor:
        m = self._normalize_mask(m)
        if m.shape[:2] != target_shape:
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode="nearest"
            ).squeeze(0).squeeze(0)
        return m

    @staticmethod
    def _mask_bbox(mask_hw: torch.Tensor):
        ys, xs = torch.where(mask_hw > 0)
        if ys.numel() == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1

    @staticmethod
    def _fit_box_with_aspect(min_bbox, aspect, W, H):
        xmin, ymin, xmax, ymax = min_bbox
        w0, h0 = xmax - xmin, ymax - ymin
        if w0 <= 0 or h0 <= 0:
            return None

        eps = 1e-6
        if (w0 / h0) < aspect - eps:
            w_req = int(math.ceil(h0 * aspect))
            h_req = h0
        elif (w0 / h0) > aspect + eps:
            w_req = w0
            h_req = int(math.ceil(w0 / aspect))
        else:
            w_req, h_req = w0, h0

        if w_req > W or h_req > H:
            return None

        cx = 0.5 * (xmin + xmax)
        l_center = int(round(cx - w_req / 2))
        l_min = max(0, xmax - w_req)
        l_max = min(W - w_req, xmin)
        if l_min > l_max:
            return None
        l = min(max(l_center, l_min), l_max)
        r = l + w_req

        cy = 0.5 * (ymin + ymax)
        t_center = int(round(cy - h_req / 2))
        t_min = max(0, ymax - h_req)
        t_max = min(H - h_req, ymin)
        if t_min > t_max:
            return None
        t = min(max(t_center, t_min), t_max)
        b = t + h_req

        return l, t, r, b

    def _calculate_target_size_from_pair(self, rw, rh, Megapixel):
        aspect = rw / rh if rh != 0 else 1.0
        target_pixels = int(Megapixel * 1024 * 1024)

        width = int((target_pixels * aspect) ** 0.5)
        height = int(width / aspect) if aspect != 0 else width

        width = max(64, (width // 64) * 64)
        height = max(64, (height // 64) * 64)

        best_w, best_h = width, height
        best_ratio_diff = abs((width / height) - aspect)

        for dw in (-64, 0, 64):
            for dh in (-64, 0, 64):
                w = max(64, width + dw)
                h = max(64, height + dh)
                ratio_diff = abs((w / h) - aspect)
                pixel_diff = abs((w * h) - target_pixels) / target_pixels
                if ratio_diff < best_ratio_diff and pixel_diff <= 0.05:
                    best_w, best_h = w, h
                    best_ratio_diff = ratio_diff

        return best_w, best_h, aspect

    def _calculate_target_size_for_aspect(self, aspect, Megapixel):
        target_pixels = int(Megapixel * 1024 * 1024)

        width = int((target_pixels * aspect) ** 0.5)
        height = int(width / aspect) if aspect != 0 else width

        width = max(64, (width // 64) * 64)
        height = max(64, (height // 64) * 64)

        best_w, best_h = width, height
        best_ratio_diff = abs((width / height) - aspect)

        for dw in (-64, 0, 64):
            for dh in (-64, 0, 64):
                w = max(64, width + dw)
                h = max(64, height + dh)
                ratio_diff = abs((w / h) - aspect)
                pixel_diff = abs((w * h) - target_pixels) / target_pixels
                if ratio_diff < best_ratio_diff and pixel_diff <= 0.05:
                    best_w, best_h = w, h
                    best_ratio_diff = ratio_diff

        return best_w, best_h, aspect

    def crop_by_mask(self, image: torch.Tensor, inpaint_mask: torch.Tensor,
                     ratio: str, Megapixel: float,
                     context_mask: torch.Tensor = None):

        squeezed = False
        if image.ndim == 4 and image.shape[0] == 1:
            squeezed = True
            image, inpaint_mask = image[0], inpaint_mask[0]
            if context_mask is not None:
                context_mask = context_mask[0]

        if inpaint_mask.ndim == 3:
            inpaint_mask = inpaint_mask[..., 0]
        if context_mask is not None and context_mask.ndim == 3:
            context_mask = context_mask[..., 0]

        inpaint_for_bbox = self._prepare_mask_for_bbox(inpaint_mask, inpaint_mask.shape[:2])
        if context_mask is not None:
            context_for_bbox = self._prepare_mask_for_bbox(context_mask, inpaint_mask.shape[:2])
            combined_mask = torch.clamp(inpaint_for_bbox + context_for_bbox, 0.0, 1.0)
        else:
            combined_mask = inpaint_for_bbox

        H, W = image.shape[0], image.shape[1]
        uncropped_mask = inpaint_for_bbox.unsqueeze(0)
        bbox = self._mask_bbox(combined_mask)

        if bbox is None:
            H, W = image.shape[0], image.shape[1]

            if ratio == "auto":
                # Use image (or mask) dimensions to determine aspect
                H, W = image.shape[0], image.shape[1]
                req_rw, req_rh = W, H
            else:
                req_rw, req_rh = RATIO_PRESETS.get(ratio, (1, 1))

            req_aspect = req_rw / req_rh if req_rh != 0 else 1.0

            # Match longer edge, compute shorter from ratio
            if W >= H:
                crop_w = W
                crop_h = min(H, int(round(W / req_aspect)))
            else:
                crop_h = H
                crop_w = min(W, int(round(H * req_aspect)))

            # Center the crop
            l = max(0, (W - crop_w) // 2)
            t = max(0, (H - crop_h) // 2)
            r = l + crop_w
            b = t + crop_h

            # Create white mask for that crop
            inpaint_mask = torch.zeros((H, W), dtype=torch.float32)
            inpaint_mask[t:b, l:r] = 1.0
            uncropped_mask = inpaint_mask.unsqueeze(0)

            # Now continue as if we had a bbox
            bbox = (l, t, r, b)

        xmin, ymin, xmax, ymax = bbox
        bbox_original = (xmin, ymin, xmax, ymax)

        # Auto mode: find closest preset from mask bbox
        if ratio == "auto":
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            if bbox_w <= 0 or bbox_h <= 0:
                # fallback to 1:1 if bbox is degenerate
                req_rw, req_rh = 1, 1
                closest_ratio = None
            else:
                gcd = math.gcd(bbox_w, bbox_h)
                simplified_width = bbox_w // gcd
                simplified_height = bbox_h // gcd
                closest_ratio = None
                min_difference = float('inf')
                req_rw, req_rh = 1, 1  # default
                for name, (rw, rh) in RATIO_PRESETS.items():
                    if name in ("auto", ""):
                        continue
                    difference = abs(simplified_width / simplified_height - rw / rh)
                    if difference < min_difference:
                        min_difference = difference
                        closest_ratio = name
                        req_rw, req_rh = rw, rh
            ratio = closest_ratio if closest_ratio else ratio
        else:
            req_rw, req_rh = RATIO_PRESETS.get(ratio, (1, 1))

        req_aspect = (req_rw / req_rh) if req_rh != 0 else 1.0
        best = None

        def candidate_score(name, box, aspect):
            l, t, r, b = box
            w0, h0 = xmax - xmin, ymax - ymin
            w, h = r - l, b - t
            area_add = (w * h) - (w0 * h0)
            cx0, cy0 = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            cx, cy = 0.5 * (l + r), 0.5 * (t + b)
            center_shift = abs(cx - cx0) + abs(cy - cy0)
            diff = abs(aspect - req_aspect)
            return (diff, area_add, center_shift, name, box, aspect)

        # Try requested ratio
        box = self._fit_box_with_aspect(bbox_original, req_aspect, W, H)
        if box is not None:
            best = candidate_score(ratio, box, req_aspect)

        # Try all other presets
        for name, (rw, rh) in RATIO_PRESETS.items():
            if name in ("auto", ""):
                continue
            if name == ratio and best is not None:
                continue
            aspect = (rw / rh) if rh != 0 else 1.0
            box = self._fit_box_with_aspect(bbox_original, aspect, W, H)
            if box is None:
                continue
            cand = candidate_score(name, box, aspect)
            if (best is None) or (cand < best):
                best = cand

        if best is not None:
            _, _, _, chosen_name, chosen_box, chosen_aspect = best
            l, t, r, b = chosen_box
            if chosen_name in RATIO_PRESETS:
                rw, rh = RATIO_PRESETS[chosen_name]
                width, height, target_aspect = self._calculate_target_size_from_pair(rw, rh, Megapixel)
                final_ratio_used = chosen_name
            else:
                width, height, target_aspect = self._calculate_target_size_for_aspect(chosen_aspect, Megapixel)
                final_ratio_used = f"custom({chosen_aspect:.6f})"

            # Strict aspect lock
            crop_w = r - l
            crop_h = b - t
            target_ratio = width / height
            crop_ratio = crop_w / crop_h if crop_h > 0 else target_ratio
            if abs(crop_ratio - target_ratio) > 1e-9:
                if crop_ratio < target_ratio:
                    extra = int(round(crop_h * target_ratio)) - crop_w
                    l = max(0, l - extra // 2)
                    r = min(W, r + (extra - extra // 2))
                else:
                    extra = int(round(crop_w / target_ratio)) - crop_h
                    t = max(0, t - extra // 2)
                    b = min(H, b + (extra - extra // 2))
        else:
            # No ratio fits → full canvas
            l, t, r, b = 0, 0, W, H
            target_aspect = W / H if H > 0 else 1.0
            width, height, _ = self._calculate_target_size_for_aspect(target_aspect, Megapixel)
            final_ratio_used = f"canvas({W}:{H})"

        # Crop and resize
        img_crop  = image[t:b, l:r, :]
        mask_crop = inpaint_mask[t:b, l:r]
        img_nchw  = img_crop.permute(2, 0, 1).unsqueeze(0)
        mask_nchw = mask_crop.unsqueeze(0).unsqueeze(0)

        img_rs  = F.interpolate(img_nchw,  size=(height, width), mode="bilinear", align_corners=False)
        mask_rs = F.interpolate(mask_nchw, size=(height, width), mode="nearest")

        out_img  = img_rs.squeeze(0).permute(1, 2, 0)
        out_mask = self._normalize_mask(mask_rs.squeeze(0).squeeze(0)).unsqueeze(0)

        if squeezed:
            out_img = out_img.unsqueeze(0)

        final_aspect = (r - l) / (b - t) if (b - t) > 0 else None

        iad_data = {
            'canvas_w': W,
            'canvas_h': H,
            'bbox_x': int(l),
            'bbox_y': int(t),
            'bbox_w': int(r - l),
            'bbox_h': int(b - t),
            "image": image.clone(),
            "uncropped_mask": uncropped_mask.clone(),
        }
        
        ui = (
            f"image_size: ({W}, {H})\n"
            f"crop_size_final: ({r - l}, {b - t})\n"
            f"target_aspect: {target_aspect}\n"
            f"final_aspect: {final_aspect}\n"
            f"bbox_original: {bbox_original}\n"
            f"bbox_final: ({l}, {t}, {r}, {b})\n"
            f"ratio_requested: {ratio} ({req_aspect:.6f})\n"
            f"ratio_used: {final_ratio_used}\n"
            f"computed_width: {width}, computed_height: {height}\n"
            f"Megapixel: {Megapixel} (1MP = 1024x1024)\n"
        )

        return (out_img, out_mask, uncropped_mask, iad_data, ui)


NODE_CLASS_MAPPINGS = {"Isolate": Isolate}
NODE_DISPLAY_NAME_MAPPINGS = {"Isolate": "Isolate"}