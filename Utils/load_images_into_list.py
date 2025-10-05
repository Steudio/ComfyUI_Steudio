# load_images_into_list.py
# Original LoadImagesFromFolderKJ from https://github.com/kijai/ComfyUI-KJNodes

import comfy.samplers
import math
import os
import torch
import math
from PIL import Image, ImageOps
import numpy as np

# Original LoadImagesFromFolderKJ code from https://github.com/kijai/ComfyUI-KJNodes
class Load_Images_into_List:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "Load_Images_into_List"
    CATEGORY = "Steudio/Utils"

    def Load_Images_into_List(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        # List all files in the directory
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        if not dir_files:
            raise FileNotFoundError(f"No valid image files found in directory '{directory}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        images = []

        for image_path in dir_files:
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image)[None,]
                images.append(image_tensor)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        if not images:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        # Concatenate images into a single tensor
        images = torch.cat(images, dim=0)
        return ([images[i].unsqueeze(0) for i in range(images.shape[0])],)

NODE_CLASS_MAPPINGS = {
    "Load Images into List": Load_Images_into_List,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Images into List": "Load Images into List",

}
