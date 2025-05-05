<center>

# **ComfyUI [**Steudio**](https://linktr.ee/steudio)**

</center>

# Divide and Conquer Node Suite 2.0.0

:pushpin: If you're updating from version 1.x.x, make sure to replace the old nodes with the new ones in your workflow to avoid potential errors.

# Intro

A suite of nodes designed to enhance image upscaling. It calculates the optimal upscale resolution and divides the image into tiles, ready for individual processing using your preferred workflow. After processing, the tiles are seamlessly merged into a larger image, offering sharper and more detailed visuals.

<img src="Images/DaC_Suite.png" alt="Node" style="width: 100%;">

## 1. Divide and Conquer Algorithm | Node

Taking into account tile dimensions, tile overlap, and the minimum scale factor, the node upscales the image to optimal dimensions, avoiding the creation of unnecessary tiles.<br>
<img src="Images/DaC_Algo.png" alt="Node" style="width: 50%;">

### Inputs / Outputs
**`::image`**:
The image you want to upscale.<br>
**`::upscale_model`(optional)**:
To use with Comfy Core node **UpscaleModelLoader**.<br>
**`IMAGE::`**:
Optimized image dimensions to connect to **Divide Image and Select Tile**.<br>
**`dac_data:`**:
Data to pass along to following nodes: **Divide Image and Select Tile** and **Combine Tiles**.<br>
**`ui:`**
Image processing algorithm summary.<br>

### Parameters


**`tile_width`**: This parameter specifies the width of each tile that the image will be divided into.*The default value is 1024 pixels.*<br>
**`tile_height`**: This parameter specifies the height of each tile that the image will be divided into.The default value is 1024 pixels.<br>
**`min_overlap`**: This parameter specifies the minimum amount of overlap between adjacent tiles to help blend the tiles seamlessly when they are combined back together. *The default value is '1/32 Tile', and it can range from "None" to "1/2 tile".*<br>
**`min_scale_factor`**: This parameter determines the minimal scale factor. The effective scale factor will be determined by the tile dimensions and tile overlap. *The default value is 3, and it can range from 1.0 to 8.0*<br>
**`tile_order`**: This parameter specifies the order in which the tiles are processed. It can be either 'linear' or 'spiral'. 'Linear' processes the tiles in a row-by-row manner, while 'spiral' processes them in an outward clockwise spiral pattern ending at the center. *The default value is "spiral".*<br>
<img src="Images/Order_Spiral.png" alt="Spiral" style="width: 25%;">
<img src="Images/Order_Linear.png" alt="Linear" style="width: 25%;"><br>
**`Scaling_method`** The image will go through a process to meet the optimal upscaled dimensions. *The default value is lanczos.*<br>
**`use_upscale_with_model`(optional)**: True/false switch to enable the feature.
Bypassed if `::upscale_model` is not connected. *The default value is true.*<br>

## 2. Divide Image and Select Tile | Node
Taking into account tile dimensions, tile overlap, and final image dimensions, the node calculates coordinates and divides the image into tiles.

<img src="Images/DaC_Divide.png" alt="Node" style="width: 50%;">

### Inputs / Outputs
**`::image`**
The image you want to upscale.<br>
**`::dac_data`**
Connect from the dac_data output of the "Divide and Conquer Algorithm" node.<br>
**`TILE(S)::`**
All tiles (0) or specific tile (#) to path through (#).<br>
**`ui::`**
Matrix visualization.<br>

### Parameters
**`position`**:
Select the tile(s) to path through. Position (0) indicates all the tiles, while Position (#) specifies an individual tile.<br>


## 3. Combine Tiles | Node
Combines the processed tiles back into a single image, applying a **Gaussian blur mask** on the overlapping pixels to ensure smooth transitions between the overlapping tiles.<br>
<img src="Images/DaC_Combine.png" alt="Node" style="width: 50%;">

### Inputs / Outputs

**`::images`**
The tiles you want to combine into one upscaled image.<br>
**`::dac_data`**
Connect from the dac_data output of the "Divide and Conquer Algorithm" node.<br>
**`image::`**
The combined image, made of multiple tiles.<br>
**`ui::`**
Matrix visualization.<br>

## Workflow example
Get the workflow directly from ComfyUI menu:<br>
Workflow > Browse Templates > comfyui_steudio

<img src="Images/DaC_Workflow.png" alt="Workflow" style="width: 100%;">

:grey_exclamation: This workflow uses the following optional nodes:<br>
[Set/Get](https://github.com/kijai/ComfyUI-KJNodes) | 
[Image Comparer](https://github.com/rgthree/rgthree-comfy) | 
[Fast Groups Bypasser](https://github.com/rgthree/rgthree-comfy) |
[ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) |
[TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache)

## Video Tutorial
Coming Soon!

:100: cropped comparison.<br>
<a href="https://imgsli.com/Mzc2MzA3">
    <img src="Images/DaC_Side-by-side.png" alt="Full image" style="width: 100%;">
</a><br>

[Full image with more details generated](https://imgsli.com/Mzc2MzA3)<br>The image has been upscaled 2×, three times.

[Full image with less details generated ](https://imgsli.com/Mzc2OTMx)<br>The image has been upscaled 2×, three times.



## TIPS
- General upscaling guidelines do apply, but the **Divide and Conquer Node Suite** offers better control per tile, enabling higher detail transfer.
- Instead of generating the entire set of tiles, a single tile can be generated as a test sample to verify your img2img parameters.
- Ensure that the input folder only contains the tiles you intend to combine.
- If seams appear in the combine image, increase the overlap.

# Installation
Install via ComfyUI-Manager or Clone this repo into `custom_modules`:

# Changelog
`Version 2.0.0` (2025-05-04)<br>
- Improved user experience.
- Scaling using model is now optional.
- Can generate all tiles or a single tile without fiddling with the links.

`Version 1.2.1` (2025-02-10)<br>
- No more abnormally large upscales.
- Given the right conditions, it is now possible to tile the image without upscaling it.
  
`Version 1.1.0` (2025-01-05)<br>
- Nodes now process images as a list instead of a batch, enabling the execution of divide and combine operations in one go.
- Improved Gaussian blur for blending masks.
- Minor fixes.

`Version 1.0.0` (2024-12-01) Initial public release.

# Acknowledgements
This repository utilizes code from:<br>
[ComfyUI](https://github.com/comfyanonymous/ComfyUI/) | [Comfyroll Studio](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) | [SimpleTiles](https://github.com/kinfolk0117/ComfyUI_SimpleTiles) | [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | [RGThree](https://github.com/rgthree/rgthree-comfy) | [Cubiq](https://github.com/cubiq/ComfyUI_essentials) | [Pythongosssss](https://github.com/pythongosssss/)<br>

# License
GNU GENERAL PUBLIC LICENSE Version 3, see [LICENSE](LICENSE)

# Thank you
Copyright (c) 2025, Steudio - https://github.com/steudio

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/steudio)
