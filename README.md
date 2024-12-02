<center>

# **ComfyUI [**Steudio**](https://linktr.ee/steudio.com)**

</center>

# Divide and Conquer Node Suite
Introducing a suite of nodes designed to enhance image upscaling. It calculates the optimal upscale resolution and seamlessly divides the image into tiles, ready for individual processing using your preferred workflow. After processing, the tiles are seamlessly merged into a larger image, offering sharper and more detailed visuals.The suite features three main nodes, with additional variants available for added flexibility.

<img src="images/DaC_Suite.png" alt="Node" style="width: 100%;">

## 1. Divide and Conquer Algorithm | Node

Taking into account tile dimensions, tile overlap, and the minimum scale factor, the node upscales the image to optimal dimensions, avoiding the creation of unnecessary tiles.<br>
<img src="images/DaC_Algo.png" alt="Node" style="width: 50%;">

### Inputs / Outputs
**`::image`**:
The image you want to upscale.<br>
**`::upscale_model`**:
To use with Comfy Core node **UpscaleModelLoader**.<br>
**`IMAGE::`**:
Optimized image dimensions to connect to **Divide Image and Select Tile**.<br>
**`dac_data:`**:
Data to pass along to following nodes: **Divide Image and Select Tile** and **Combine Tiles**.<br>

### Parameters

**`Scaling_method`**: The image will go through a secondary process to meet the optimal upscaled dimensions. *The default value is lanczos.*<br>
**`tile_width`**: This parameter specifies the width of each tile that the image will be divided into.*The default value is 1024 pixels.*<br>
**`tile_height`**: This parameter specifies the height of each tile that the image will be divided into.The default value is 1024 pixels.<br>
**`overlap`**: This parameter specifies the minimum amount of overlap between adjacent tiles to help blend the tiles seamlessly when they are combined back together. *The default value is '1/32 Tile', and it can range from "None" to "1/2 tile".*<br>
**`min_scale_factor`**: This parameter determines the minimal scale factor. The effective scale factor will be determined by the tile dimensions and tile overlap. *The default value is 3, and it can range from 1.01 to 8.*<br>
**`tile_order`**: This parameter specifies the order in which the tiles are processed. It can be either 'linear' or 'spiral'. 'Linear' processes the tiles in a row-by-row manner, while 'spiral' processes them in an outward clockwise spiral pattern ending at the center. *The default value is "spiral".*<br>
<img src="images/Order_Spiral.png" alt="Spiral" style="width: 25%;">
<img src="images/Order_Linear.png" alt="Linear" style="width: 25%;">

## 2. Divide Image and Select Tile | Node
Taking into account tile dimensions, tile overlap, and final image dimensions, the node calculates coordinates and divides the image into tiles.

<img src="images/DaC_Divide.png" alt="Node" style="width: 50%;">

**`::image`**:
The image you want to upscale.<br>
**`::dac_data`**:
Connect from dac_data "Divide and Conquer" node.<br>
**`::position`**:
Connect a comfyCore Primitive node,
to automate the process, select increment and remeber to start at 0<br>
**`SELECTED TILES::`**:
The tile to process next of the upscaled image.<br>
**`ALL TILES::`**:
All the tiles.<br>


## 3. Combine Tiles | Node
Combines the processed tiles back into a single image, applying a **Gaussian blur mask** on the overlapping pixels to ensure smooth transitions between the overlapping tiles.<br>
<img src="images/DaC_Combine.png" alt="Node" style="width: 50%;">

**`::image`**:
The tiles you want to combine into one upscaled image.<br>
**`::dac_data`**:
Must be connected to dac_data from "Divide and Conquer" node.<br>
**`IMAGE::`**:
The combined image, made of multiple tiles.<br>


## Terminal
Useful information is available in the terminal window:
```
Divide and Conquer algorithm:
Original Image Size: 1024x1024
Upscaled Image Size: 3008x3008
Grid: 3x3
overlap_x: 32
overlap_y: 32
effective_upscale: 2.94
```
```
Divide and Conquer matrix:
(0,0) (992,0) (1984,0)
(0,992) (992,992) (1984,992)
(0,1984) (992,1984) (1984,1984)
```

## Workflow example
Download the following workflow from
[here](Examples\Workflow\Divide_and_Conquer_Workflow_Example.json)
or 
drag and drop the workflow image into ComfyUI.

<img src="Examples\Workflow\Divide_and_Conquer_Workflow_Example.png" alt="Workflow" style="width: 100%;">

:grey_exclamation: This workflow uses the following optional nodes:<br>
[Set/Get](https://github.com/kijai/ComfyUI-KJNodes) | 
[Image Comparer](https://github.com/rgthree/rgthree-comfy) | 
[Display Any](https://github.com/rgthree/rgthree-comfy) |
[ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2)

## How to use this workflow
<details open><summary>ALGORITHM</summary>
:one: Load your image.<br>

:two: Select your prefered upscale [model](https://openmodeldb.info).<br>
:three: Choose your paramaters.<br>

<img src="Images\Group_ALGORITHM.png" alt="Workflow" style="width: 100%;">
</details>
<details open><summary>DIVIDE</summary>
:four: Reset value to "0". <br>
:five: Change control_after_generate to "increment".


<img src="Images\Group_DIVIDE.png" alt="Workflow" style="width: 100%;">
</details>
<details open><summary>CONQUER</summary>
After the image is divided into tiles, it is a "simple" img2img process.

:grey_exclamation: While any models will work, this example is using [Flux.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [Flux.1-dev-Controlnet-Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler).

Prompt is generate per tile using [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2).

:six: Each tile must be saved into an **empty folder**.


<img src="Images\Group_CONQUER.png" alt="Workflow" style="width: 100%;">

:exclamation: Ensure that only COMBINE group is muted (Set Group Nodes to Never)

:exclamation: Change Queue to Queue (Instant) and click Queue (Instant) to start the process.<br>
<img src="Images\Menu_Queue_Instant.png" alt="Workflow" style="width: 50%;"><br>
:exclamation: While not being the most elegant solution, it is working really well to stop generating after the last tile has been processed successfully. Just close the pop-up window.<br>
<img src="Images\Menu_Error.png" alt="Workflow" style="width: 50%;"><br>
</details>

<details open><summary>COMBINE</summary>
:exclamation: Change Queue (Instant) to Queue<br>
<img src="Images\Menu_Queue.png" alt="Workflow" style="width: 50%;"><br>
:exclamation:Ensure that only DIVIDE and CONQUER groups are muted (Set Group Nodes to Never)<br>
:seven: Use the same folder as in :six:<br>

A load images from folder node like [KJnodes LoadImagesFromFolderKJ](https://github.com/kijai/ComfyUI-KJNodes) is required to load the images to **Combine Tiles** node for processing.<br>
:bowtie: Enjoy your masterpiece.<br>
<img src="Images\Group_COMBINE.png" alt="Workflow" style="width: 100%;">
</details>

:100: cropped comparison:
<img src="images/DaC_Side-by-side.png" alt="Workflow" style="width: 100%;">

## TIPS
- General upscaling guidelines do apply, but the **Divide and Conquer Node Suite** offers better control per tile, enabling higher detail transfer.
- Instead of generating the entire set of tiles, a single tile can be generated as a test sample to verify your img2img parameters.
- Ensure that the input folder only contains the tiles you intend to combine.
- If seams appear in the combine image, increase the overlap.

# Installation
Install via ComfyUI-Manager or Clone this repo into `custom_modules`:

```
cd ComfyUI/custom_nodes
git clone https://github.com/Steudio/ComfyUI_Steudio.git
```

# Changelog
### 2024-12-01
`Version 1.0.0` Initial public release.

# Acknowledgements
This repository uses some code from:<br>
[ComfyUI](https://github.com/comfyanonymous/ComfyUI/) | [Comfyroll Studio](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) | [SimpleTiles](https://github.com/kinfolk0117/ComfyUI_SimpleTiles)<br>

# License
GNU GENERAL PUBLIC LICENSE Version 3, see [LICENSE](LICENSE)

# Thank you
Copyright (c) 2024, Steudio - https://github.com/steudio

