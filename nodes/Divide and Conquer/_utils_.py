# _utils_.py

import numpy as np

def calculate_overlap(tile_size, overlap_fraction):
    return int(overlap_fraction * tile_size)

def create_tile_coordinates(image_width, image_height, tile_width, tile_height,
                            overlap_x, overlap_y, grid_x, grid_y, tile_order):
    offset_x = tile_width - overlap_x
    offset_y = tile_height - overlap_y

    base_tiles = []
    for row in range(grid_y):
        for col in range(grid_x):
            x = min(col * offset_x, image_width - tile_width)
            y = min(row * offset_y, image_height - tile_height)
            base_tiles.append((x, y, row, col))

    if tile_order == 1:
        spiral_tiles = []
        visited = set()
        x, y = grid_x // 2, grid_y // 2
        dx, dy = 1, 0
        layer = 1

        while len(spiral_tiles) < len(base_tiles):
            for _ in range(2):
                for _ in range(layer):
                    if 0 <= x < grid_x and 0 <= y < grid_y and (x, y) not in visited:
                        index = y * grid_x + x
                        spiral_tiles.append(base_tiles[index])
                        visited.add((x, y))
                    x += dx
                    y += dy
                dx, dy = -dy, dx
            layer += 1

        spiral_tiles.reverse()
        base_tiles = spiral_tiles

    matrix = [['--' for _ in range(grid_x)] for _ in range(grid_y)]
    for idx, (x, y, row, col) in enumerate(base_tiles):
        matrix[row][col] = f"{idx + 1} ({x},{y})"

    tiles_only = [(x, y) for (x, y, _, _) in base_tiles]
    return tiles_only, matrix

def generate_matrix_ui(matrix):
    return "Divide and Conquer Matrix:\n" + '\n'.join([' '.join(row) for row in matrix])

def generate_tile_mask_np(x, y, tile_width, tile_height, upscaled_width, upscaled_height, f_overlap_x, f_overlap_y):
    mask_np = np.zeros((tile_height, tile_width), dtype=np.float32)

    # Do not apply gaussian to tile at the edge of the image
    # 1234 Detect corners top/left top/right bottom/left bottom/right and grid >1
    if x == 0 and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[0:tile_height - f_overlap_y, 0:tile_width - f_overlap_x] = 1.0
    elif x == upscaled_width - tile_width and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[0:tile_height - f_overlap_y, f_overlap_x:tile_width] = 1.0
    elif x == 0 and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[f_overlap_y:tile_height, 0:tile_width - f_overlap_x] = 1.0
    elif x == upscaled_width - tile_width and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[f_overlap_y:tile_height, f_overlap_x:tile_width] = 1.0
    # 5678 Detect corners 3 edges and grid =1
    elif x == 0 and y == 0 and upscaled_height == tile_height:
        mask_np[0:tile_height, 0:tile_width - f_overlap_x] = 1.0
    elif x == upscaled_width - tile_width and y == 0 and upscaled_height == tile_height:
        mask_np[0:tile_height, f_overlap_x:tile_width] = 1.0
    elif x == 0 and y == 0 and upscaled_width == tile_width:
        mask_np[0:tile_height - f_overlap_y, 0:tile_width] = 1.0
    elif x == 0 and y == upscaled_height - tile_height and upscaled_width == tile_width:
        mask_np[f_overlap_y:tile_height, 0:tile_width] = 1.0
    # 9 12 Detect top or bottom edges
    elif x != 0 and x != upscaled_width - tile_width and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[0:tile_height - f_overlap_y, f_overlap_x:tile_width - f_overlap_x] = 1.0
    elif x != 0 and x != upscaled_width - tile_width and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[f_overlap_y:tile_height, f_overlap_x:tile_width - f_overlap_x] = 1.0
    # 10 11 Detect left or right edges
    elif x == 0 and y != 0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[f_overlap_y:tile_height - f_overlap_y, 0:tile_width - f_overlap_x] = 1.0
    elif x == upscaled_width - tile_width and y != 0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[f_overlap_y:tile_height - f_overlap_y, f_overlap_x:tile_width] = 1.0
    # 13 Detect top and bottom edges    
    elif x != 0 and x != upscaled_width - tile_width and y == 0 and upscaled_height == tile_height and upscaled_width != tile_width:
        mask_np[0:tile_height, f_overlap_x:tile_width - f_overlap_x] = 1.0
    # 14 Detect left and right edges
    elif x == 0 and y != 0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width == tile_width:
        mask_np[f_overlap_y:tile_height - f_overlap_y, 0:tile_width] = 1.0
    # 15 Detect not touching any edges
    elif x != 0 and x != upscaled_width - tile_width and y != 0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
        mask_np[f_overlap_y:tile_height - f_overlap_y, f_overlap_x:tile_width - f_overlap_x] = 1.0

    return mask_np