import cv2
import numpy as np


def apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError('Expected a nonempty 8-bit grayscale image')
    if not np.isfinite(clip_limit) or not 0 < clip_limit <= 40:
        raise ValueError('CLAHE clip limit must be greater than 0 and at most 40')
    if len(tile_grid_size) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 64
        for size in tile_grid_size):
            raise ValueError('CLAHE tile grid must contain two integers from 1 to 64')
    
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(gray)
