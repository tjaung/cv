import cv2
import numpy as np


def morphological_cleanup(mask, fill_holes=True):
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.size == 0:
        raise ValueError('Expected a nonempty 8-bit binary mask')
    if not np.all((mask == 0) | (mask == 255)):
        raise ValueError('Binary mask values must be 0 or 255')
    
    filled = mask.copy()

    if fill_holes:
        flooded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(flooded, None, (0, 0), 255)
        holes = cv2.bitwise_not(flooded[1:-1, 1:-1])
        filled = cv2.bitwise_or(mask, holes)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(filled, kernel, iterations=1)

    return cv2.erode(dilated, kernel, iterations=1)
