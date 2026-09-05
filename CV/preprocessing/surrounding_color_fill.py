import cv2
import numpy as np


def fill_surrounding_color(image, glare_mask, plate_mask, surrounding_radius=5):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty uint8 BGR image')
    for mask in (glare_mask, plate_mask):
        if mask.shape != image.shape[:2] or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
            raise ValueError('Expected matching binary uint8 masks')
    if isinstance(surrounding_radius, bool) or not isinstance(surrounding_radius, int) or not 1 <= surrounding_radius <= 31:
        raise ValueError('Surrounding radius must be an integer between 1 and 31')
    
    selected = cv2.bitwise_and(glare_mask, plate_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
    donors = (plate_mask != 0) & (glare_mask == 0)
    result = image.copy()
    radius = surrounding_radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        left, top = max(0, x - radius), max(0, y - radius)
        right, bottom = min(image.shape[1], x + width + radius), min(image.shape[0], y + height + radius)
        region = labels[top:bottom, left:right] == label
        ring = (cv2.dilate(region.astype(np.uint8), kernel) != 0) & donors[top:bottom, left:right]
        if ring.any():
            color = np.rint(np.median(image[top:bottom, left:right][ring], axis=0)).astype(np.uint8)
            result[top:bottom, left:right][region] = color

    return result
