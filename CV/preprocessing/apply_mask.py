import cv2
import numpy as np

def apply_plate_mask(normalized_image, mask):
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != normalized_image.shape[:2]:
        raise ValueError('Mask must be uint8 and match the image height and width')
    return cv2.bitwise_and(normalized_image, normalized_image, mask=mask)
