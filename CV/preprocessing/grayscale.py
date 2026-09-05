import cv2
import numpy as np

def to_grayscale(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError('Expected a nonempty 8-bit BGR image')

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
