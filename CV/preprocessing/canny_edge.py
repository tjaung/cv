import cv2
import numpy as np

def apply_canny_edge_detection(image, low_threshold=100, high_threshold=200):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError('Expected a nonempty 8-bit grayscale image')
    
    return cv2.Canny(image, low_threshold, high_threshold)