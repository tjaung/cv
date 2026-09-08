import cv2
import numpy as np


def apply_sobel_edge_bold(gray, strength=1.0):
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError('Expected a nonempty 8-bit grayscale image')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Sobel strength must be between 0 and 1')
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    edge_strength = np.clip(cv2.magnitude(gx, gy), 0, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_strength = cv2.dilate(edge_strength, kernel, iterations=1)
    darkened = gray.astype(np.float32) - strength * edge_strength

    return np.rint(np.clip(darkened, 0, 255)).astype(np.uint8)
