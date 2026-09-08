import cv2
import numpy as np


def sobel_features(image):
    if image.dtype != np.uint8 or not image.size or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError('Expected nonempty uint8 grayscale or BGR image')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REFLECT_101)
    magnitude = np.hypot(gx, gy)

    return {'gx': gx, 'gy': gy, 'magnitude': magnitude,
            'direction': np.degrees(np.arctan2(gy, gx)), 'direction_valid': magnitude > 1e-6}
