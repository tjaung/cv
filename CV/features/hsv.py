import cv2
import numpy as np

HSV_RANGES = ((0.0, 360.0), (0.0, 100.0), (0.0, 100.0))


def hsv_features(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty uint8 BGR image')

    hsv = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] %= 360
    hsv[:, :, 1:] *= 100
    return hsv


def hsv_histograms(hsv, mask=None, bins=64):
    if hsv.ndim != 3 or hsv.shape[2] != 3 or not hsv.size or not np.isfinite(hsv).all():
        raise ValueError('Expected finite H×W×3 HSV matrix')
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 256:
        raise ValueError('Bins must be an integer between 2 and 256')
    if mask is not None and (mask.shape != hsv.shape[:2] or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError('Expected matching binary mask')

    pixels = hsv.reshape(-1, 3) if mask is None else hsv[mask != 0]
    hue_valid = (pixels[:, 1] >= 5) & (pixels[:, 2] >= 5)

    return {name: np.histogram(pixels[hue_valid, i] if i == 0 else pixels[:, i], bins=bins, range=limits)
            for i, (name, limits) in enumerate(zip(('H', 'S', 'V'), HSV_RANGES))}
