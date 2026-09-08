import cv2
import numpy as np

LAB_RANGES = ((0.0, 100.0), (-128.0, 128.0), (-128.0, 128.0))
LAB_CHANNELS = ('L', 'a', 'b')


def cielab_features(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty uint8 BGR image')

    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def lab_histograms(lab, mask=None, bins=64):
    if lab.ndim != 3 or lab.shape[2] != 3 or not lab.size or not np.isfinite(lab).all():
        raise ValueError('Expected finite H×W×3 LAB matrix')
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 256:
        raise ValueError('Bins must be an integer between 2 and 256')
    if mask is not None and (mask.shape != lab.shape[:2] or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError('Expected matching binary mask')

    pixels = lab.reshape(-1, 3) if mask is None else lab[mask != 0]

    return {name: np.histogram(pixels[:, i], bins=bins, range=limits)
            for i, (name, limits) in enumerate(zip(LAB_CHANNELS, LAB_RANGES))}
