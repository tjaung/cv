import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

FRANGI_NAMES = [f'frangi_{polarity}_{stat}' for polarity in ('dark', 'bright')
                for stat in ([f'hist_{i:02d}' for i in range(12)] + ['mean', 'std', 'max'])]


def frangi_features(image, mask=None, sigmas=(1., 2., 3.), beta=.5, gamma=.05):
    """Return dark/bright response maps in [0, 1] and a valid-interior mask.

    Sigmas are pixel scales. Excludes the largest Gaussian support at the plate
    boundary so the segmented black background cannot masquerade as a ridge.
    """
    if image.dtype != np.uint8 or not image.size or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError('Expected nonempty uint8 grayscale or BGR image')
    sigmas = tuple(sigmas)
    if not sigmas or not all(np.isfinite(s) and s > 0 for s in sigmas) or not np.isfinite(beta) or not np.isfinite(gamma) or beta <= 0 or gamma <= 0:
        raise ValueError('Scales, beta and gamma must be finite and positive')
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image).astype(np.float64) / 255
    valid = np.ones(gray.shape, bool) if mask is None else np.asarray(mask) != 0
    if valid.shape != gray.shape:
        raise ValueError('Mask must match image dimensions')
    radius = int(np.ceil(4 * max(sigmas)))
    valid = cv2.erode(valid.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    result = {key: np.zeros(gray.shape) for key in ('dark', 'bright')}
    for sigma in sigmas:
        # Removing the DC level avoids finite derivative-kernel leakage on flat plates.
        centered = gray - gray.mean()
        xx = gaussian_filter(centered, sigma, order=(0, 2), mode='reflect') * sigma ** 2
        yy = gaussian_filter(centered, sigma, order=(2, 0), mode='reflect') * sigma ** 2
        xy = gaussian_filter(centered, sigma, order=(1, 1), mode='reflect') * sigma ** 2
        delta = np.hypot(xx - yy, 2 * xy)
        a, b = (xx + yy - delta) / 2, (xx + yy + delta) / 2
        small, large = np.where(abs(a) <= abs(b), a, b), np.where(abs(a) <= abs(b), b, a)
        ratio = np.divide(small, large, out=np.zeros_like(small), where=abs(large) > 1e-12)
        response = np.exp(-ratio ** 2 / (2 * beta ** 2)) * (-np.expm1(-(small ** 2 + large ** 2) / (2 * gamma ** 2)))
        for key, sign in [('dark', 1), ('bright', -1)]:
            result[key] = np.maximum(result[key], np.where(valid & (sign * large > 1e-12), response, 0))
    return {**result, 'valid': valid}


def frangi_descriptor(features, region=np.s_[:, :]):
    """Normalized 12-bin histogram plus mean/std/max for each ridge polarity."""
    valid = features['valid'][region]
    vector = []
    for key in ('dark', 'bright'):
        values = features[key][region][valid]
        counts = np.histogram(values, bins=12, range=(0, 1))[0]
        vector.extend(counts / max(1, values.size))
        vector.extend([values.mean(), values.std(), values.max()] if values.size else [0., 0., 0.])
    return np.asarray(vector, dtype=np.float64)
