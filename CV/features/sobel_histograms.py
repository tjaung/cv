import numpy as np

def sobel_histograms(features, mask=None, bins=64):
    magnitude, direction = features['magnitude'], features['direction']
    if magnitude.ndim != 2 or not magnitude.size or direction.shape != magnitude.shape or not np.isfinite(magnitude).all() or not np.isfinite(direction).all() or (magnitude < 0).any():
        raise ValueError('Expected matching finite magnitude and direction matrices')
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 256:
        raise ValueError('Bins must be an integer between 2 and 256')
    if mask is not None and (mask.shape != magnitude.shape or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError('Expected matching binary mask')

    selected = np.ones(magnitude.shape, bool) if mask is None else mask != 0
    strengths = magnitude[selected]
    angles = np.mod(direction[selected], 180)

    return {'magnitude': np.histogram(strengths, bins=bins, range=(0, 181)),
            'orientation': np.histogram(angles, bins=bins, range=(0, 180), weights=strengths.astype(np.float64))}
