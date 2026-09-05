import numpy as np

def increase_contrast(gray, factor=1.5):
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError('Expected a nonempty 8-bit grayscale image')
    if not np.isfinite(factor) or not 1 <= factor <= 10:
        raise ValueError('Contrast factor must be between 1 and 10')
    
    adjusted = (gray.astype(np.float32) - 127.5) * factor + 127.5

    return np.rint(np.clip(adjusted, 0, 255)).astype(np.uint8)
