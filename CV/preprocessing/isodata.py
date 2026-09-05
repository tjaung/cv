import numpy as np

def isodata_threshold(gray):
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError("Expected a nonempty 8-bit grayscale image")
    
    histogram = np.bincount(gray.ravel(), minlength=256)
    intensities = np.arange(256)
    threshold = float(gray.mean())

    for _ in range(256):
        lower = intensities <= threshold
        low_count = histogram[lower].sum()
        high_count = histogram[~lower].sum()
        if low_count == 0 or high_count == 0:
            break
        low_mean = np.dot(intensities[lower], histogram[lower]) / low_count
        high_mean = np.dot(intensities[~lower], histogram[~lower]) / high_count
        updated = float((low_mean + high_mean) / 2)
        if int(updated) == int(threshold):
            return updated
        threshold = updated

    return threshold


def apply_isodata(gray, threshold_offset=0):
    if not np.isfinite(threshold_offset) or not -100 <= threshold_offset <= 100:
        raise ValueError('Threshold offset must be between -100 and 100')
    
    automatic_threshold = isodata_threshold(gray)
    bright = gray > automatic_threshold
    border = np.concatenate((bright[0], bright[-1], bright[1:-1, 0], bright[1:-1, -1]))
    background_is_bright = np.count_nonzero(border) > border.size / 2
    threshold = float(np.clip(automatic_threshold + threshold_offset, 0, 255))
    bright = gray > threshold
    foreground = ~bright if background_is_bright else bright
    
    return foreground.astype(np.uint8) * 255, threshold
