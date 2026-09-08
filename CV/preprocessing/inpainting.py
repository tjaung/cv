import cv2
import numpy as np

def glare_threshold(gray, plate_mask, threshold=220):
    if gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError('Expected nonempty uint8 grayscale image')
    _validate_mask(plate_mask, gray.shape)
    if not np.isfinite(threshold) or not 0 <= threshold <= 255:
        raise ValueError('Glare threshold must be between 0 and 255')
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(mask, plate_mask)


def _validate_mask(mask, shape):
    if mask.dtype != np.uint8 or mask.shape != shape or not np.isin(mask, [0, 255]).all():
        raise ValueError('Expected matching binary uint8 mask (0 or 255)')


def build_inpainting_mask(mask, plate_mask, patch_size=5):
    _validate_mask(mask, plate_mask.shape)
    _validate_mask(plate_mask, mask.shape)
    if not isinstance(patch_size, int) or not 1 <= patch_size <= 31:
        raise ValueError('Patch size must be an integer between 1 and 31')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    expanded = cv2.dilate(filled, np.ones((patch_size, patch_size), np.uint8))
    return cv2.bitwise_and(expanded, plate_mask)


def patch_based_inpainting(image, mask, patch_size=5, plate_mask=None):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty uint8 BGR image')
    if plate_mask is None:
        plate_mask = np.full(image.shape[:2], 255, np.uint8)

    _validate_mask(plate_mask, image.shape[:2])
    expanded = build_inpainting_mask(mask, plate_mask, patch_size)
    donors = (plate_mask != 0) & (expanded == 0)
    if not expanded.any() or not donors.any():
        return image.copy()
    
    source = image.copy()
    outside = plate_mask == 0
    if outside.any():
        _, labels = cv2.distanceTransformWithLabels(
            (~donors).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        colors = np.zeros((int(labels.max()) + 1, 3), np.uint8)
        colors[labels[donors]] = image[donors]
        source[outside] = colors[labels[outside]]
        
    repaired = cv2.inpaint(source, expanded, inpaintRadius=patch_size, flags=cv2.INPAINT_TELEA)
    result = image.copy()
    result[expanded != 0] = repaired[expanded != 0]

    return result
