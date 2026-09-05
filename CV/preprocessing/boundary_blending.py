import cv2
import numpy as np

def blend_glare_boundary(original, filled, glare_mask, blend_width=3):
    if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3 or not original.size:
        raise ValueError('Expected nonempty uint8 BGR image')
    if filled.shape != original.shape or filled.dtype != np.uint8:
        raise ValueError('Fill image must match original')
    if glare_mask.shape != original.shape[:2] or glare_mask.dtype != np.uint8 or not np.isin(glare_mask, [0, 255]).all():
        raise ValueError('Expected matching binary uint8 mask')
    if isinstance(blend_width, bool) or not isinstance(blend_width, int) or not 0 <= blend_width <= 31:
        raise ValueError('Blend width must be an integer between 0 and 31')
    
    padded = cv2.copyMakeBorder(glare_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    alpha = np.clip(distance / max(1, blend_width), 0, 1)[..., None]
    result = original.copy()
    selected = glare_mask != 0
    blended = np.rint(original.astype(np.float32) * (1 - alpha) + filled.astype(np.float32) * alpha)
    result[selected] = np.clip(blended[selected], 0, 255).astype(np.uint8)

    return result
