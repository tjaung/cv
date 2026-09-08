import cv2
import numpy as np

def adjust_luminance_to_middle_by_percentage(image, percentage):
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be an 8-bit BGR image")
    
    if percentage == 0:
        return image.copy()

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)
    target_luminance = 127.5
    adjustment = luminance * (percentage / 100)
    adjusted_luminance = np.where(
        luminance > target_luminance,
        np.maximum(luminance - adjustment, target_luminance),
        np.minimum(luminance + adjustment, target_luminance),
    )
    ycrcb[:, :, 0] = np.rint(adjusted_luminance).astype(np.uint8)

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


