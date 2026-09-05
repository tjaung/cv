import cv2
import numpy as np


def extract_plate_mask(mask, fill_holes=True, allow_border_touching=False):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    candidates = np.arange(1, count)
    if not allow_border_touching:
        candidates = np.setdiff1d(candidates, border_labels)
    if candidates.size == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    
    largest = candidates[np.argmax(stats[candidates, cv2.CC_STAT_AREA])]
    mask = (labels == largest).astype(np.uint8) * 255

    if fill_holes:
        flooded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(flooded, None, (0, 0), 255)
        holes = cv2.bitwise_not(flooded[1:-1, 1:-1])
        mask = cv2.bitwise_or(mask, holes)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    return cv2.erode(mask, kernel, iterations=3,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
