"""Sobel-based unsigned HOG with angular voting and L2-Hys block normalization."""
import cv2
import numpy as np

from .sobel import sobel_features

HOG_NAMES = [f'hog_block_{by}_{bx}_cell_{cy}_{cx}_bin_{angle}'
             for by in range(3) for bx in range(3)
             for cy in range(2) for cx in range(2) for angle in range(9)]


def hog_from_gradients(magnitude, direction, valid, cells=(4, 4)):
    """Fixed cell grid; 9 unsigned bins, overlapping 2x2 cells, L2-Hys.

    Gradients must be computed before cropping into patches so patch seams
    do not introduce artificial edges. Invalid/padded pixels cast no votes.
    """
    if magnitude.ndim != 2 or magnitude.shape != direction.shape or magnitude.shape != valid.shape or not magnitude.size:
        raise ValueError('Expected matching nonempty gradient and mask matrices')
    rows, cols = cells
    if rows < 2 or cols < 2:
        raise ValueError('HOG needs at least 2x2 cells')
    h, w = magnitude.shape
    yy = np.minimum(np.arange(h) * rows // h, rows - 1)
    xx = np.minimum(np.arange(w) * cols // w, cols - 1)
    cell_ids = yy[:, None] * cols + xx[None, :]
    angles = np.mod(direction.astype(np.float64), 180) / 20
    lower = np.floor(angles).astype(int) % 9
    fraction = angles - np.floor(angles)
    strength = np.where(valid, magnitude, 0).astype(np.float64)
    hist = np.zeros((rows * cols, 9), np.float64)
    np.add.at(hist, (cell_ids.ravel(), lower.ravel()), (strength * (1 - fraction)).ravel())
    np.add.at(hist, (cell_ids.ravel(), ((lower + 1) % 9).ravel()), (strength * fraction).ravel())
    hist = hist.reshape(rows, cols, 9)
    blocks = np.zeros((rows - 1, cols - 1, 2, 2, 9))
    for row in range(rows - 1):
        for col in range(cols - 1):
            block = hist[row:row + 2, col:col + 2].copy()
            block /= np.sqrt(np.sum(block ** 2) + 1e-10)
            block = np.minimum(block, .2)
            block /= np.sqrt(np.sum(block ** 2) + 1e-10)
            blocks[row, col] = block
    return {'descriptor': blocks.ravel(), 'cells': hist, 'blocks': blocks,
            'orientation': blocks.sum(axis=(0, 1, 2, 3))}


def hog_features(image, mask=None):
    """Whole-ROI HOG preview; models use the same descriptor per patch."""
    gradients = sobel_features(image)
    mask = np.full(image.shape[:2], 255, np.uint8) if mask is None else (mask != 0).astype(np.uint8) * 255
    if mask.shape != image.shape[:2]:
        raise ValueError('Mask must match image dimensions')
    valid = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    x, y, w, h = cv2.boundingRect(mask)
    bounds = (x, y, w, h)
    if not w or not h:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    result = hog_from_gradients(gradients['magnitude'][y:y+h, x:x+w], gradients['direction'][y:y+h, x:x+w], valid[y:y+h, x:x+w])
    visualization = np.zeros(image.shape[:2], np.uint8)
    strengths = np.zeros((4, 4, 9))
    for by in range(3):
        for bx in range(3):
            strengths[by:by+2, bx:bx+2] += result['blocks'][by, bx]
    maximum = max(float(strengths.max()), 1e-10)
    for row in range(4):
        for col in range(4):
            center = np.array([x + (col + .5) * w / 4, y + (row + .5) * h / 4])
            for angle in range(9):
                radians = np.radians(angle * 20)
                half = np.array([np.cos(radians), np.sin(radians)]) * min(w, h) / 12
                line = np.zeros_like(visualization)
                cv2.line(line, tuple(np.rint(center-half).astype(int)), tuple(np.rint(center+half).astype(int)),
                         round(strengths[row, col, angle] / maximum * 255), 2, cv2.LINE_AA)
                np.maximum(visualization, line, out=visualization)
    result.update(visualization=visualization, bounds=list(bounds), valid_pixels=int(valid.sum()))
    return result
