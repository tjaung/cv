import cv2
import numpy as np


def lbp_features(image):
    if image.dtype != np.uint8 or not image.size or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError('Expected nonempty uint8 grayscale or BGR image')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    output = np.zeros(gray.shape, np.uint8)
    height, width = gray.shape

    if min(height, width) < 3:
        return output
    
    values = gray.astype(np.float64)
    center = values[1:-1, 1:-1]
    bits = []

    for angle in np.arange(8) * np.pi / 4:
        dx, dy = round(float(np.cos(angle)), 12), round(float(-np.sin(angle)), 12)
        x0, y0 = int(np.floor(dx)), int(np.floor(dy))
        fx, fy = dx - x0, dy - y0
        sample = np.zeros(center.shape, np.float64)
        for oy, wy in [(y0, 1 - fy), (y0 + 1, fy)]:
            for ox, wx in [(x0, 1 - fx), (x0 + 1, fx)]:
                if wx * wy:
                    sample += values[1 + oy:height - 1 + oy, 1 + ox:width - 1 + ox] * wx * wy
        bits.append(sample >= center - 1e-9)

    transitions = sum((bits[i] != bits[(i + 1) % 8]).astype(np.uint8) for i in range(8))
    ones = sum(bit.astype(np.uint8) for bit in bits)
    output[1:-1, 1:-1] = np.where(transitions <= 2, ones, 9)

    return output


def lbp_histograms(codes, plate_mask):
    if codes.ndim != 2 or codes.dtype != np.uint8 or not codes.size or (codes > 9).any():
        raise ValueError('Expected nonempty uint8 LBP codes in 0–9')
    if plate_mask.shape != codes.shape or not np.isin(plate_mask, [0, 1, 255]).all():
        raise ValueError('Expected matching binary plate mask')

    mask = (plate_mask != 0).astype(np.uint8)
    valid = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0

    if mask.any():
        x, y, width, height = cv2.boundingRect(mask)
    else:
        x = y = width = height = 0

    xs = np.linspace(x, x + width, 5).astype(int)
    ys = np.linspace(y, y + height, 5).astype(int)

    regions = []
    for row in range(4):
        for col in range(4):
            left, right, top, bottom = xs[col], xs[col + 1], ys[row], ys[row + 1]
            tile_codes = codes[top:bottom, left:right][valid[top:bottom, left:right]]
            regions.append({'row': row, 'col': col, 'bounds': [int(left), int(top), int(right), int(bottom)],
                            'counts': np.bincount(tile_codes, minlength=10)})

    return {'counts': np.bincount(codes[valid], minlength=10), 'valid': valid,
            'bounds': [x, y, x + width, y + height], 'regions': regions}
