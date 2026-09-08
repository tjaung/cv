from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np

from CV.features import cielab_features, sobel_features
from CV.preprocessing import preprocess_plate

CHANNELS = [('L', (0, 100)), ('a', (-128, 128)), ('b', (-128, 128)),
            ('gx', (-128, 128)), ('gy', (-128, 128)), ('magnitude', (0, 181))]
FEATURE_NAMES = [f'{name}_hist_{i:02d}' for name, _ in CHANNELS for i in range(12)]
FEATURE_NAMES += [f'direction_hist_{i:02d}' for i in range(18)]
FEATURE_NAMES += [f'{name}_{stat}' for name, _ in CHANNELS for stat in ('mean', 'std')]
FEATURE_NAMES += ['strong_edge_fraction', 'direction_cos', 'direction_sin']
FEATURE_VERSION = 'lab-sobel-patches-v1'


@dataclass
class PatchData:
    values: np.ndarray
    records: list[dict]
    plate: np.ndarray | None = None
    mask: np.ndarray | None = None


def pipeline_signature():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    sources = sorted((root / 'preprocessing').glob('*.py'))
    sources += [root / 'features/cielab.py', root / 'features/sobel.py', Path(__file__)]
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    return digest.hexdigest()


def extract_patch_features(image, image_id='', patch_size=64, min_coverage=0.5):
    if not isinstance(patch_size, int) or not 16 <= patch_size <= 256:
        raise ValueError('Patch size must be an integer from 16 to 256')
    if not 0 < min_coverage <= 1:
        raise ValueError('Coverage must be in (0, 1]')

    result = preprocess_plate(image)
    plate, mask = result.final_plate, result.plate_mask
    valid = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0

    if not valid.any():
        raise ValueError('No valid plate region found')

    lab = cielab_features(plate)
    sobel = sobel_features(plate)
    maps = [lab[:, :, i] for i in range(3)] + [sobel[key] for key in ('gx', 'gy', 'magnitude')]
    x, y, width, height = cv2.boundingRect(mask)

    vectors, records = [], []
    for top in range(y, y + height, patch_size):
        for left in range(x, x + width, patch_size):
            right, bottom = min(left + patch_size, x + width), min(top + patch_size, y + height)
            selected = valid[top:bottom, left:right]
            pixels = int(selected.sum())
            coverage = pixels / selected.size
            if pixels < 16 or coverage < min_coverage:
                continue
            channels = [matrix[top:bottom, left:right][selected] for matrix in maps]

            vector = []
            for values, (_, limits) in zip(channels, CHANNELS):
                vector.extend(np.histogram(values, bins=12, range=limits)[0] / pixels)
            magnitudes = channels[-1].astype(np.float64)
            angles = sobel['direction'][top:bottom, left:right][selected]
            histogram = np.histogram(angles, bins=18, range=(-180, 180), weights=magnitudes)[0]
            total = float(histogram.sum())
            histogram = histogram / total if total else histogram
            vector.extend(.5 * histogram + .25 * np.roll(histogram, 1) + .25 * np.roll(histogram, -1))

            for values in channels:
                vector.extend([float(values.mean()), float(values.std())])
            radians = np.radians(angles)
            vector.extend([float(np.mean(magnitudes >= 10)),
                           float(np.sum(magnitudes * np.cos(radians)) / total) if total else 0,
                           float(np.sum(magnitudes * np.sin(radians)) / total) if total else 0])
            vectors.append(vector)
            records.append({'image_id': image_id, 'patch_id': len(records), 'left': left, 'top': top,
                            'right': right, 'bottom': bottom, 'pixels': pixels, 'coverage': coverage})
    if not vectors:
        raise ValueError('No patches meet minimum plate coverage')

    return PatchData(np.asarray(vectors, dtype=np.float64), records, plate, mask)
