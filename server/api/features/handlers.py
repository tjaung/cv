"""Feature previews and pooled color/gradient distributions for dataset exploration."""

import base64
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import HTTPException, Query

from CV.features import cielab_features, lab_histograms, sobel_features, hsv_features, hsv_histograms, sobel_histograms, lbp_features, lbp_histograms, hog_features, frangi_features
from CV.preprocessing import preprocess_plate
from ..shared.os_helpers import folder_path, image_path as resolve_image_path, IMAGE_EXTENSIONS

Source = Literal['original', 'preprocessed']
Split = Literal['all', 'train', 'test']
HISTOGRAM_RANGES = {'L': (0, 100), 'a': (-128, 128), 'b': (-128, 128),
                    'H': (0, 360), 'S': (0, 100), 'V': (0, 100),
                    'magnitude': (0, 181), 'orientation': (0, 180), 'lbp': (-0.5, 9.5), 'hog': (-10, 170), 'frangi_dark': (0, 1), 'frangi_bright': (0, 1)}


def _load(path, source):
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(422, 'Could not decode image')
    result = preprocess_plate(image)
    # Same ROI in both modes: changes in background area cannot bias comparison.
    return (image if source == 'original' else result.final_plate), result.plate_mask


def _png(image, interpolation=cv2.INTER_AREA):
    height, width = image.shape[:2]
    scale = min(1, 512 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=interpolation)
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise HTTPException(500, 'Could not encode feature image')
    return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')


def _map(name, values, valid, limits, palette, matrix_size):
    low, high = limits
    unit = np.clip((values - low) / (high - low), 0, 1)
    encoded = np.rint(unit * 255).astype(np.uint8)
    if palette == 'diverging':
        # BGR: blue → white → red, with zero at the midpoint.
        color = np.stack([np.minimum(2 * (1 - unit), 1), 1 - np.abs(2 * unit - 1), np.minimum(2 * unit, 1)], axis=-1)
        heat = np.rint(color * 255).astype(np.uint8)
    else:
        heat = cv2.applyColorMap(encoded, cv2.COLORMAP_HSV if palette == 'cyclic' else cv2.COLORMAP_VIRIDIS)
    heat[~valid] = 0
    encoded[~valid] = 0
    height, width = values.shape
    scale = min(1, matrix_size / max(height, width))
    ys = np.linspace(0, height - 1, max(1, round(height * scale))).astype(int)
    xs = np.linspace(0, width - 1, max(1, round(width * scale))).astype(int)
    sampled = values[np.ix_(ys, xs)]
    selected = valid[np.ix_(ys, xs)]
    matrix = [[round(float(value), 3) if keep else None for value, keep in zip(row, valid_row)]
              for row, valid_row in zip(sampled, selected)]
    pixels = values[valid]
    return {'name': name, 'range': [low, high], 'palette': palette,
            'heatmap': _png(heat), 'transformed': _png(encoded),
            'matrix': matrix, 'x': xs.tolist(), 'y': ys.tolist(),
            'stats': {'min': float(pixels.min()), 'max': float(pixels.max()), 'mean': float(pixels.mean())} if pixels.size else None}


def get_features(image_path: Annotated[str, Query(min_length=1)], source: Source = 'original',
                 matrix_size: Annotated[int, Query(ge=8, le=128)] = 64):
    path = resolve_image_path(image_path)
    image, mask = _load(path, source)
    valid = mask != 0
    lab = cielab_features(image)
    sobel = sobel_features(image)
    # Remove one pixel at the ROI boundary to avoid measuring the black cutout.
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    maps = [_map('L*', lab[:, :, 0], valid, (0, 100), 'sequential', matrix_size),
            _map('a*', lab[:, :, 1], valid, (-128, 128), 'diverging', matrix_size),
            _map('b*', lab[:, :, 2], valid, (-128, 128), 'diverging', matrix_size)]
    for name, key, limits, palette in [('Gradient X', 'gx', (-128, 128), 'diverging'),
                                     ('Gradient Y', 'gy', (-128, 128), 'diverging'),
                                     ('Magnitude', 'magnitude', (0, 181), 'sequential'),
                                     ('Direction (°)', 'direction', (-180, 180), 'cyclic')]:
        keep = interior & sobel['direction_valid'] if key == 'direction' else interior
        maps.append(_map(name, sobel[key], keep, limits, palette, matrix_size))
    ridges = frangi_features(image, mask)
    for key in ('dark', 'bright'):
        maps.append(_map(f'Frangi · {key} ridges', ridges[key], ridges['valid'], (0, 1), 'sequential', matrix_size))
    codes = lbp_features(image)
    texture = lbp_histograms(codes, mask)
    display = np.rint(codes.astype(np.float32) * (255 / 9)).astype(np.uint8)
    display[~texture['valid']] = 0
    def distribution(counts):
        pixels = int(counts.sum())
        return {'counts': counts.tolist(), 'pixels': pixels,
                'histogram': (counts / pixels).tolist() if pixels else counts.tolist()}
    lbp = {'image': _png(display, cv2.INTER_NEAREST), 'bounds': texture['bounds'], **distribution(texture['counts']),
           'regions': [{**{key: value for key, value in region.items() if key != 'counts'},
                        **distribution(region['counts'])} for region in texture['regions']]}
    hog_result = hog_features(image, mask)
    weights = hog_result['orientation']
    hog = {'image': _png(hog_result['visualization']), 'bounds': hog_result['bounds'],
           'descriptor': hog_result['descriptor'].tolist(), 'cells': hog_result['cells'].tolist(),
           'histogram': (weights / weights.sum()).tolist() if weights.sum() else weights.tolist(),
           'valid_pixels': hog_result['valid_pixels']}
    hist = lab_histograms(lab, mask)
    return {'name': path.name, 'source': source, 'width': image.shape[1], 'height': image.shape[0],
            'plate_pixels': int(valid.sum()), 'image': _png(image), 'mask': _png(mask), 'maps': maps, 'lbp': lbp, 'hog': hog,
            'histograms': {key: {'counts': counts.tolist(), 'edges': edges.tolist()} for key, (counts, edges) in hist.items()}}


@lru_cache(maxsize=512)
def _image_histogram(path, mtime_ns, size, source, bins):
    # File metadata in the cache key invalidates entries when image files change.
    image, mask = _load(Path(path), source)
    features = sobel_features(image)
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    strengths = features['magnitude'][interior]
    hist = {**lab_histograms(cielab_features(image), mask, bins),
            **hsv_histograms(hsv_features(image), mask, bins),
            **sobel_histograms(features, interior, bins)}
    texture = lbp_histograms(lbp_features(image), mask)
    hist['hog'] = (hog_features(image, mask)['orientation'], np.arange(10) * 20 - 10)
    ridges = frangi_features(image, mask)
    for key in ('dark', 'bright'):
        hist[f'frangi_{key}'] = np.histogram(ridges[key][ridges['valid']], bins=bins, range=(0, 1))
    hist['lbp'] = (texture['counts'], np.arange(11) - 0.5)
    summary = {'interior_pixels': int(strengths.size), 'magnitude_sum': float(strengths.sum(dtype=np.float64)),
               'strong_edge_pixels': int(np.count_nonzero(strengths >= 10))}
    return hist, int(np.count_nonzero(mask)), summary, np.stack([region['counts'] for region in texture['regions']])


def get_feature_histograms(source: Source = 'original', split: Split = 'all',
                           bins: Annotated[int, Query(ge=8, le=128)] = 64):
    root = folder_path('metal_plate')
    groups = {}
    skipped = []
    for part in (['train', 'test'] if split == 'all' else [split]):
        split_dir = root / part
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root)
            if len(relative.parts) < 3 or any(piece.startswith('.') for piece in relative.parts):
                continue
            label = relative.parts[1]
            group = groups.setdefault(label, {'name': label, 'images': 0, 'pixels': 0, 'splits': {},
                                               'interior_pixels': 0, 'magnitude_sum': 0.0, 'strong_edge_pixels': 0,
                                               'lbp_region_counts': np.zeros((16, 10), np.int64),
                                               'counts': {key: np.zeros(10 if key == 'lbp' else 9 if key == 'hog' else bins, np.float64) for key in HISTOGRAM_RANGES}})
            try:
                safe = resolve_image_path('metal_plate', str(relative))
                stat = safe.stat()
                hist, pixels, summary, lbp_regions = _image_histogram(str(safe), stat.st_mtime_ns, stat.st_size, source, bins)
                if pixels == 0:
                    skipped.append({'path': str(relative), 'reason': 'No plate found'})
                    continue
            except (HTTPException, OSError) as error:
                skipped.append({'path': str(relative), 'reason': error.detail if isinstance(error, HTTPException) else 'Could not read image'})
                continue
            group['lbp_region_counts'] += lbp_regions
            group['images'] += 1
            group['pixels'] += pixels
            for key, value in summary.items():
                group[key] += value
            group['splits'][part] = group['splits'].get(part, 0) + 1
            for key in group['counts']:
                group['counts'][key] += hist[key][0]
    output = []
    for label in sorted(groups, key=lambda value: (value != 'good', value)):
        group = groups[label]
        group['lbp_regions'] = [{'pixels': int(counts.sum()),
                                  'histogram': (counts / counts.sum()).tolist() if counts.sum() else counts.tolist()}
                                 for counts in group.pop('lbp_region_counts')]
        counts_by_channel = group.pop('counts')
        group['totals'] = {key: float(counts.sum()) for key, counts in counts_by_channel.items()}
        group['histograms'] = {key: (counts / counts.sum()).tolist() if counts.sum() else counts.tolist()
                               for key, counts in counts_by_channel.items()}
        n = group['interior_pixels']
        group['sobel_summary'] = {'mean_magnitude': group.pop('magnitude_sum') / n if n else None,
                                  'strong_edge_fraction': group.pop('strong_edge_pixels') / n if n else None}

        output.append(group)
    return {'source': source, 'split': split, 'weighting': 'pooled valid pixels per channel; orientation weighted by gradient magnitude; HOG pools block-normalized orientation weights; nonempty curves sum to 1',
            'edges': {key: np.linspace(*limits, 11 if key == 'lbp' else 10 if key == 'hog' else bins + 1).tolist() for key, limits in HISTOGRAM_RANGES.items()},
            'groups': output, 'skipped': skipped}
