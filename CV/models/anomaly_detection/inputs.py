"""Explicit anomaly experiment inputs; classifier and CNN pipelines stay unchanged."""
from pathlib import Path
from typing import Literal
import hashlib

import cv2
import numpy as np

from CV.preprocessing import segment_plate
from CV.preprocessing.preprocessing import PreprocessingResult
from CV.preprocessing.luminance_correction import adjust_luminance_to_middle_by_percentage
from CV.preprocessing.apply_mask import apply_plate_mask
from CV.preprocessing.grayscale import to_grayscale
from CV.preprocessing.inpainting import glare_threshold
from CV.preprocessing.surrounding_color_fill import fill_surrounding_color
from CV.preprocessing.boundary_blending import blend_glare_boundary
from CV.preprocessing.gaussian_blur import apply_gaussian_blur
from CV.preprocessing.clahe import apply_clahe
from CV.preprocessing.contrast import increase_contrast
from CV.features import cielab_features, sobel_features, hog_from_gradients, frangi_features, frangi_descriptor
from CV.models.patch_features import PatchData, CHANNELS
from CV.models.feature_sets import extract_model_features, feature_signature, LAB_INDICES

PreprocessingVariant = Literal['full', 'normalized', 'raw']
VARIANTS = ('full', 'normalized', 'raw')


def validate_variant(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown anomaly preprocessing variant')
    return variant


def input_signature(feature_set, preprocessing='full'):
    validate_variant(preprocessing)
    base = feature_signature(feature_set)
    # Preserve compatibility for the existing, unchanged full pipeline.
    if preprocessing == 'full':
        return base
    digest = hashlib.sha256((base + preprocessing).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def normalized_pipeline(image, blur_size=5, fill_holes=True, allow_border_touching=True,
                        sobel_strength=1., threshold_offset=20, glare_cutoff=220,
                        surrounding_radius=5, run_glare_fill=True, blend_width=3, contrast_factor=1.5):
    """Historical normalization branch: derive the mask from ORIGINAL grayscale.

    Apply the mask to luminance-corrected color, then run the usual plate repair.
    This intentionally does not threshold a normalized grayscale image.
    """
    segmented = segment_plate(image, blur_size, fill_holes, allow_border_touching, sobel_strength, threshold_offset)
    normalized = adjust_luminance_to_middle_by_percentage(image, 10)
    mask = segmented.plate_mask
    plate = apply_plate_mask(normalized, mask)
    gray = to_grayscale(plate)
    glare = glare_threshold(gray, mask, glare_cutoff)
    filled = fill_surrounding_color(plate, glare, mask, surrounding_radius) if run_glare_fill else plate.copy()
    blended = blend_glare_boundary(plate, filled, glare, blend_width)
    blurred = apply_gaussian_blur(to_grayscale(blended), blur_size)
    blurred[mask == 0] = 0
    clahe = apply_clahe(blurred)
    clahe[mask == 0] = 0
    contrast = increase_contrast(clahe, contrast_factor)
    contrast[mask == 0] = 0
    color = cv2.cvtColor(blended, cv2.COLOR_BGR2YCrCb)
    color[:, :, 0] = contrast
    final = cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR)
    final[mask == 0] = 0
    return PreprocessingResult(segmented.gray, segmented.blurred, segmented.edge_bold,
                               segmented.threshold, segmented.threshold_mask, segmented.cleaned_mask,
                               mask, plate, gray, glare, filled, blended, blurred, clahe, contrast, final)


def extract_anomaly_features(image, image_id='', patch_size=64, min_coverage=.5,
                             feature_set='lab_sobel_hog_frangi', preprocessing='full'):
    validate_variant(preprocessing)
    if preprocessing == 'full':
        return extract_model_features(image, image_id, patch_size, min_coverage, feature_set)
    if not isinstance(patch_size, int) or not 16 <= patch_size <= 256 or not 0 < min_coverage <= 1:
        raise ValueError('Invalid patch size or coverage')
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty uint8 BGR image')
    if preprocessing == 'normalized':
        result = normalized_pipeline(image)
        plate, mask = result.final_plate, result.plate_mask
    else:
        # No segmentation, repair, or luminance adjustment: background is included.
        plate, mask = image.copy(), np.full(image.shape[:2], 255, np.uint8)
    return describe_input(plate, mask, image_id, patch_size, min_coverage, feature_set)


def describe_input(plate, mask, image_id, patch_size, min_coverage, feature_set):
    """Same LAB/Sobel/HOG/Frangi descriptor schema as the existing full pipeline."""
    from CV.models.feature_sets import feature_names
    feature_names(feature_set)
    valid = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    lab, sobel = cielab_features(plate), sobel_features(plate)
    maps = [lab[:, :, i] for i in range(3)] + [sobel[k] for k in ('gx', 'gy', 'magnitude')]
    ridges = frangi_features(plate, mask) if feature_set == 'lab_sobel_hog_frangi' else None
    x, y, width, height = cv2.boundingRect(mask)
    vectors, records = [], []
    for top in range(y, y + height, patch_size):
        for left in range(x, x + width, patch_size):
            right, bottom = min(left + patch_size, x + width), min(top + patch_size, y + height)
            region = np.s_[top:bottom, left:right]
            selected = valid[region]
            pixels = int(selected.sum())
            coverage = pixels / selected.size
            if pixels < 16 or coverage < min_coverage:
                continue
            channels = [matrix[region][selected] for matrix in maps]
            vector = []
            for values, (_, limits) in zip(channels, CHANNELS):
                vector.extend(np.histogram(values, bins=12, range=limits)[0] / pixels)
            magnitudes = channels[-1].astype(np.float64)
            angles = sobel['direction'][region][selected]
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
            if feature_set == 'lab_hog':
                vector = [vector[i] for i in LAB_INDICES]
            if feature_set != 'lab_sobel':
                vector.extend(hog_from_gradients(sobel['magnitude'][region], sobel['direction'][region], selected)['descriptor'])
            if ridges is not None:
                vector.extend(frangi_descriptor(ridges, region))
            vectors.append(vector)
            records.append(dict(image_id=image_id, patch_id=len(records), left=left, top=top,
                                right=right, bottom=bottom, pixels=pixels, coverage=coverage))
    if not vectors:
        raise ValueError('No patches meet minimum plate coverage')
    return PatchData(np.asarray(vectors, dtype=np.float64), records, plate, mask)
