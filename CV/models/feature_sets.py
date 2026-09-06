import hashlib
from pathlib import Path

import cv2
import numpy as np

from CV.features.frangi import FRANGI_NAMES, frangi_features, frangi_descriptor
from CV.features.hog import HOG_NAMES, hog_from_gradients
from CV.features.sobel import sobel_features
from .patch_features import FEATURE_NAMES, FEATURE_VERSION, extract_patch_features, pipeline_signature

FEATURE_SETS = ('lab_sobel', 'lab_hog', 'lab_sobel_hog', 'lab_sobel_hog_frangi')
LAB_INDICES = [i for i, name in enumerate(FEATURE_NAMES) if name.split('_')[0] in ('L', 'a', 'b')]


def feature_names(feature_set='lab_sobel'):
    if feature_set not in FEATURE_SETS:
        raise ValueError('Unknown feature set')
    if feature_set == 'lab_sobel':
        return list(FEATURE_NAMES)
    return ([FEATURE_NAMES[i] for i in LAB_INDICES] if feature_set == 'lab_hog' else list(FEATURE_NAMES)) + HOG_NAMES + (FRANGI_NAMES if feature_set == 'lab_sobel_hog_frangi' else [])


def feature_version(feature_set='lab_sobel'):
    if feature_set == 'lab_sobel_hog_frangi':
        return 'lab-sobel-hog-frangi-v1'
    return FEATURE_VERSION if feature_set == 'lab_sobel' else f'{feature_set}-hog4x4-l2hys-v1'


def feature_signature(feature_set='lab_sobel'):
    base = pipeline_signature()
    if feature_set == 'lab_sobel':
        return base
    feature_names(feature_set)
    digest = hashlib.sha256(base.encode())
    for path in [Path(__file__), Path(__file__).parents[1] / 'features/hog.py']:
        digest.update(path.read_bytes())
    if feature_set == 'lab_sobel_hog_frangi':
        digest.update((Path(__file__).parents[1] / 'features/frangi.py').read_bytes())
    return digest.hexdigest()


def extract_model_features(image, image_id='', patch_size=64, min_coverage=.5, feature_set='lab_sobel'):
    feature_names(feature_set)
    patches = extract_patch_features(image, image_id, patch_size, min_coverage)
    if feature_set == 'lab_sobel':
        return patches
    gradients = sobel_features(patches.plate)
    valid = cv2.erode(patches.mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    descriptors = []
    for record in patches.records:
        region = np.s_[record['top']:record['bottom'], record['left']:record['right']]
        descriptors.append(hog_from_gradients(gradients['magnitude'][region], gradients['direction'][region], valid[region])['descriptor'])
    base = patches.values[:, LAB_INDICES] if feature_set == 'lab_hog' else patches.values
    patches.values = np.column_stack((base, descriptors))
    if feature_set == 'lab_sobel_hog_frangi':
        ridges = frangi_features(patches.plate, patches.mask)
        descriptors = [frangi_descriptor(ridges, np.s_[r['top']:r['bottom'], r['left']:r['right']]) for r in patches.records]
        patches.values = np.column_stack((patches.values, descriptors))
    return patches
