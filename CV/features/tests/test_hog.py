import tempfile
import unittest

import numpy as np

from CV.features.hog import hog_features, hog_from_gradients, HOG_NAMES
from CV.models import PCAAnomalyDetector, OneClassSVMDetector, PatchData
from CV.models.feature_sets import extract_model_features, feature_names, feature_signature
from CV.models.patch_features import pipeline_signature, extract_patch_features


class HOGTests(unittest.TestCase):
    def test_angular_votes_normalization_mask_and_spatial_order(self):
        magnitude = np.ones((32, 32))
        direction = np.full((32, 32), 10.)
        valid = np.ones((32, 32), bool)
        result = hog_from_gradients(magnitude, direction, valid)
        self.assertEqual(result['descriptor'].shape, (324,))
        self.assertEqual(len(HOG_NAMES), 324)
        np.testing.assert_allclose(result['cells'][:, :, 0], 32.)
        np.testing.assert_allclose(result['cells'][:, :, 1], 32.)
        np.testing.assert_allclose((result['blocks'] ** 2).sum(axis=(2, 3, 4)), 1., atol=1e-8)
        opposite = hog_from_gradients(magnitude, direction + 180, valid)
        np.testing.assert_array_equal(opposite['descriptor'], result['descriptor'])
        scaled = hog_from_gradients(magnitude * 4, direction, valid)
        np.testing.assert_allclose(result['descriptor'], scaled['descriptor'])
        blank = hog_from_gradients(magnitude, direction, ~valid)
        self.assertFalse(blank['descriptor'].any())
        localized = valid.copy(); localized[:, 16:] = False
        half = hog_from_gradients(magnitude, direction, localized)
        self.assertFalse(half['cells'][:, 2:].any())
        self.assertTrue(half['cells'][:, :2].any())

    def test_flat_image_and_extractor_schema_legacy_unchanged(self):
        image = np.full((140, 140, 3), 220, np.uint8)
        flat = hog_features(image)
        self.assertFalse(flat['descriptor'].any())
        image[20:120, 20:120] = np.random.default_rng(9).integers(30, 70, (100, 100, 3), dtype=np.uint8)
        legacy = extract_patch_features(image)
        np.testing.assert_array_equal(legacy.values, extract_model_features(image).values)
        self.assertEqual(feature_signature(), pipeline_signature())
        for name, width in [('lab_hog', 366), ('lab_sobel_hog', 429)]:
            result = extract_model_features(image, feature_set=name)
            self.assertEqual(result.values.shape, (len(legacy.records), width))
            self.assertEqual(len(feature_names(name)), width)
            self.assertTrue(np.isfinite(result.values).all())
            self.assertNotEqual(feature_signature(name), pipeline_signature())

    def test_both_model_families_persist_and_export_hog_schema(self):
        rng = np.random.default_rng(3)
        for feature_set in ('lab_hog', 'lab_sobel_hog'):
            width = len(feature_names(feature_set))
            train = PatchData(rng.normal(size=(30, width)), [{'image_id': f'train/{i // 5}'} for i in range(30)])
            held = PatchData(rng.normal(size=(10, width)), [{'image_id': f'held/{i // 5}'} for i in range(10)])
            for detector in (PCAAnomalyDetector, OneClassSVMDetector):
                model = detector(feature_set=feature_set).fit(train, held)
                with tempfile.TemporaryDirectory() as directory:
                    model.save(directory)
                    loaded = detector.load(directory)
                    self.assertEqual(model.feature_names, loaded.feature_names)
                    np.testing.assert_allclose(model._project(held.values)[2], loaded._project(held.values)[2])
                    from pathlib import Path
                    self.assertIn('hog_block_0_0', (Path(directory) / 'training_features.csv').read_text().splitlines()[0])
