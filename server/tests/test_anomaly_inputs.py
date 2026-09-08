import unittest
from unittest.mock import patch
import numpy as np

from CV.models.anomaly_detection.inputs import (
    describe_input, extract_anomaly_features, input_signature, normalized_pipeline,
)
from CV.features.feature_sets import FEATURE_SETS, extract_model_features, feature_signature
from CV.preprocessing import preprocess_plate, segment_plate
from CV.preprocessing.luminance_correction import adjust_luminance_to_middle_by_percentage


class AnomalyInputTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((140, 140, 3), 220, np.uint8)
        rng = np.random.default_rng(42)
        self.image[20:120, 20:120] = rng.integers(30, 120, (100, 100, 3), dtype=np.uint8)

    def test_descriptor_schema_matches_existing_extractor(self):
        result = preprocess_plate(self.image)
        for feature_set in FEATURE_SETS:
            with self.subTest(feature_set=feature_set):
                expected = extract_model_features(self.image, 'image', 32, feature_set=feature_set)
                actual = describe_input(result.final_plate, result.plate_mask, 'image', 32, .5, feature_set)
                np.testing.assert_array_equal(actual.values, expected.values)
                self.assertEqual(actual.records, expected.records)

    def test_normalization_preserves_original_segmentation_branch(self):
        original = segment_plate(self.image)
        result = normalized_pipeline(self.image)
        np.testing.assert_array_equal(result.plate_mask, original.plate_mask)
        np.testing.assert_array_equal(result.gray, original.gray)
        corrected = adjust_luminance_to_middle_by_percentage(self.image, 10)
        inside = result.plate_mask != 0
        self.assertTrue(inside.any())
        np.testing.assert_array_equal(result.plate[inside], corrected[inside])
        self.assertFalse(np.array_equal(result.plate, original.plate))
        self.assertFalse(result.final_plate[~inside].any())

    def test_raw_skips_preprocessing_and_includes_background(self):
        with patch('CV.models.anomaly_detection.inputs.normalized_pipeline', side_effect=AssertionError), \
             patch('CV.models.anomaly_detection.inputs.segment_plate', side_effect=AssertionError), \
             patch('CV.models.anomaly_detection.inputs.extract_model_features', side_effect=AssertionError):
            result = extract_anomaly_features(self.image, patch_size=32, preprocessing='raw')
        np.testing.assert_array_equal(result.plate, self.image)
        self.assertTrue((result.mask == 255).all())
        self.assertEqual((result.records[0]['left'], result.records[0]['top']), (0, 0))
        self.assertEqual(result.values.shape[1], 459)

    def test_variant_signatures_and_existing_compatibility(self):
        feature_set = 'lab_sobel_hog_frangi'
        self.assertEqual(input_signature(feature_set, 'full'), feature_signature(feature_set))
        self.assertEqual(len({input_signature(feature_set, p) for p in ('full', 'normalized', 'raw')}), 3)
        with self.assertRaises(ValueError):
            input_signature(feature_set, 'unknown')
