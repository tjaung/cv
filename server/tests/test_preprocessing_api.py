import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from server.tests.test_models_api import request


class PreprocessingPipelineTests(unittest.TestCase):
    def test_segmentation_preserves_original_plate_colors(self):
        from CV.preprocessing import segment_plate, preprocess_plate

        image = np.full((140, 140, 3), 220, np.uint8)
        image[20:120, 20:120] = (90, 40, 20)
        segmented = segment_plate(image)
        self.assertTrue(segmented.plate_mask.any())
        inside = segmented.plate_mask != 0
        np.testing.assert_array_equal(segmented.plate[inside], image[inside])
        self.assertFalse(segmented.plate[~inside].any())
        full = preprocess_plate(image)
        np.testing.assert_array_equal(full.plate, segmented.plate)
        np.testing.assert_array_equal(full.plate_mask, segmented.plate_mask)

    def test_anomaly_previews_normalized_color_and_unprocessed_raw(self):
        from CV.models.anomaly_detection.inputs import normalized_pipeline
        from CV.preprocessing.luminance_correction import adjust_luminance_to_middle_by_percentage
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'metal_plate/train/good/000.png'
            path.parent.mkdir(parents=True)
            image = np.full((140, 140, 3), 220, np.uint8)
            image[20:120, 20:120] = (90, 40, 20)
            cv2.imwrite(str(path), image)
            with patch('server.api.os_helpers.DATASET_ROOT', root):
                def get(pipeline, step=None):
                    query = {'image_path': 'metal_plate/train/good/000.png', 'pipeline': pipeline}
                    if step is not None:
                        query['step'] = step
                    return asyncio.run(request('/preprocessing/pipeline/', query=query))
                for step, expected in [(1, adjust_luminance_to_middle_by_percentage(image, 10)),
                                       (15, normalized_pipeline(image).final_plate)]:
                    status, body = get('normalized', step)
                    self.assertEqual(status, 200)
                    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(body, np.uint8), 1), expected)
                with patch('server.api.preprocessing.segment_plate', side_effect=AssertionError), \
                     patch('server.api.preprocessing.preprocess_plate', side_effect=AssertionError), \
                     patch('server.api.preprocessing.normalized_pipeline', side_effect=AssertionError):
                    status, body = get('raw')
                    self.assertEqual(status, 200)
                    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(body, np.uint8), 1), image)
                self.assertEqual(get('raw', 1)[0], 422)
                for pipeline, count in [('normalized', 16), ('raw', 1)]:
                    status, body = asyncio.run(request('/preprocessing/steps/', query={'pipeline': pipeline}))
                    self.assertEqual(status, 200)
                    self.assertEqual([s['number'] for s in json.loads(body)['steps']], list(range(count)))

    def test_distinct_steps_and_segmentation_only_patch_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'metal_plate/train/good/000.png'
            path.parent.mkdir(parents=True)
            raw = np.full((96, 96, 3), 80, np.uint8)
            cv2.imwrite(str(path), raw)
            mask = np.full((96, 96), 255, np.uint8)
            gray = raw[:, :, 0]
            result = SimpleNamespace(gray=gray, blurred=gray, edge_bold=gray,
                                     threshold=100., threshold_mask=mask, cleaned_mask=mask,
                                     plate_mask=mask, plate=raw)
            with patch('server.api.os_helpers.DATASET_ROOT', root):
                def get(query):
                    return asyncio.run(request('/preprocessing/pipeline/', query={'image_path': 'metal_plate/train/good/000.png', **query}))
                for pipeline, count, last in [('segmentation', 9, 'Patch extraction'), ('full', 15, 'Contrast / final plate')]:
                    status, body = asyncio.run(request('/preprocessing/steps/', query={'pipeline': pipeline}))
                    self.assertEqual(status, 200)
                    steps = json.loads(body)['steps']
                    self.assertEqual(len(steps), count)
                    self.assertEqual(steps[-1]['name'], last)
                with patch('server.api.preprocessing.segment_plate', return_value=result), \
                     patch('server.api.preprocessing.preprocess_plate', side_effect=AssertionError('Segmentation must not run glare repair')):
                    status, body = get({'pipeline': 'segmentation', 'step': 7})
                    self.assertEqual(status, 200)
                    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(body, np.uint8), 1), raw)
                    status, body = get({'pipeline': 'segmentation'})
                    self.assertEqual(status, 200)
                    preview = cv2.imdecode(np.frombuffer(body, np.uint8), 1)
                    self.assertFalse(np.array_equal(preview, raw))
                    np.testing.assert_array_equal(preview[0, 0], [0, 210, 255])
                self.assertEqual(get({'pipeline': 'segmentation', 'step': 9})[0], 422)
                self.assertEqual(get({'pipeline': 'unknown'})[0], 422)
                # Existing default endpoint keeps the full final image contract.
                self.assertEqual(get({})[0], 200)
