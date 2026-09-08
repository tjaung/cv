import asyncio
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

import cv2
import numpy as np
import torch

from server.api.cnn import handlers as cnn

from server.api.anomaly_detection import models

from server.api.shared import result_store as results
from server.tests.test_models_api import request
from CV.models.object_detection.resnet import PlateResNet


class CNNAPITests(unittest.TestCase):
    def test_train_test_inspect_and_retain_results(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            for split in ('train', 'test'):
                for i, label in enumerate(('good', 'partial_rust', 'scratches', 'total_rust')):
                    path = dataset / 'metal_plate' / split / label / '000.png'
                    path.parent.mkdir(parents=True)
                    image = np.random.default_rng(i + (10 if split == 'test' else 0)).integers(20, 200, (48, 48, 3), dtype=np.uint8)
                    cv2.imwrite(str(path), image)
            with patch('server.api.shared.os_helpers.DATASET_ROOT', dataset), \
                 patch.object(cnn, 'ARTIFACT_ROOT', root / 'artifacts'), \
                 patch.object(results, 'RESULTS_ROOT', root / 'results'), \
                 patch.object(cnn, '_device', return_value='cpu'), \
                 patch('CV.models.object_detection.resnet.PlateResNet', side_effect=lambda names, **kwargs: PlateResNet(names, weights=None)), \
                 patch('CV.models.object_detection.resnet.segment_plate', side_effect=lambda image: SimpleNamespace(plate=image, plate_mask=np.ones(image.shape[:2], np.uint8))):
                models._jobs.clear()
                def call(path, method='GET', query=None, body=None):
                    status, data = asyncio.run(request(path, method, query, body))
                    self.assertEqual(status, 200, data)
                    return json.loads(data)
                def finish(job_id):
                    for _ in range(600):
                        job = call('/models/jobs/' + job_id)
                        if job['status'] in ('complete', 'failed'):
                            self.assertEqual(job['status'], 'complete', job)
                            return job['result']
                        time.sleep(.05)
                    self.fail('CNN job timed out')
                self.assertEqual(call('/cnn/')['runs'], [])
                job = call('/cnn/train/', 'POST', body={'epochs': 1, 'batch_size': 2, 'num_workers': 0})
                run_id = finish(job['job_id'])['run_id']
                summary = call('/cnn/')['runs'][0]
                self.assertIsNone(summary['test'])
                self.assertEqual(len(summary['class_names']), 4)
                self.assertEqual(len(summary['history']), 1)
                weights = call('/cnn/weights/', query={'run_id': run_id})
                self.assertEqual(np.shape(weights['classifier_weights']), (4, 512))
                job = call('/cnn/test/', 'POST', body={'run_id': run_id})
                finish(job['job_id'])
                summary = call('/cnn/')['runs'][0]
                self.assertEqual(sum(map(sum, summary['test']['confusion_matrix'])), 4)
                predictions = call('/cnn/predictions/', query={'run_id': run_id})
                self.assertEqual(len(predictions), 4)
                view = call('/cnn/inspect/', query={'run_id': run_id, 'image_index': 0})
                self.assertEqual(view['path'], predictions[0]['path'])
                self.assertTrue(view['input_image'].startswith('data:image/png;base64,'))
                self.assertEqual(len(view['features']), 512)
                self.assertEqual(len(view['feature_maps']), 8)
                self.assertTrue((results.data_root('cnn') / run_id / 'test_predictions.csv').exists())
                # Selected checkpoint copies and all display data survive cleanup.
                self.assertTrue(call('/cnn/save/', 'POST', body={'run_id': run_id})['saved'])
                shutil.rmtree(cnn.ARTIFACT_ROOT)
                self.assertEqual(call('/cnn/inspect/', query={'run_id': run_id, 'image_index': 0}), view)
                self.assertTrue(call('/cnn/')['runs'][0]['model_saved'])
                self.assertEqual(cnn._load_model(run_id).class_names, tuple(summary['class_names']))
                shutil.rmtree(results.saved_root('cnn'))
                self.assertFalse(call('/cnn/')['runs'][0]['model_available'])
                self.assertEqual(call('/cnn/predictions/', query={'run_id': run_id}), predictions)
                status, _ = asyncio.run(request('/cnn/test/', 'POST', body={'run_id': run_id}))
                self.assertEqual(status, 410)
                status, _ = asyncio.run(request('/cnn/weights/', query={'run_id': '../escape'}))
                self.assertEqual(status, 400)
                # The second variant retains its own run and uses the same split.
                job = call('/cnn/train/', 'POST', body={'epochs': 1, 'batch_size': 2, 'num_workers': 0, 'variant': 'defect_weighted'})
                weighted_id = finish(job['job_id'])['run_id']
                weighted = next(r for r in call('/cnn/')['runs'] if r['run_id'] == weighted_id)
                self.assertEqual(weighted['config']['variant'], 'defect_weighted')
                self.assertEqual(weighted['class_weights'], [1., 2., 2., 2.])
                self.assertEqual(weighted['split_counts'], summary['split_counts'])
                self.assertEqual(results.read_json(results.data_root('cnn') / weighted_id / 'split.json'),
                                 results.read_json(results.data_root('cnn') / run_id / 'split.json'))
                self.assertEqual(len(call('/cnn/')['runs']), 2)
                models._jobs.clear()
