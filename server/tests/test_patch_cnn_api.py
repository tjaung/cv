import asyncio
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from CV.models.object_detection.patch_resnet import PatchResNet
from CV.models.object_detection.multiclass_patch_resnet import MulticlassPatchResNet
from server.api.cnn import handlers as cnn
from server.api.anomaly_detection import models
from server.api.shared import result_store as results
from server.tests.test_models_api import request


class PatchCNNAPITests(unittest.TestCase):
    def test_patch_training_testing_and_overlay(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ('good', 'major_rust', 'scratches', 'total_rust'):
                for i in range(2):
                    split = 'train' if label == 'good' else 'test'
                    path = root / 'dataset/metal_plate' / split / label / f'{i}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image = np.random.default_rng(i+sum(map(ord, label))).integers(0, 255, (96, 96, 3), dtype=np.uint8)
                    cv2.imwrite(str(path), image)
                    if label != 'good':
                        mask_path = root / 'dataset/metal_plate/ground_truth' / label / f'{i}_mask.png'
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        mask = np.zeros((96, 96), np.uint8)
                        mask[3:5, 3:5] = 255
                        cv2.imwrite(str(mask_path), mask)
            with patch('server.api.shared.os_helpers.DATASET_ROOT', root / 'dataset'), \
                 patch.object(cnn, 'ARTIFACT_ROOT', root / 'artifacts'), \
                 patch.object(results, 'RESULTS_ROOT', root / 'results'), \
                 patch.object(cnn, '_device', return_value='cpu'), \
                 patch('CV.models.object_detection.patch_resnet.PatchResNet', side_effect=lambda **kwargs: PatchResNet(weights=None)), \
                 patch('CV.models.object_detection.patch_resnet.segment_plate', side_effect=lambda image: SimpleNamespace(plate=image, plate_mask=np.ones(image.shape[:2], np.uint8))):
                models._jobs.clear()
                def call(path, method='GET', body=None, query=None):
                    status, data = asyncio.run(request(path, method, query, body))
                    self.assertEqual(status, 200, data)
                    return json.loads(data)
                def finish(job):
                    for _ in range(600):
                        status = call('/models/jobs/' + job['job_id'])
                        if status['status'] in ('complete', 'failed'):
                            self.assertEqual(status['status'], 'complete', status)
                            return status['result']
                        time.sleep(.05)
                    self.fail('Job timed out')
                run = finish(call('/cnn/train/', 'POST', {'variant': 'patch', 'epochs': 1, 'batch_size': 2, 'num_workers': 0}))
                finish(call('/cnn/test/', 'POST', run))
                summary = call('/cnn/')['runs'][0]
                self.assertEqual(summary['class_names'], ['good', 'defect'])
                self.assertEqual(summary['split_counts']['test'], {'defect': 3, 'good': 1})
                self.assertEqual(sum(map(sum, summary['test']['patch_metrics']['confusion_matrix'])), 16)
                view = call('/cnn/inspect/', query={**run, 'image_index': 0})
                self.assertEqual(len(view['patches']), 4)
                self.assertTrue(view['activation_image'].startswith('data:image/png;base64,'))
                self.assertTrue((results.data_root('cnn') / run['run_id'] / 'test_patches.csv').exists())
                with patch('CV.models.object_detection.multiclass_patch_resnet.MulticlassPatchResNet', side_effect=lambda names, **kwargs: MulticlassPatchResNet(names, weights=None)):
                    multi = finish(call('/cnn/train/', 'POST', {'variant': 'patch_multiclass', 'epochs': 1, 'batch_size': 2, 'num_workers': 0}))
                    finish(call('/cnn/test/', 'POST', multi))
                    report = next(r for r in call('/cnn/')['runs'] if r['run_id'] == multi['run_id'])
                    self.assertEqual(report['class_names'], ['good', 'major_rust', 'scratches', 'total_rust'])
                    self.assertEqual(np.shape(report['test']['patch_metrics']['confusion_matrix']), (4, 4))
                    inspection = call('/cnn/inspect/', query={**multi, 'image_index': 0})
                    for row in inspection['patches']:
                        self.assertEqual(len(row['probabilities']), 4)
                        self.assertEqual(row['prediction'], report['class_names'][np.argmax(row['probabilities'])])
                        self.assertEqual(row['actual'], report['class_names'][row['target']])
                    self.assertEqual(cnn._load_model(multi['run_id']).network.fc.out_features, 4)
                models._jobs.clear()
