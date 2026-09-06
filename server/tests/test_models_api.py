import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

import cv2
import numpy as np

from server.main import app
from server.api import models


async def request(path, method='GET', query=None, body=None):
    messages = []
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(body or {}).encode(), 'more_body': False}
    async def send(message):
        messages.append(message)
    await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method,
               'scheme': 'http', 'path': path, 'raw_path': path.encode(), 'query_string': urlencode(query or {}).encode(),
               'root_path': '', 'headers': [(b'content-type', b'application/json')],
               'server': ('test', 80), 'client': ('client', 1234)}, receive, send)
    start = next(m for m in messages if m['type'] == 'http.response.start')
    data = b''.join(m.get('body', b'') for m in messages)
    return start['status'], data


class ModelAPITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        dataset = root / 'dataset'
        rng = np.random.default_rng(5)
        for i in range(8):
            image = np.full((140, 140, 3), 220, np.uint8)
            image[20:120, 20:120] = np.clip(rng.normal((90 + i, 40, 20), 3, (100, 100, 3)), 0, 255).astype(np.uint8)
            path = dataset / 'metal_plate/train/good' / f'{i:03}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
        for label in ['good', 'rust']:
            path = dataset / 'metal_plate/test' / label / '000.png'
            path.parent.mkdir(parents=True)
            sample = image.copy()
            if label == 'rust':
                sample[40:100, 40:100] = (20, 50, 160)
            cv2.imwrite(str(path), sample)
        for target, value in [('server.api.os_helpers.DATASET_ROOT', dataset), ('server.api.models.ARTIFACT_ROOT', root / 'models')]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        models._jobs.clear()
        models._model.cache_clear()
        self.addCleanup(models._model.cache_clear)

    def call(self, path, method='GET', query=None, body=None):
        status, data = asyncio.run(request(path, method, query, body))
        return status, json.loads(data)

    def finish(self, job_id):
        for _ in range(200):
            _, job = self.call('/models/jobs/' + job_id)
            if job['status'] not in ('queued', 'running'):
                self.assertEqual(job['status'], 'complete', job)
                return job['result']
            time.sleep(.025)
        self.fail('Job did not finish')

    def test_training_scoring_exports_and_evaluation(self):
        status, empty = self.call('/models/')
        self.assertEqual(status, 200)
        self.assertEqual(empty['models'], [])
        status, job = self.call('/models/pca/train/', 'POST', body={'patch_size': 32})
        self.assertEqual(status, 200)
        trained = self.finish(job['job_id'])
        model_id = trained['model_id']
        status, report = self.call('/models/pca/', query={'model_id': model_id})
        self.assertEqual(status, 200)
        self.assertEqual(report['config']['quantile'], .99)
        weights = np.asarray(report['component_weights'])
        self.assertEqual(weights.shape, (report['components'], report['features']))
        np.testing.assert_allclose((weights ** 2).sum(axis=1), 1.0)
        self.assertEqual(report['training_images'] + report['calibration_images'], 8)
        train_ids = {p['image_id'] for p in report['training_points']}
        held_ids = {p['image_id'] for p in report['calibration_points']}
        self.assertFalse(train_ids & held_ids)
        _, page = self.call('/models/pca/features/', query={'view': 'pca', 'limit': 2})
        self.assertEqual(len(page['rows']), 2)
        self.assertEqual(len(page['columns']), report['components'])
        _, job = self.call('/models/pca/test/', 'POST', body={'image_path': 'metal_plate/test/rust/000.png'})
        result = self.finish(job['job_id'])
        self.assertEqual(result['membership'], 'unseen')
        self.assertGreater(len(result['patches']), 0)
        self.assertTrue(result['anomaly_map'].startswith('data:image/png;base64,'))
        _, page = self.call('/models/pca/features/', query={'split': 'test', 'test_id': result['test_id'], 'view': 'reconstructed'})
        self.assertEqual(len(page['columns']), report['features'])
        status, csv_data = asyncio.run(request('/models/pca/download/', query={'file': 'test', 'test_id': result['test_id']}))
        self.assertEqual(status, 200)
        self.assertIn(b'direction_cos', csv_data)
        self.assertIn(b'reconstructed_', csv_data)
        _, job = self.call('/models/pca/evaluate/', 'POST')
        evaluation = self.finish(job['job_id'])
        self.assertEqual(evaluation['images'], 2)
        self.assertEqual(evaluation['tp'] + evaluation['tn'] + evaluation['fp'] + evaluation['fn'], 2)
        _, overview = self.call('/models/')
        self.assertEqual(overview['models'][0]['evaluation']['images'], 2)
        self.assertEqual(self.call('/models/pca/', query={'model_id': '../escape'})[0], 400)
        self.assertEqual(self.call('/models/pca/test/', 'POST', body={'image_path': '../escape.png'})[0], 400)
        self.assertEqual(self.call('/models/pca/train/', 'POST', body={'patch_size': 1})[0], 422)

    def test_sweep_keeps_every_run_and_evaluates_each_metric(self):
        status, job = self.call('/models/pca/sweep/', 'POST', body={
            'patch_sizes': [32, 64], 'variance_targets': [.9],
            'distance_metrics': ['l1', 'l2', 'mahalanobis']})
        self.assertEqual(status, 200)
        result = self.finish(job['job_id'])
        self.assertEqual(result['failures'], [])
        self.assertEqual(len(result['model_ids']), 6)
        _, listing = self.call('/models/')
        self.assertEqual(len(listing['models']), 6)
        self.assertEqual({r['config']['distance_metric'] for r in listing['models']}, {'l1', 'l2', 'mahalanobis'})
        for report in listing['models']:
            self.assertEqual(report['evaluation']['images'], 2)
            _, details = self.call('/models/pca/', query={'model_id': report['model_id']})
            self.assertEqual(details['config'], report['config'])
            self.assertEqual(details['plate_threshold'], report['plate_threshold'])
        status, _ = self.call('/models/pca/sweep/', 'POST', body={'distance_metrics': ['invalid']})
        self.assertEqual(status, 422)
        _, job = self.call('/models/pca/evaluate_all/', 'POST')
        self.assertEqual(len(self.finish(job['job_id'])['model_ids']), 6)
