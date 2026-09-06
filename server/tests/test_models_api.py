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

    def test_svm_grid_deduplicates_linear_gamma_and_supports_inspection(self):
        status, job = self.call('/models/one_class_svm/grid/', 'POST', body={
            'patch_size': 32, 'nus': [.05], 'kernels': ['rbf', 'linear'], 'gammas': ['scale', .1]})
        self.assertEqual(status, 200)
        result = self.finish(job['job_id'])
        self.assertEqual(result['failures'], [])
        self.assertEqual(len(result['model_ids']), 3)
        _, listing = self.call('/models/')
        self.assertEqual(len(listing['models']), 3)
        for report in listing['models']:
            self.assertEqual(report['config']['model_type'], 'one_class_svm')
            self.assertEqual(report['evaluation']['images'], 2)
            _, details = self.call('/models/one_class_svm/', query={'model_id': report['model_id']})
            self.assertGreater(details['config']['support_vectors'], 0)
        model_id = result['model_ids'][0]
        _, job = self.call('/models/one_class_svm/test/', 'POST', query={'model_id': model_id}, body={'image_path': 'metal_plate/test/rust/000.png'})
        test = self.finish(job['job_id'])
        self.assertEqual(test['model_id'], model_id)
        self.assertTrue(test['anomaly_map'].startswith('data:image/png;base64,'))
        self.assertLess(test['map_min'], test['map_max'])
        _, features = self.call('/models/pca/features/', query={'model_id': model_id, 'split': 'test', 'test_id': test['test_id']})
        self.assertGreater(features['total'], 0)
        for body in ({'nus': [0]}, {'gammas': [-1]}, {'kernels': ['unknown']}):
            status, _ = self.call('/models/one_class_svm/grid/', 'POST', body=body)
            self.assertEqual(status, 422)

    def test_hog_previews_and_model_feature_sets(self):
        status, preview = self.call('/features/', query={'image_path': 'metal_plate/test/good/000.png'})
        self.assertEqual(status, 200)
        self.assertEqual({m['name'] for m in preview['maps'] if m['name'].startswith('Frangi')}, {'Frangi · dark ridges', 'Frangi · bright ridges'})
        self.assertEqual(len(preview['hog']['descriptor']), 324)
        self.assertEqual(len(preview['hog']['histogram']), 9)
        status, hist = self.call('/features/histograms/', query={'split': 'test'})
        self.assertEqual(status, 200)
        self.assertEqual(len(hist['edges']['hog']), 10)
        for key in ('frangi_dark', 'frangi_bright'):
            self.assertEqual(len(hist['edges'][key]), 65)
            for group in hist['groups']:
                self.assertAlmostEqual(sum(group['histograms'][key]), 1.)
        for group in hist['groups']:
            self.assertEqual(len(group['histograms']['hog']), 9)
        _, job = self.call('/models/pca/train/', 'POST', body={'patch_size': 32, 'feature_set': 'lab_sobel_hog_frangi'})
        pca_id = self.finish(job['job_id'])['model_id']
        _, report = self.call('/models/pca/', query={'model_id': pca_id})
        self.assertEqual(report['features'], 459)
        _, page = self.call('/models/pca/features/', query={'model_id': pca_id})
        self.assertEqual(page['columns'], report['feature_names'])
        _, job = self.call('/models/one_class_svm/grid/', 'POST', body={'patch_size': 32, 'feature_set': 'lab_sobel_hog_frangi', 'kernels': ['rbf'], 'nus': [.05], 'gammas': ['scale']})
        result = self.finish(job['job_id'])
        self.assertEqual(result['failures'], [])
        _, report = self.call('/models/pca/', query={'model_id': result['model_ids'][0]})
        self.assertEqual(report['features'], 459)
        _, job = self.call('/models/one_class_svm/test/', 'POST', query={'model_id': report['model_id']}, body={'image_path': 'metal_plate/test/rust/000.png'})
        self.assertEqual(self.finish(job['job_id'])['model_id'], report['model_id'])

    def test_complete_preprocessing_output_feeds_every_feature_path(self):
        from CV.preprocessing import preprocess_plate
        from CV.preprocessing.contrast import increase_contrast
        from CV.models.feature_sets import extract_model_features
        from server.api.features import _load
        from server.api.os_helpers import image_path
        path = image_path('metal_plate/test/rust/000.png')
        image = cv2.imread(str(path))
        result = preprocess_plate(image)
        expected = increase_contrast(result.clahe_plate, 1.5)
        expected[result.plate_mask == 0] = 0
        np.testing.assert_array_equal(result.contrast_plate, expected)
        self.assertFalse(hasattr(result, 'segmented_edges'))
        self.assertFalse(result.final_plate[result.plate_mask == 0].any())
        self.assertEqual(result.final_plate.shape, image.shape)
        self.assertTrue(np.any(result.final_plate[:, :, 0] != result.final_plate[:, :, 2]))
        np.testing.assert_array_equal(_load(path, 'preprocessed')[0], result.final_plate)
        for feature_set in ('lab_sobel', 'lab_hog', 'lab_sobel_hog', 'lab_sobel_hog_frangi'):
            patches = extract_model_features(image, feature_set=feature_set)
            np.testing.assert_array_equal(patches.plate, result.final_plate)
        _, stages = self.call('/preprocessing/steps/')
        self.assertEqual(stages['steps'][-1]['name'], 'Contrast / final plate')
        self.assertFalse(any('Canny' in s['name'] for s in stages['steps']))
        status, png = asyncio.run(request('/preprocessing/pipeline/', query={'image_path': 'metal_plate/test/rust/000.png'}))
        self.assertEqual(status, 200)
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), result.final_plate)
        status, _ = self.call('/preprocessing/pipeline/', query={'image_path': 'metal_plate/test/rust/000.png', 'contrast_factor': .5})
        self.assertEqual(status, 422)

    def test_clear_preserves_history_and_retrain_replaces_models(self):
        from server.api.model_lifecycle import catalog
        subset = {k: v for k, v in catalog().items() if v['patch_size'] == 32 and v['variance_target'] == .95
                  and ((v['model_type'] == 'pca' and v['distance_metric'] == 'l2') or
                       (v['model_type'] == 'one_class_svm' and v['kernel'] == 'rbf' and v['nu'] == .05 and v['gamma'] == 'scale'))}
        override = patch('server.api.model_lifecycle.catalog', return_value=subset)
        override.start()
        self.addCleanup(override.stop)
        # Include a duplicate configuration and both families.
        for _ in range(2):
            _, job = self.call('/models/pca/train/', 'POST', body={'patch_size': 32})
            self.finish(job['job_id'])
        _, job = self.call('/models/one_class_svm/grid/', 'POST', body={'patch_size': 32, 'kernels': ['rbf'], 'nus': [.05], 'gammas': ['scale']})
        self.finish(job['job_id'])
        _, before = self.call('/models/')
        self.assertEqual(len(before['models']), 3)
        old_ids = {r['model_id'] for r in before['models']}
        _, cleared = self.call('/models/clear/', 'POST')
        self.assertEqual(cleared['removed'], 3)
        self.assertEqual(cleared['configurations'], 2)
        _, listing = self.call('/models/')
        self.assertEqual(listing['models'], [])
        self.assertEqual(len(listing['history']), 3)
        archived_svm = next(r for r in listing['history'] if r['config'].get('model_type') == 'one_class_svm')
        self.assertEqual(archived_svm['evaluation']['images'], 2)
        status, saved = self.call('/models/history/' + archived_svm['model_id'])
        self.assertEqual(status, 200)
        self.assertEqual(saved['evaluation'], archived_svm['evaluation'])
        self.assertFalse(any((models.ARTIFACT_ROOT / i).exists() for i in old_ids))
        self.assertFalse((models.ARTIFACT_ROOT / 'current.json').exists())
        _, job = self.call('/models/retrain_all/', 'POST')
        result = self.finish(job['job_id'])
        self.assertEqual(len(result['model_ids']), 2)
        self.assertEqual(result['failures'], [])
        _, listing = self.call('/models/')
        self.assertEqual(len(listing['models']), 2)
        self.assertTrue(all(r['evaluation']['images'] == 2 for r in listing['models']))
        previous_ids = set(result['model_ids'])
        _, job = self.call('/models/retrain_all/', 'POST')
        result = self.finish(job['job_id'])
        self.assertEqual(result['failures'], [])
        _, listing = self.call('/models/')
        self.assertEqual(len(listing['models']), 2)
        self.assertEqual(len(listing['history']), 5)
        self.assertFalse(any((models.ARTIFACT_ROOT / i).exists() for i in previous_ids))
        retained_ids = {r['model_id'] for r in listing['models']}
        with patch('server.api.models._evaluate_pca', side_effect=ValueError('Evaluation failed')):
            _, job = self.call('/models/retrain_all/', 'POST')
            failed = self.finish(job['job_id'])
        self.assertEqual(len(failed['failures']), 2)
        _, listing = self.call('/models/')
        self.assertEqual({r['model_id'] for r in listing['models']}, retained_ids)
        self.assertEqual(len(listing['history']), 5)
        self.assertTrue((models.ARTIFACT_ROOT / json.loads((models.ARTIFACT_ROOT / 'current.json').read_text())['model_id']).exists())

    def test_parallel_retrain_and_failed_save_cleanup(self):
        from server.api.model_lifecycle import catalog
        subset = {k: v for k, v in catalog().items() if v['patch_size'] in (32, 64) and v['variance_target'] == .95
                  and v['model_type'] == 'pca' and v['distance_metric'] == 'l2'}
        with patch('server.api.model_lifecycle.catalog', return_value=subset):
            _, job = self.call('/models/retrain_all/', 'POST')
            result = self.finish(job['job_id'])
            self.assertEqual(len(result['model_ids']), 2)
            self.assertEqual(result['failures'], [])
            _, listing = self.call('/models/')
            self.assertEqual(len(listing['models']), 2)
            before = set(models.ARTIFACT_ROOT.iterdir())
            with patch('CV.models.PCAAnomalyDetector.save', side_effect=OSError(28, 'No space left on device')):
                _, job = self.call('/models/retrain_all/', 'POST')
                result = self.finish(job['job_id'])
            self.assertTrue(result['stopped_early'])
            self.assertEqual(set(models.ARTIFACT_ROOT.iterdir()), before)

    def test_all_parameter_catalog_and_read_only_projection(self):
        from server.api.model_lifecycle import catalog, recipe, recipe_key
        recipes = list(catalog().values())
        self.assertEqual(len(recipes), 135)
        self.assertEqual(sum(r['model_type'] == 'pca' for r in recipes), 27)
        self.assertTrue(all(r['feature_set'] == 'lab_sobel_hog_frangi' for r in recipes))
        report = {'config': {'patch_size': 64, 'variance_target': .95, 'feature_set': 'lab_hog', 'distance_metric': 'squared_l2'}}
        one = recipe_key(recipe(report))
        report['config'].update(feature_set='lab_sobel', distance_metric='l2')
        self.assertEqual(one, recipe_key(recipe(report)))
        self.assertEqual(self.call('/models/pca/train/', 'POST', body={'feature_set': 'lab_sobel'})[0], 422)
        self.assertEqual(self.call('/models/one_class_svm/grid/', 'POST', body={'feature_set': 'lab_hog'})[0], 422)
        _, job = self.call('/models/pca/train/', 'POST', body={'patch_size': 32})
        model_id = self.finish(job['job_id'])['model_id']
        status, projection = self.call('/models/projection/', query={'model_id': model_id, 'image_path': 'metal_plate/test/rust/000.png'})
        self.assertEqual(status, 200)
        self.assertEqual(projection['model_id'], model_id)
        self.assertTrue(projection['patches'])
        self.assertFalse((models.ARTIFACT_ROOT / model_id / 'tests').exists())
        self.assertEqual(self.call('/models/projection/', query={'model_id': model_id, 'image_path': '../escape.png'})[0], 400)
