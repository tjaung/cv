import asyncio
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from server.api.comparison import handlers as comparison
from server.api.anomaly_detection import models
from server.api.shared import result_store as results
from server.tests.test_models_api import request


@dataclass
class Sample:
    path: str
    label: str


class ChoiceRng:
    def __init__(self, choices):
        self.choices = iter(choices)
    def random(self):
        return next(self.choices)
    def integers(self, length):
        return 0
    def choice(self, values):
        return values[0]


class SamplingTests(unittest.TestCase):
    def test_single_has_independent_class_and_transform_choices(self):
        samples = [Sample('good.png', 'good'), Sample('rust.png', 'major_rust')]
        for defect_draw, label in [(.1, 'good'), (.9, 'major_rust')]:
            for transform_draw, transformed in [(.1, False), (.9, True)]:
                selected = comparison.choose_samples(samples, 'single', 20, ChoiceRng([defect_draw, transform_draw]))
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0][0].label, label)
                self.assertEqual(bool(selected[0][1]), transformed)

    def test_batch_has_no_repeated_images(self):
        samples = [Sample(str(i), 'good') for i in range(20)]
        selected = comparison.choose_samples(samples, 'set', 12, np.random.default_rng(8))
        self.assertEqual(len({s.path for s, _ in selected}), 12)
        self.assertTrue(any(angle for _, angle in selected))
        self.assertTrue(any(not angle for _, angle in selected))

    def test_rotation_keeps_content_and_expands_canvas(self):
        image = np.full((40, 60, 3), 220, np.uint8)
        image[0:4, 0:4] = (10, 20, 30)
        result = comparison.rotate_image(image, 45)
        self.assertGreater(result.shape[0], image.shape[0])
        self.assertGreater(result.shape[1], image.shape[1])
        self.assertTrue(np.any(np.all(result == (10, 20, 30), axis=2)))
        np.testing.assert_array_equal(result[0, 0], (220, 220, 220))

    def test_business_metrics_separate_class_errors_and_failed_predictions(self):
        rows = [dict(actual='scratches', prediction='major_rust', supports_classes=True, elapsed_ms=10),
                dict(actual='good', prediction='defect', supports_classes=False, elapsed_ms=20),
                dict(actual='total_rust', prediction='good', supports_classes=True, elapsed_ms=30),
                dict(actual='good', error='No plate', elapsed_ms=40)]
        report = comparison.summarize(rows)
        self.assertEqual((report['tp'], report['tn'], report['fp'], report['fn']), (1, 0, 1, 1))
        self.assertEqual(report['errors'], 1)
        self.assertEqual(report['class_accuracy'], 0)
        self.assertEqual(report['defect_recall'], .5)
        self.assertEqual(report['mean_ms'], 20)
        self.assertIsNone(comparison.summarize([])['binary_accuracy'])


class ComparisonAPITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        dataset = self.root / 'dataset'
        for split in ('train', 'test'):
            for label, color in [('good', 70), ('major_rust', 130)]:
                directory = dataset / 'metal_plate' / split / label
                directory.mkdir(parents=True)
                cv2.imwrite(str(directory / '000.png'), np.full((80, 100, 3), color, np.uint8))
        for target, value in [('server.api.shared.os_helpers.DATASET_ROOT', dataset), ('server.api.shared.result_store.RESULTS_ROOT', self.root / 'results')]:
            patcher = patch(target, value); patcher.start(); self.addCleanup(patcher.stop)
        models._jobs.clear()
        comparison._predictors.clear()
        self.addCleanup(comparison._predictors.clear)
        self.entries = [dict(id='anomaly:'+'a'*32, family='anomaly', name='Anomaly', config={}, available=True, reason=None),
                        dict(id='cnn:'+'b'*32, family='cnn', name='CNN', config={}, available=True, reason=None)]

    def call(self, path, method='GET', body=None):
        import json
        status, data = asyncio.run(request(path, method, body=body))
        return status, json.loads(data)

    def finish(self, job_id):
        for _ in range(400):
            value = models.get_model_job(job_id)
            if value['status'] not in ('running', 'queued'):
                self.assertEqual(value['status'], 'complete', value)
                return value
            time.sleep(.025)
        self.fail('Comparison did not finish')

    def test_same_generated_dataset_is_used_by_all_models_and_saved(self):
        observed = {}
        root = self.root
        class FakePredictor:
            def __init__(self, entry):
                self.id = entry['id']; self.classification = entry['family'] == 'cnn'; self.exposure = {}
            def predict(self, image):
                # Full dataset must exist before any model gets an input.
                assert len(list((root/'results/data/comparison').glob('*/images/*.png'))) == 2
                observed.setdefault(self.id, []).append((image.shape, image.tobytes()))
                return {'prediction': 'good' if image[0,0,0] == 70 else 'major_rust' if self.classification else 'defect'}
        with patch.object(comparison, '_saved_models', return_value=self.entries), patch.object(comparison, '_model_version', return_value='v1'), patch.object(comparison, 'Predictor', FakePredictor):
            status, loaded = self.call('/comparison/load/', 'POST', {'model_ids': [m['id'] for m in self.entries]})
            self.assertEqual(status, 200)
            self.finish(loaded['job_id'])
            status, started = self.call('/comparison/run/', 'POST', {'model_ids': [m['id'] for m in self.entries], 'count': 4})
            self.assertEqual(status, 200)
            self.finish(started['job_id'])
        self.assertEqual(observed[self.entries[0]['id']], observed[self.entries[1]['id']])
        status, data = self.call('/comparison/results/'+started['run_id'])
        self.assertEqual(status, 200)
        self.assertEqual(data['status'], 'complete')
        self.assertEqual(len(data['samples']), 2)
        self.assertEqual(data['source_split'], 'test')
        self.assertTrue(all(s['original_split'] == 'test' for s in data['samples']))
        self.assertEqual(len(data['predictions']), 4)
        self.assertTrue(all(s['evaluated'] == 2 for s in data['summaries']))
        self.assertTrue(all(s['binary_accuracy'] == 1 for s in data['summaries']))
        self.assertTrue(all('random_score' not in s for s in data['samples']))
        self.assertTrue((results.data_root('comparison')/started['run_id']/'predictions.csv').exists())
        status, _ = asyncio.run(request('/comparison/images/'+started['run_id']+'/0'))
        self.assertEqual(status, 200)

    def test_loading_reuses_predictors_and_warmup_across_runs(self):
        constructed, calls = [], []
        class FakePredictor:
            def __init__(self, entry):
                constructed.append(entry['id'])
                self.classification = False
                self.exposure = {}
            def predict(self, image):
                calls.append(1)
                return {'prediction': 'good'}
        ids = [self.entries[0]['id']]
        with patch.object(comparison, '_saved_models', return_value=self.entries), patch.object(comparison, '_model_version', return_value='v1') as version, patch.object(comparison, 'Predictor', FakePredictor):
            self.assertEqual(self.call('/comparison/run/', 'POST', {'model_ids': ids})[0], 409)
            for _ in range(2):
                status, job = self.call('/comparison/load/', 'POST', {'model_ids': ids})
                self.assertEqual(status, 200)
                self.finish(job['job_id'])
                status, job = self.call('/comparison/run/', 'POST', {'model_ids': ids, 'mode': 'single'})
                self.assertEqual(status, 200)
                self.finish(job['job_id'])
                data = self.call('/comparison/results/' + job['run_id'])[1]
                self.assertTrue(all(s['original_split'] == 'test' for s in data['samples']))
            self.assertEqual(constructed, ids)
            self.assertEqual(len(calls), 3)  # One warm-up, two timed predictions.
            self.assertTrue(self.call('/comparison/models/')[1]['models'][0]['loaded'])
            version.return_value = 'v2'
            self.assertFalse(self.call('/comparison/models/')[1]['models'][0]['loaded'])
            self.assertEqual(self.call('/comparison/run/', 'POST', {'model_ids': ids})[0], 409)
            status, job = self.call('/comparison/load/', 'POST', {'model_ids': [self.entries[1]['id']]})
            self.finish(job['job_id'])
            self.assertNotIn(ids[0], comparison._predictors)

    def test_unknown_models_are_rejected_without_starting_a_job(self):
        with patch.object(comparison, '_saved_models', return_value=self.entries):
            self.assertEqual(self.call('/comparison/run/', 'POST', {'model_ids':['../../model']})[0], 422)
        self.assertFalse(models._jobs)

    def test_catalog_excludes_temporary_models(self):
        directory = self.root/'artifacts/cnn'/('a'*32)
        directory.mkdir(parents=True)
        (directory/'model.pt').touch()
        self.assertEqual(comparison._saved_models(), [])

    def test_saved_classifier_predicts_from_image_features(self):
        from CV.models.classifiers import KNNClassifier
        from CV.features.feature_sets import extract_model_features
        from CV.models.classifiers.base import FEATURE_SET
        run_id, model_id = 'c'*32, 'd'*32
        directory = results.saved_root('classifiers')/run_id
        directory.mkdir(parents=True)
        images = []
        for color in [(90,40,20),(20,50,160)]:
            image = np.full((140,140,3),220,np.uint8)
            image[20:120,20:120] = color
            images.append(image)
        model = KNNClassifier(n_neighbors=1, patch_size=32)
        features = []
        for image in images:
            values = extract_model_features(image,patch_size=32,feature_set=FEATURE_SET).values
            features.append(np.r_[values.mean(axis=0),values.std(axis=0)])
        model.fit(features,['good','major_rust']); model.save(directory/f'{model_id}.joblib')
        listing = comparison._saved_models()
        self.assertEqual(len(listing),1)
        predictor = comparison.Predictor(listing[0])
        self.assertEqual(predictor.predict(images[0])['prediction'],'good')
        self.assertEqual(predictor.predict(images[1])['prediction'],'major_rust')
