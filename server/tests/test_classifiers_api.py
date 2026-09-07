import json
import shutil
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from server.api import classifiers, models
from CV.models.classifiers import PCAClassifier


class ClassifierAPITests(unittest.TestCase):
    def test_shared_split_cv_models_and_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / 'metal_plate'
            for split in ('train', 'test'):
                for label in ('good', 'rust', 'scratch'):
                    folder = dataset / split / label
                    folder.mkdir(parents=True)
                    for i in range(10):
                        (folder / f'{i}.png').write_bytes(f'{split}-{label}-{i}'.encode())
            def extract(self, samples):
                rows = []
                for sample in samples:
                    rng = np.random.default_rng(int(sample.sha256[:8], 16))
                    rows.append(rng.normal(('good', 'rust', 'scratch').index(sample.label)*3, .1, 918))
                return np.array(rows)
            def submit(kind, fn):
                models._jobs['classifier-test'] = {'status': 'running', 'kind': kind}
                try:
                    return fn('classifier-test')
                finally:
                    del models._jobs['classifier-test']
            configs = [dict(kind='pca', variance_target=.95, metric='euclidean'),
                       dict(kind='svm', variance_target=.95, kernel='linear', C=1.),
                       dict(kind='knn', variance_target=.95, n_neighbors=3, weights='distance')]
            with patch('server.api.result_store.RESULTS_ROOT', root / 'results'), patch.object(classifiers, 'ROOT', root / 'artifacts'), patch.object(classifiers, 'folder_path', return_value=dataset), patch.object(models, '_submit', side_effect=submit), patch.object(PCAClassifier, 'extract_features', extract), patch.object(classifiers, 'configurations', return_value=configs):
                self.assertIsNone(classifiers.list_classifiers()['summary'])
                result = classifiers.train_classifiers(classifiers.TrainingRequest())
                self.assertEqual(result['failures'], [])
                self.assertEqual(result['models'], 3)
                summary = classifiers.list_classifiers()['summary']
                self.assertEqual(len(summary['train']), 48)
                self.assertEqual(len(summary['test']), 12)
                self.assertFalse({s['path'] for s in summary['train']} & {s['path'] for s in summary['test']})
                for report in summary['models']:
                    self.assertEqual(len(report['cv']['macro_f1']['folds']), 5)
                    self.assertEqual(len(report['evaluation']['predictions']), 12)
                    plot = classifiers.inspect_classifier(report['id'], 0)
                    self.assertEqual(len(plot['training']), 48)
                    self.assertEqual(plot['selected']['path'], summary['test'][0]['path'])
                    other = classifiers.inspect_classifier(report['id'], 1)
                    self.assertEqual(plot['bounds'], other['bounds'])
                    self.assertEqual(plot['training'], other['training'])
                    self.assertNotEqual(plot['selected']['path'], other['selected']['path'])
                    self.assertEqual(len(plot['regions']), 45)
                    self.assertEqual(plot['regions'], other['regions'])
                    self.assertTrue(set(label for row in plot['regions'] for label in row) <= set(summary['classes']))
                    for image in range(len(summary['test'])):
                        point = classifiers.inspect_classifier(report['id'], image)['selected']['point']
                        self.assertTrue(all(plot['bounds'][0][axis] <= point[axis] <= plot['bounds'][1][axis] for axis in (0, 1)))
                    if report['config']['kind'] == 'knn':
                        self.assertEqual(len(plot['neighbors']), 3)
                run_id = summary['run_id']
                metrics = classifiers.get_classifier_training_metrics(run_id)
                self.assertEqual(set(metrics), {r['id'] for r in summary['models']})
                for report in summary['models']:
                    self.assertEqual(len(metrics[report['id']]['predictions']), len(summary['train']))
                    self.assertEqual(metrics[report['id']], report['training_evaluation'])
                # Older artifacts get the same training metrics without refitting.
                summary_path = classifiers.ROOT / run_id / 'summary.json'
                legacy = json.loads(summary_path.read_text())
                for report in legacy['models']:
                    del report['training_evaluation']
                summary_path.write_text(json.dumps(legacy))
                with patch('CV.models.classifiers.base.ImageClassifier.fit', side_effect=AssertionError('Must not retrain')):
                    self.assertEqual(classifiers.get_classifier_training_metrics(run_id), metrics)
                with self.assertRaises(Exception):
                    classifiers.get_classifier_training_metrics('../invalid')

                training_result = classifiers.classify_review_image(run_id, 0)
                holdout_result = classifiers.classify_review_image(run_id, len(summary['train']))
                self.assertEqual(training_result['membership'], 'training')
                self.assertEqual(training_result['path'], summary['train'][0]['path'])
                self.assertEqual(holdout_result['membership'], 'holdout')
                self.assertEqual(holdout_result['path'], summary['test'][0]['path'])
                for result in (training_result, holdout_result):
                    self.assertEqual({p['model_id'] for p in result['predictions']}, {r['id'] for r in summary['models']})
                    self.assertTrue(all(p['correct'] == (p['prediction'] == result['label']) for p in result['predictions']))
                for report in summary['models']:
                    prediction = next(p for p in holdout_result['predictions'] if p['model_id'] == report['id'])
                    self.assertEqual(prediction['prediction'], report['evaluation']['predictions'][0])
                with self.assertRaises(Exception):
                    classifiers.classify_review_image('../anything', 0)
                with self.assertRaises(Exception):
                    classifiers.classify_review_image(run_id, -1)
                with self.assertRaises(Exception):
                    classifiers.classify_review_image(run_id, 60)
                curve_body = classifiers.CurveRequest(run_id=run_id, model_id=summary['models'][0]['id'])
                self.assertIsNone(classifiers.get_classifier_curves(run_id, curve_body.model_id)['curve'])
                fitted_path = classifiers.ROOT / run_id / f'{curve_body.model_id}.joblib'
                original_bytes = fitted_path.read_bytes()
                def curve_submit(kind, task):
                    models._jobs['curve-test'] = {'status': 'running', 'kind': kind}
                    task('curve-test')
                    models._jobs['curve-test']['status'] = 'complete'
                    return {'job_id': 'curve-test'}
                try:
                    with patch.object(models, '_submit', side_effect=curve_submit):
                        classifiers.train_classifier_curves(curve_body)
                    curves = classifiers.get_classifier_curves(run_id, curve_body.model_id)
                    self.assertEqual(len(curves['curve']['points']), 5)
                    self.assertEqual(curves['job']['status'], 'complete')
                    self.assertEqual(fitted_path.read_bytes(), original_bytes)
                finally:
                    models._jobs.pop('curve-test', None)
                # Invalid identifiers cannot access arbitrary artifact files.
                with self.assertRaises(Exception):
                    classifiers.inspect_classifier('../model', 0)

                # All stored predictions and charts remain usable without artifacts.
                from server.api import result_store as results
                selected_id = summary['models'][0]['id']
                results.save_selected_model(results.SaveModelRequest(family='classifiers', run_id=run_id, model_id=selected_id))
                shutil.rmtree(classifiers.ROOT)
                retained = classifiers.list_classifiers()['summary']
                self.assertEqual(len(retained['models']), 3)
                self.assertEqual(sum(r['model_saved'] for r in retained['models']), 1)
                self.assertEqual(classifiers.get_classifier_training_metrics(run_id), metrics)
                self.assertEqual(classifiers.classify_review_image(run_id, 0), training_result)
                self.assertEqual(classifiers.classify_review_image(run_id, 48), holdout_result)
                for report in retained['models']:
                    plot = classifiers.inspect_classifier(report['id'], 0)
                    self.assertEqual(len(plot['training']), 48)
                    self.assertEqual(len(plot['regions']), 45)
                self.assertEqual(len(classifiers.get_classifier_curves(run_id, selected_id)['curve']['points']), 5)
                with (results.data_root('classifiers') / run_id / 'predictions.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 180)
                self.assertTrue((results.saved_root('classifiers') / run_id / f'{selected_id}.joblib').exists())

    def test_parameter_grid(self):
        self.assertEqual(len(list(classifiers.configurations())), 42)
