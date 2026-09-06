"""Model replacement with durable recipes and evaluation-only history."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from threadpoolctl import threadpool_limits
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
from pathlib import Path
import shutil

from fastapi import HTTPException
from fastapi.responses import FileResponse

from CV.models import OneClassSVMDetector, PCAAnomalyDetector
from . import models


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def recipe(report):
    config = report['config']
    value = dict(model_type=config.get('model_type', 'pca'), patch_size=config['patch_size'],
                 variance_target=config['variance_target'], feature_set='lab_sobel_hog_frangi',
                 seed=report.get('seed', 42), min_coverage=config.get('min_coverage', .5))
    if value['model_type'] == 'one_class_svm':
        value.update(kernel=config['kernel'], nu=config['nu'], gamma=config['gamma'])
    else:
        value['distance_metric'] = config.get('distance_metric', 'l2')
        if value['distance_metric'] == 'squared_l2':
            value['distance_metric'] = 'l2'
    return value


def recipe_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def catalog():
    # Full supported grid, independent of which configurations were trained
    # before. Feature subsets and equivalent legacy squared-L2 runs collapse.
    saved = {}
    for patch, variance in product((32, 64, 128), (.9, .95, .99)):
        base = dict(patch_size=patch, variance_target=variance, feature_set='lab_sobel_hog_frangi', seed=42, min_coverage=.5)
        for metric in ('l1', 'l2', 'mahalanobis'):
            value = dict(base, model_type='pca', distance_metric=metric)
            saved[recipe_key(value)] = value
        for kernel, nu in product(('rbf', 'linear'), (.01, .05, .1)):
            for gamma in (('scale', .01, .1) if kernel == 'rbf' else ('not_used',)):
                value = dict(base, model_type='one_class_svm', kernel=kernel, nu=nu, gamma=gamma)
                saved[recipe_key(value)] = value
    return saved


def history():
    return [json.loads(path.read_text()) for path in sorted((models.ARTIFACT_ROOT / 'history').glob('*.json'),
                                                          key=lambda p: p.stat().st_mtime, reverse=True)]


def archive(directory):
    report = json.loads((directory / 'report.json').read_text())
    evaluation_file = directory / 'evaluation.json'
    evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else None
    tests = []
    for path in (directory / 'tests').glob('*/result.json'):
        result = json.loads(path.read_text())
        tests.append({key: value for key, value in result.items() if key not in ('image', 'anomaly_map', 'coverage_mask')})
    snapshot = {**report, 'evaluation': evaluation, 'archived_at': datetime.now(timezone.utc).isoformat(),
                'recipe_key': recipe_key(recipe(report)), 'test_results': tests}
    _write(models.ARTIFACT_ROOT / 'history' / f'{directory.name}.json', snapshot)


def _remove(directory):
    # Only generated, validated model directories can be removed.
    if directory.parent != models.ARTIFACT_ROOT or len(directory.name) != 32 or any(c not in '0123456789abcdef' for c in directory.name):
        raise ValueError('Invalid model directory')
    shutil.rmtree(directory)


def clear_models():
    with models._lock:
        if any(job['status'] in ('queued', 'running') for job in models._jobs.values()):
            raise HTTPException(409, 'Wait for the current model job before clearing models')
        saved = catalog()
        _write(models.ARTIFACT_ROOT / 'recipes.json', saved)
        directories = [p for p in models.ARTIFACT_ROOT.iterdir() if p.is_dir() and len(p.name) == 32
                       and all(c in '0123456789abcdef' for c in p.name)]
        # Complete all snapshots before removing any fitted artifacts.
        for directory in directories:
            if (directory / 'report.json').exists():
                archive(directory)
        for directory in directories:
            _remove(directory)
        (models.ARTIFACT_ROOT / 'current.json').unlink(missing_ok=True)
        models._model.cache_clear()
    return {'removed': len(directories), 'configurations': len(saved), 'history_records': len(history())}


def retrain_all_models():
    def run(job_id):
        saved = catalog()
        if not saved:
            raise ValueError('No saved model configurations. Train a parameter grid first.')
        _write(models.ARTIFACT_ROOT / 'recipes.json', saved)
        completed, failures = [], []
        stop = Event()
        previous = {}
        for path in models.ARTIFACT_ROOT.glob('*/report.json'):
            if not (path.parent / '.pending').exists():
                previous.setdefault(recipe_key(recipe(json.loads(path.read_text()))), []).append(path.parent)
        # One worker per patch size: reuse extracted patches across its 45
        # configurations without cache races or duplicate extraction.
        groups = {}
        for key, parameters in saved.items():
            groups.setdefault(parameters['patch_size'], []).append((key, parameters))
        with models._lock:
            models._jobs[job_id].update(combination=0, combinations=len(saved), completed=0, failed=0, workers=min(3, len(groups)))
        def train_one(key, parameters, cache):
            old = previous.get(key, [])
            new_id = None
            try:
                body = models.TrainRequest(patch_size=parameters['patch_size'], variance_target=parameters['variance_target'],
                                           feature_set=parameters['feature_set'], seed=parameters['seed'])
                def factory():
                    if parameters['model_type'] == 'one_class_svm':
                        detector = OneClassSVMDetector(parameters['variance_target'], parameters['patch_size'], parameters['nu'],
                                                      parameters['kernel'], parameters['gamma'] if parameters['kernel'] == 'rbf' else 'scale',
                                                      feature_set=parameters['feature_set'])
                        detector.config['min_coverage'] = parameters['min_coverage']
                        return detector
                    return PCAAnomalyDetector(parameters['variance_target'], parameters['patch_size'], parameters['min_coverage'],
                                              parameters['distance_metric'], parameters['feature_set'])
                new_id = models._train_pca(body, job_id, cache, model_factory=factory, publish=False)['model_id']
                result = models._evaluate_pca(new_id, job_id, cache)
                if not result['images']:
                    raise ValueError('No test images could be evaluated; previous model retained')
                with models._lock:
                    for directory in old:
                        archive(directory)
                    for directory in old:
                        _remove(directory)
                    (models.ARTIFACT_ROOT / new_id / '.pending').unlink()
                    _write(models.ARTIFACT_ROOT / 'current.json', {'model_id': new_id})
                    models._model.cache_clear()
                    completed.append(new_id)
            except Exception as error:
                if isinstance(error, OSError) and error.errno == 28:
                    stop.set()
                if new_id and (models.ARTIFACT_ROOT / new_id / '.pending').exists():
                    _remove(models.ARTIFACT_ROOT / new_id)
                with models._lock:
                    failures.append({'parameters': parameters, 'error': str(error)})
            finally:
                with models._lock:
                    models._jobs[job_id].update(combination=len(completed) + len(failures), completed=len(completed), failed=len(failures))
        def train_group(items):
            cache = {}
            for key, parameters in items:
                if stop.is_set():
                    break
                train_one(key, parameters, cache)
        # Avoid three concurrent BLAS operations each taking all CPU threads.
        with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=3, thread_name_prefix='model-grid') as pool:
            list(pool.map(train_group, groups.values()))
        return {'model_ids': completed, 'failures': failures, 'stopped_early': stop.is_set()}
    return models._submit('retrain_all', run)


def download_history(model_id: str):
    if len(model_id) != 32 or any(c not in '0123456789abcdef' for c in model_id):
        raise HTTPException(400, 'Invalid model ID')
    path = models.ARTIFACT_ROOT / 'history' / f'{model_id}.json'
    if not path.exists():
        raise HTTPException(404, 'History record not found')
    return FileResponse(path, media_type='application/json', filename=f'{model_id}-results.json')
