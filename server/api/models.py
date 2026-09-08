"""Asynchronous PCA training/evaluation and inspectable model artifacts."""

from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import hashlib
import json
import math
import shutil
from itertools import product
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
import uuid

import cv2
import numpy as np
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from CV.models import OneClassSVMDetector, PCAAnomalyDetector, PatchData, extract_patch_features
from CV.models.anomaly_detection.inputs import PreprocessingVariant, input_signature, extract_anomaly_features
from .os_helpers import folder_path, image_path, IMAGE_EXTENSIONS
from .features import _png

from . import result_store as results

ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / 'artifacts/models/pca'
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pca')
_jobs = {}
_lock = Lock()


class TrainRequest(BaseModel):
    preprocessing: PreprocessingVariant = 'full'
    patch_size: int = Field(default=64, ge=16, le=256)
    variance_target: float = Field(default=.95, gt=0, lt=1)
    seed: int = Field(default=42, ge=0)
    distance_metric: Literal['l1', 'l2', 'mahalanobis'] = 'l2'
    feature_set: Literal['lab_sobel_hog_frangi'] = 'lab_sobel_hog_frangi'


class SweepRequest(BaseModel):
    preprocessing: PreprocessingVariant = 'full'
    feature_set: Literal['lab_sobel_hog_frangi'] = 'lab_sobel_hog_frangi'
    patch_sizes: list[Literal[32, 64, 128]] = Field(default=[32, 64, 128], min_length=1, max_length=3)
    variance_targets: list[Literal[.9, .95, .99]] = Field(default=[.9, .95, .99], min_length=1, max_length=3)
    distance_metrics: list[Literal['l1', 'l2', 'mahalanobis']] = Field(default=['l1', 'l2', 'mahalanobis'], min_length=1, max_length=3)
    seed: int = Field(default=42, ge=0)


class TestRequest(BaseModel):
    image_path: str = Field(min_length=1)


def _directory(model_id=None):
    if model_id is None:
        pointer = ARTIFACT_ROOT / 'current.json'
        if not pointer.exists():
            raise HTTPException(404, 'Train a PCA model first')
        model_id = json.loads(pointer.read_text())['model_id']
    if len(model_id) != 32 or any(c not in '0123456789abcdef' for c in model_id):
        raise HTTPException(400, 'Invalid model ID')
    directory = ARTIFACT_ROOT / model_id
    if not (directory / 'model.json').exists():
        directory = results.saved_root('anomaly') / model_id
    if not (directory / 'model.json').exists():
        raise HTTPException(404, 'Model not found')
    return directory


@lru_cache(maxsize=3)
def _model(directory):
    metadata = json.loads((Path(directory) / 'model.json').read_text())
    detector = OneClassSVMDetector if metadata['config'].get('model_type') == 'one_class_svm' else PCAAnomalyDetector
    return detector.load(directory)


def _progress(job_id, phase, done=0, total=0):
    with _lock:
        _jobs[job_id].update(phase=phase, done=done, total=total)


def _submit(kind, task):
    with _lock:
        if any(job['status'] in ('queued', 'running') for job in _jobs.values()):
            raise HTTPException(409, 'A model job is already running')
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {'job_id': job_id, 'kind': kind, 'status': 'queued', 'phase': 'Queued', 'done': 0, 'total': 0}
    def run():
        with _lock:
            _jobs[job_id]['status'] = 'running'
        try:
            result = task(job_id)
            with _lock:
                _jobs[job_id].update(status='complete', result=result, phase='Complete')
        except Exception as error:
            with _lock:
                _jobs[job_id].update(status='failed', error=str(error.detail if isinstance(error, HTTPException) else error))
    _executor.submit(run)
    return {'job_id': job_id}


def get_model_job(job_id: str):
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(404, 'Job not found; server may have restarted')
        return dict(_jobs[job_id])


def get_models():
    with _lock:
        active = next((dict(job) for job in _jobs.values() if job['status'] in ('queued', 'running')), None)
    with _lock:
        reports = []
        for file in sorted(ARTIFACT_ROOT.glob('*/report.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            directory = file.parent
            if (directory / '.pending').exists() or not (directory / 'model.npz').exists():
                continue
            report = json.loads(file.read_text())
            evaluation = json.loads((directory / 'evaluation.json').read_text()) if (directory / 'evaluation.json').exists() else None
            if 'explained_variance_ratio' not in report:
                with np.load(directory / 'model.npz', allow_pickle=False) as arrays:
                    report['explained_variance_ratio'] = arrays['explained_ratio'].tolist()
            report['compatible'] = report['config'].get('pipeline_signature') == input_signature(report['config'].get('feature_set', 'lab_sobel'), report['config'].get('preprocessing', 'full'))
            reports.append({**report, 'evaluation': evaluation})
        from .model_lifecycle import catalog, history
        seen = {r['model_id'] for r in reports}
        for path in results.data_root('anomaly').glob('*/report.json'):
            if path.parent.name in seen or (ARTIFACT_ROOT / path.parent.name / '.pending').exists():
                continue
            report = results.read_json(path)
            evaluation = path.parent / 'evaluation.json'
            reports.append({**report, 'evaluation': results.read_json(evaluation) if evaluation.exists() else None, 'compatible': False})
        for report in reports:
            model_id = report['model_id']
            report['results_saved'] = (results.data_root('anomaly') / model_id / 'detail.json.gz').exists()
            report['model_saved'] = (results.saved_root('anomaly') / model_id / 'model.npz').exists()
            report['model_available'] = report['model_saved'] or (ARTIFACT_ROOT / model_id / 'model.npz').exists()
        return {'models': reports, 'active_job': active, 'history': history(), 'retrain_configurations': len(catalog())}


def get_pca_model(model_id: str | None = None):
    if model_id:
        results.validate_id(model_id)
        snapshot = results.data_root('anomaly') / model_id / 'detail.json.gz'
        if snapshot.exists():
            return results.read_json(snapshot)
    directory = _directory(model_id)
    model = _model(str(directory))
    return {**model.summary(), **json.loads((directory / 'report.json').read_text()),
            'component_weights': model.components.tolist()}


def _train_pca(body: TrainRequest, job_id=None, feature_cache=None, model_factory=None, publish=True):
    root = folder_path('metal_plate', 'train', 'good')
    paths = [image_path('metal_plate', 'train', 'good', str(path.relative_to(root))) for path in sorted(root.rglob('*'))
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not any(p.startswith('.') for p in path.relative_to(root).parts)]
    if len(paths) < 5:
        raise HTTPException(422, 'Need at least five good training images for image-disjoint calibration')
    def train(job_id):
        model = model_factory() if model_factory else PCAAnomalyDetector(body.variance_target, body.patch_size, distance_metric=body.distance_metric, feature_set=body.feature_set, preprocessing=body.preprocessing)
        groups = {}
        for path in paths:
            groups.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(path)
        if len(groups) < 5:
            raise ValueError('Need at least five distinct good images; identical files stay in the same split')
        keys = sorted(groups)
        shuffled = np.random.default_rng(body.seed).permutation(len(keys))
        calibration_keys = {keys[i] for i in shuffled[:max(2, math.ceil(.2 * len(keys))) ]}
        values = {'training': [], 'calibration': []}
        records = {'training': [], 'calibration': []}
        manifest, skipped = [], []
        done = 0
        for key in keys:
            split = 'calibration' if key in calibration_keys else 'training'
            for path in groups[key]:
                name = 'metal_plate/train/good/' + path.relative_to(root).as_posix()
                _progress(job_id, 'Extracting good-image patches', done, len(paths))
                image = cv2.imread(str(path))
                try:
                    if image is None:
                        raise ValueError('Could not decode image')
                    cache_key = (name, body.patch_size, body.feature_set, model.config['min_coverage'], body.preprocessing)
                    patches = feature_cache.get(cache_key) if feature_cache is not None else None
                    if patches is None:
                        patches = extract_anomaly_features(image, name, body.patch_size, min_coverage=model.config['min_coverage'], feature_set=body.feature_set, preprocessing=body.preprocessing)
                        if feature_cache is not None:
                            feature_cache[cache_key] = PatchData(patches.values, patches.records)
                except ValueError as error:
                    skipped.append({'image_id': name, 'reason': str(error)})
                    done += 1
                    continue
                values[split].append(patches.values)
                records[split].extend(patches.records)
                manifest.append({'image_id': name, 'split': split, 'sha256': key, 'patches': len(patches.values)})
                done += 1
        if not values['training'] or len({r['image_id'] for r in records['calibration']}) < 2:
            raise ValueError('Not enough usable training/calibration images after segmentation')
        _progress(job_id, 'Fitting scaler/PCA and calibrating the 99th percentile', done, len(paths))
        model.fit(PatchData(np.vstack(values['training']), records['training']),
                  PatchData(np.vstack(values['calibration']), records['calibration']))
        model_id = uuid.uuid4().hex
        directory = ARTIFACT_ROOT / model_id
        directory.mkdir(parents=True, exist_ok=True)
        if not publish:
            (directory / '.pending').touch()
        try:
            model.save(directory)
        except Exception:
            # The caller does not yet know this ID if saving fails.
            shutil.rmtree(directory)
            raise
        summary = model.summary()
        report = {key: summary[key] for key in ('config', 'features', 'components', 'retained_variance', 'patch_threshold',
                  'plate_threshold', 'training_images', 'calibration_images', 'training_patches', 'calibration_patches', 'explained_variance_ratio')}
        report.update(model_id=model_id, name='One-class SVM' if model.config.get('model_type') == 'one_class_svm' else 'Patch PCA', seed=body.seed, manifest=manifest, skipped=skipped)
        (directory / 'report.json').write_text(json.dumps(report))
        results.export_anomaly_model(model, report)
        if publish:
            temp = ARTIFACT_ROOT / 'current.tmp'
            temp.write_text(json.dumps({'model_id': model_id}))
            temp.replace(ARTIFACT_ROOT / 'current.json')
        return {'model_id': model_id}
    return train(job_id) if job_id else _submit('train', train)

def train_pca(body: TrainRequest):
    return _train_pca(body)


def _test(directory, path, relative):
    model = _model(str(directory))
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Could not decode image')
    result, patches, z, reconstructed, errors = model.test_image(image, relative)
    test_id = uuid.uuid4().hex
    destination = directory / 'tests' / test_id
    destination.mkdir(parents=True)
    model.export_features(destination / 'features.csv', patches, 'test')
    np.savez_compressed(destination / 'features.npz', raw=patches.values, scores=z, reconstructed=reconstructed, errors=errors)
    map_min = min(0.0, float(model.train_errors.min()), float(model.cal_errors.min()))
    map_span = max((model.patch_threshold - map_min) * 2, 1e-8)
    heat = np.zeros(image.shape[:2], np.uint8)
    coverage = np.zeros_like(heat)
    annotated = image.copy()
    for row in result['patches']:
        x0, y0, x1, y1 = [row[key] for key in ('left', 'top', 'right', 'bottom')]
        heat[y0:y1, x0:x1] = round(float(np.clip((row['error'] - map_min) / map_span, 0, 1)) * 255)
        coverage[y0:y1, x0:x1] = 255
        if row['anomalous']:
            cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (30, 30, 240), 2)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    colored[(patches.mask == 0) | (coverage == 0)] = 0
    result.update(test_id=test_id, model_id=directory.name, image_id=relative,
                  image=_png(annotated), anomaly_map=_png(colored), coverage_mask=_png(cv2.bitwise_and(coverage, patches.mask)),
                  map_min=map_min, map_max=map_min + map_span,
                  membership='training' if relative in {r['image_id'] for r in model.train_records} else
                  'calibration' if relative in {r['image_id'] for r in model.cal_records} else 'unseen')
    (destination / 'result.json').write_text(json.dumps(result))
    return result


def test_pca(body: TestRequest, model_id: str | None = None):
    directory = _directory(model_id)
    path = image_path(body.image_path)
    return _submit('test', lambda job_id: _test(directory, path, body.image_path))


def _evaluate_pca(model_id=None, job_id=None, feature_cache=None):
    directory = _directory(model_id)
    root = folder_path('metal_plate', 'test')
    paths = [path for path in sorted(root.rglob('*')) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    def evaluate(job_id):
        rows, skipped = [], []
        model = _model(str(directory))
        for index, path in enumerate(paths):
            relative = 'metal_plate/test/' + path.relative_to(root).as_posix()
            _progress(job_id, 'Evaluating untouched test images', index, len(paths))
            try:
                safe = image_path(relative)
                image = cv2.imread(str(safe))
                if image is None:
                    raise ValueError('Could not decode image')
                if model.config['pipeline_signature'] != input_signature(model.config.get('feature_set', 'lab_sobel'), model.config.get('preprocessing', 'full')):
                    raise ValueError('Preprocessing changed; retrain this model')
                cache_key = (relative, model.config['patch_size'], model.config.get('feature_set', 'lab_sobel'), model.config['min_coverage'], model.config.get('preprocessing', 'full'))
                patches = feature_cache.get(cache_key) if feature_cache is not None else None
                if patches is None:
                    patches = extract_anomaly_features(image, relative, model.config['patch_size'], model.config['min_coverage'], model.config.get('feature_set', 'lab_sobel'), model.config.get('preprocessing', 'full'))
                    if feature_cache is not None:
                        feature_cache[cache_key] = PatchData(patches.values, patches.records)
                result, _, _, _ = model.score(patches)
                results.export_anomaly_projection(directory.name, relative, result)
                label = path.relative_to(root).parts[0]
                rows.append({'image_id': relative, 'label': label, 'actual': 'GOOD' if label == 'good' else 'BAD',
                             'prediction': result['prediction'], 'score': result['plate_score'],
                             'patches': len(result['patches']), 'anomalous_patches': result['anomalous_patches']})
            except (ValueError, HTTPException) as error:
                skipped.append({'image_id': relative, 'reason': str(error)})
        tp = sum(r['actual'] == 'BAD' and r['prediction'] == 'BAD' for r in rows)
        fp = sum(r['actual'] == 'GOOD' and r['prediction'] == 'BAD' for r in rows)
        tn = sum(r['actual'] == 'GOOD' and r['prediction'] == 'GOOD' for r in rows)
        fn = sum(r['actual'] == 'BAD' and r['prediction'] == 'GOOD' for r in rows)
        groups = {label: {'images': sum(r['label'] == label for r in rows),
                         'flagged': sum(r['label'] == label and r['prediction'] == 'BAD' for r in rows)} for label in sorted({r['label'] for r in rows})}
        result = {'images': len(rows), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                  'recall': tp / (tp + fn) if tp + fn else None, 'false_positive_rate': fp / (fp + tn) if fp + tn else None,
                  'precision': tp / (tp + fp) if tp + fp else None, 'groups': groups, 'rows': rows, 'skipped': skipped}
        temp = directory / 'evaluation.tmp'
        temp.write_text(json.dumps(result))
        temp.replace(directory / 'evaluation.json')
        with (directory / 'evaluation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'label', 'actual', 'prediction', 'score', 'patches', 'anomalous_patches'])
            writer.writeheader()
            writer.writerows(rows)
        results.export_anomaly_evaluation(directory.name, result)
        return result
    return evaluate(job_id) if job_id else _submit('evaluate', evaluate)


def evaluate_pca(model_id: str | None = None):
    return _evaluate_pca(model_id)


def train_pca_sweep(body: SweepRequest):
    combinations = list(product(sorted(set(body.patch_sizes)), sorted(set(body.variance_targets)), sorted(set(body.distance_metrics))))
    def run(job_id):
        completed, failures, cache = [], [], {}
        for index, (patch_size, variance_target, distance_metric) in enumerate(combinations):
            with _lock:
                _jobs[job_id].update(combination=index + 1, combinations=len(combinations))
            parameters = dict(patch_size=patch_size, variance_target=variance_target, distance_metric=distance_metric, seed=body.seed, feature_set=body.feature_set, preprocessing=body.preprocessing)
            try:
                trained = _train_pca(TrainRequest(**parameters), job_id, cache)
                _evaluate_pca(trained['model_id'], job_id, cache)
                completed.append(trained['model_id'])
            except (ValueError, HTTPException) as error:
                failures.append({'parameters': parameters, 'error': str(error)})
        return {'model_ids': completed, 'failures': failures}
    return _submit('sweep', run)


def evaluate_all_pca():
    ids = [r['model_id'] for r in get_models()['models'] if r['compatible']]
    if not ids:
        raise HTTPException(422, 'No compatible models to evaluate')
    def run(job_id):
        cache = {}
        for index, model_id in enumerate(ids):
            with _lock:
                _jobs[job_id].update(combination=index + 1, combinations=len(ids))
            _evaluate_pca(model_id, job_id, cache)
        return {'model_ids': ids, 'failures': []}
    return _submit('evaluate_all', run)


def get_pca_features(model_id: str | None = None, split: Literal['training', 'calibration', 'test'] = 'training',
                     view: Literal['raw', 'standardized', 'reconstructed', 'pca'] = 'raw',
                     offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=100)] = 20,
                     test_id: str | None = None):
    directory = _directory(model_id)
    model = _model(str(directory))
    if split == 'test':
        test_dir = _test_directory(directory, test_id)
        with np.load(test_dir / 'features.npz', allow_pickle=False) as saved:
            raw = saved['raw']
        records = json.loads((test_dir / 'result.json').read_text())['patches']
    else:
        raw = model.train_raw if split == 'training' else model.cal_raw
        records = model.train_records if split == 'training' else model.cal_records
    subset = raw[offset:offset + limit]
    z, reconstructed, errors = model._project(subset)
    values = {'raw': subset, 'standardized': (subset - model.mean) / model.scale, 'reconstructed': reconstructed, 'pca': z}[view]
    columns = model.feature_names if view != 'pca' else [f'PC{i + 1}' for i in range(values.shape[1])]
    return {'total': len(raw), 'columns': columns, 'rows': [{'record': row, 'values': values[i].tolist(), 'error': float(errors[i])}
             for i, row in enumerate(records[offset:offset + limit])]}


def _test_directory(directory, test_id):
    if not test_id or len(test_id) != 32 or any(c not in '0123456789abcdef' for c in test_id):
        raise HTTPException(400, 'Invalid test ID')
    path = directory / 'tests' / test_id
    if not (path / 'result.json').exists():
        raise HTTPException(404, 'Test result not found')
    return path


def download_pca_features(file: Literal['training', 'calibration', 'components', 'test', 'evaluation'] = 'training',
                          model_id: str | None = None, test_id: str | None = None):
    directory = _directory(model_id)
    if file == 'test':
        path = _test_directory(directory, test_id) / 'features.csv'
    else:
        path = directory / {'training': 'training_features.csv', 'calibration': 'calibration_features.csv',
                            'components': 'components.csv', 'evaluation': 'evaluation.csv'}[file]
    if not path.exists():
        raise HTTPException(404, 'Run the relevant model operation first')
    return FileResponse(path, media_type='text/csv', filename=path.name)


def get_model_projection(image_path: str, model_id: str):
    results.validate_id(model_id)
    snapshot = results.anomaly_projection_path(model_id, image_path)
    if snapshot.exists():
        return results.read_json(snapshot)
    directory = _directory(model_id)
    from .os_helpers import image_path as resolve_image_path
    path = resolve_image_path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(422, 'Could not decode image')
    try:
        result, _, _, _, _ = _model(str(directory)).test_image(image, image_path)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    output = {**result, 'model_id': model_id, 'image_id': image_path}
    results.export_anomaly_projection(model_id, image_path, output)
    return output
