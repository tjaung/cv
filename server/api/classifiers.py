"""Supervised mixed-split experiments, fold-local preprocessing and stable 2-D inspection."""
from collections import Counter
from itertools import product
from functools import lru_cache
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from CV.models.classifiers import ImageStore, PCAClassifier, SVMClassifier, KNNClassifier
from .os_helpers import folder_path
from . import models

from . import result_store as results

ROOT = Path(__file__).resolve().parents[2] / 'artifacts/classifiers'


class TrainingRequest(BaseModel):
    seed: int = Field(default=42, ge=0)


def configurations():
    for variance in (.9, .95, .99):
        for metric in ('euclidean', 'manhattan'):
            yield dict(kind='pca', variance_target=variance, metric=metric)
        for kernel, c in product(('linear', 'rbf'), (.1, 1., 10.)):
            yield dict(kind='svm', variance_target=variance, kernel=kernel, C=c)
        for k, weights in product((3, 5, 9), ('uniform', 'distance')):
            yield dict(kind='knn', variance_target=variance, n_neighbors=k, weights=weights)


def _write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def _current():
    pointer = ROOT / 'current.json'
    if not pointer.exists():
        pointer = results.data_root('classifiers') / 'current.json'
    if not pointer.exists():
        return None
    return _run_directory(json.loads(pointer.read_text())['run_id'])



def _run_directory(run_id):
    results.validate_id(run_id)
    artifact = ROOT / run_id
    return artifact if (artifact / 'summary.json').exists() else results.data_root('classifiers') / run_id


def _model_file(directory, model_id):
    artifact = directory / f'{model_id}.joblib'
    saved = results.saved_root('classifiers') / directory.name / artifact.name
    if artifact.exists():
        return artifact
    if saved.exists():
        return saved
    raise HTTPException(410, 'Fitted model is not retained. Saved results remain viewable; retrain this configuration to compute new explanations or curves.')

def list_classifiers():
    directory = _current()
    with models._lock:
        active = next((dict(j) for j in models._jobs.values() if j['kind'] == 'classifiers' and j['status'] in ('queued', 'running')), None)
    summary = results.read_json(directory / 'summary.json') if directory else None
    if summary:
        for report in summary['models']:
            report['model_saved'] = (results.saved_root('classifiers') / directory.name / f"{report['id']}.joblib").exists()
            report['model_available'] = report['model_saved'] or (ROOT / directory.name / f"{report['id']}.joblib").exists()
    return {'summary': {**summary, 'run_id': directory.name} if summary else None,
            'active_job': active, 'configurations': len(list(configurations()))}


def train_classifiers(body: TrainingRequest):
    def run(job_id):
        split = ImageStore(folder_path('metal_plate')).sample(.2, body.seed)
        counts = Counter(s.label for s in split.train)
        unique = {label: len({s.sha256 for s in split.train if s.label == label}) for label in counts}
        folds = min(5, min(unique.values()))
        if folds < 2:
            raise ValueError('Each class needs at least two distinct training images after the 80/20 split for cross-validation')
        root = folder_path('metal_plate').resolve().parent
        def record(s):
            return {'path': Path(s.path).relative_to(root).as_posix(), 'label': s.label, 'original_split': s.original_split}
        summary = {'seed': body.seed, 'test_fraction': .2, 'folds': folds, 'train': [record(s) for s in split.train],
                   'test': [record(s) for s in split.test], 'classes': sorted(counts), 'models': [], 'failures': []}
        extractor = PCAClassifier()
        rows = []
        samples = split.train + split.test
        for i, sample in enumerate(samples):
            models._progress(job_id, 'Extracting classifier image features', i, len(samples))
            rows.append(extractor.extract_features([sample])[0])
        x = np.vstack(rows)
        train, test = x[:len(split.train)], x[len(split.train):]
        y = np.array([s.label for s in split.train])
        truth = np.array([s.label for s in split.test])
        groups = np.array([s.sha256 for s in split.train])
        cv = list(StratifiedGroupKFold(folds, shuffle=True, random_state=body.seed).split(train, y, groups))
        if any(set(y[a]) != set(y) or set(y[b]) != set(y) for a, b in cv):
            raise ValueError('Could not represent every class in each CV fold; add distinct examples per class')
        directory = ROOT / uuid.uuid4().hex
        directory.mkdir(parents=True)
        split.save(directory / 'split.json')
        _write(directory / 'cv_folds.json', [{'train_indices': a.tolist(), 'validation_indices': b.tolist()} for a, b in cv])
        extractor.export_features(directory / 'training_features.csv', split.train, train)
        extractor.export_features(directory / 'test_features.csv', split.test, test)
        np.savez_compressed(directory / 'features.npz', train=train, test=test)
        _write(directory / 'summary.json', summary)
        _write(ROOT / 'current.json', {'run_id': directory.name})
        results.export_classifier_summary(directory, summary)
        configs = list(configurations())
        with threadpool_limits(limits=1):
            for i, config in enumerate(configs):
                with models._lock:
                    models._jobs[job_id].update(combination=i, combinations=len(configs))
                models._progress(job_id, 'Cross-validating and fitting classifiers', i, len(configs))
                try:
                    params = {k: v for k, v in config.items() if k != 'kind'}
                    model = {'pca': PCAClassifier, 'svm': SVMClassifier, 'knn': KNNClassifier}[config['kind']](**params)
                    if config.get('n_neighbors', 1) > min(len(a) for a, _ in cv):
                        raise ValueError('Not enough training images per fold for this neighbor count')
                    scores = cross_validate(model.pipeline, train, y, cv=cv,
                                            scoring={'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy', 'macro_f1': 'f1_macro'},
                                            n_jobs=1, error_score='raise')
                    model.fit(train, y)
                    training_evaluation = model.test(train, y)
                    training_evaluation['balanced_accuracy'] = float(balanced_accuracy_score(y, training_evaluation['predictions']))
                    evaluation = model.test(test, truth)
                    evaluation['balanced_accuracy'] = float(balanced_accuracy_score(truth, evaluation['predictions']))
                    model_id = uuid.uuid4().hex
                    model.save(directory / f'{model_id}.joblib')
                    results.export_classifier_plot(directory, model_id, model, summary, train, test)
                    summary['models'].append({'id': model_id, 'config': config, 'evaluation': evaluation, 'training_evaluation': training_evaluation,
                                              'cv': {key[5:]: {'mean': float(value.mean()), 'std': float(value.std()), 'folds': value.tolist()}
                                                     for key, value in scores.items() if key.startswith('test_')}})
                except Exception as error:
                    summary['failures'].append({'config': config, 'error': str(error)})
                    if isinstance(error, OSError) and error.errno == 28:
                        raise
                _write(directory / 'summary.json', summary)
                results.export_classifier_summary(directory, summary)
        return {'models': len(summary['models']), 'failures': summary['failures']}
    return models._submit('classifiers', run)


def inspect_classifier(model_id: str, image_index: int = 0):
    directory = _current()
    if directory is None:
        raise HTTPException(404, 'Train classifiers first')
    summary = json.loads((directory / 'summary.json').read_text())
    report = next((r for r in summary['models'] if r['id'] == model_id), None)
    if report is None or not 0 <= image_index < len(summary['test']):
        raise HTTPException(404, 'Model or test image not found')
    snapshot = results.data_root('classifiers') / directory.name / 'plots' / f'{model_id}.json.gz'
    if snapshot.exists():
        plot = results.read_json(snapshot)
        index = len(summary['train']) + image_index
        return {**{k:v for k,v in plot.items() if k not in ('samples','neighbors')}, 'selected': plot['samples'][index], 'neighbors': plot['neighbors'][index]}
    model = joblib.load(_model_file(directory, model_id))
    with np.load(directory / 'features.npz') as data:
        train, test = data['train'], data['test']
    selected = test[image_index:image_index + 1]
    transform = model.pipeline[:-1]
    z = transform.transform(train)
    point = transform.transform(selected)
    # Fixed per-model display bounds cover every saved holdout image. These
    # projections only size the display; no held-out data is used to fit PCA.
    xy = np.pad(z[:, :2], ((0, 0), (0, max(0, 2 - z.shape[1]))))
    selected_xy = np.pad(point[0, :2], (0, max(0, 2 - z.shape[1])))
    test_z = transform.transform(test)
    test_xy = np.pad(test_z[:, :2], ((0, 0), (0, max(0, 2 - z.shape[1]))))
    extent = np.vstack((xy, test_xy))
    low, high = extent.min(axis=0), extent.max(axis=0)
    margin = np.maximum((high - low) * .08, .5)
    low, high = low - margin, high + margin
    classifier = model.pipeline['classifier']
    # A fixed slice: higher PCs stay at their training means, never at the
    # selected image. Sample cell centers so the colored tiles align with axes.
    size = 45
    coordinates = [low[axis] + (np.arange(size) + .5) / size * (high[axis] - low[axis]) for axis in (0, 1)]
    xx, yy = np.meshgrid(*coordinates)
    grid = np.tile(z.mean(axis=0), (size * size, 1))
    grid[:, 0] = xx.ravel()
    if z.shape[1] > 1:
        grid[:, 1] = yy.ravel()
    with threadpool_limits(limits=1):
        regions = classifier.predict(grid).reshape(size, size).tolist()
    neighbors = []
    if report['config']['kind'] == 'knn':
        distances, indices = classifier.kneighbors(point)
        neighbors = [{'index': int(i), 'distance': float(d), **summary['train'][i]} for i, d in zip(indices[0], distances[0])]
    return {'training': [{**r, 'point': xy[i].tolist()} for i, r in enumerate(summary['train'])],
            'selected': {**summary['test'][image_index], 'point': selected_xy.tolist(), 'prediction': model.predict(selected)[0]},
            'neighbors': neighbors, 'support_indices': classifier.support_.tolist() if hasattr(classifier, 'support_') else [],
            'regions': regions, 'bounds': [low.tolist(), high.tolist()],
            'variance': model.pipeline['pca'].explained_variance_ratio_[:2].tolist(), 'dimensions': z.shape[1]}


def classify_review_image(run_id: str, image_index: int):
    """Score one image in a frozen run's train+holdout inventory with every model.

    Use the saved vectors so this review cannot change model features or splits.
    No training, writes, or new feature extraction occur here.
    """
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise HTTPException(400, 'Invalid classifier run ID')
    directory = _run_directory(run_id)
    if not (directory / 'summary.json').is_file():
        raise HTTPException(404, 'Classifier run not found')
    summary = json.loads((directory / 'summary.json').read_text())
    inventory = summary['train'] + summary['test']
    if not 0 <= image_index < len(inventory):
        raise HTTPException(404, 'Image not found in classifier run')
    if not summary['models']:
        raise HTTPException(409, 'No fitted classifiers in this run')
    training = image_index < len(summary['train'])
    membership = 'training' if training else 'holdout'
    index = image_index if training else image_index - len(summary['train'])
    sample = inventory[image_index]
    predictions = []
    for report in summary['models']:
        evaluation = report.get('training_evaluation' if training else 'evaluation')
        if evaluation:
            prediction = evaluation['predictions'][index]
        else:
            with np.load(directory / 'features.npz') as data:
                features = data['train' if training else 'test'][index:index+1]
            model = joblib.load(_model_file(directory, report['id']))
            prediction = str(model.predict(features)[0])
        predictions.append({'model_id':report['id'], 'prediction':prediction, 'correct':prediction == sample['label']})
    return {'run_id': run_id, 'image_index': image_index, **sample, 'membership': membership, 'predictions': predictions}


@lru_cache(maxsize=128)
def _saved_training_evaluation(directory, model_id, model_mtime, features_mtime):
    """Compute training scores for older saved runs without fitting anything."""
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    truth = [s['label'] for s in summary['train']]
    with np.load(directory / 'features.npz') as data:
        train = data['train']
    model = joblib.load(_model_file(directory, model_id))
    with threadpool_limits(limits=1):
        evaluation = model.test(train, truth)
    evaluation['balanced_accuracy'] = float(balanced_accuracy_score(truth, evaluation['predictions']))
    return evaluation


def get_classifier_training_metrics(run_id: str):
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise HTTPException(400, 'Invalid classifier run ID')
    directory = _run_directory(run_id)
    if not (directory / 'summary.json').is_file():
        raise HTTPException(404, 'Classifier run not found')
    summary = json.loads((directory / 'summary.json').read_text())
    metrics = {}
    for report in summary['models']:
        metrics[report['id']] = report.get('training_evaluation') or _saved_training_evaluation(
            str(directory), report['id'], _model_file(directory, report['id']).stat().st_mtime_ns,
            (directory / 'features.npz').stat().st_mtime_ns)
    return metrics


class CurveRequest(BaseModel):
    run_id: str
    model_id: str


def _curve_directory(run_id, model_id):
    if any(len(value) != 32 or any(c not in '0123456789abcdef' for c in value) for value in (run_id, model_id)):
        raise HTTPException(400, 'Invalid run or model ID')
    directory = _run_directory(run_id)
    if not (directory / 'summary.json').is_file():
        raise HTTPException(404, 'Classifier run not found')
    summary = json.loads((directory / 'summary.json').read_text())
    if not any(r['id'] == model_id for r in summary['models']):
        raise HTTPException(404, 'Classifier not found')
    return directory


def get_classifier_curves(run_id: str, model_id: str):
    directory = _curve_directory(run_id, model_id)
    path = results.data_root('classifiers') / directory.name / f'{model_id}-learning-curves.json'
    if not path.exists():
        path = directory / f'{model_id}-learning-curves.json'
    with models._lock:
        job = next((dict(j) for j in reversed(list(models._jobs.values())) if j.get('curve_run') == run_id and j.get('curve_model') == model_id), None)
    return {'curve': json.loads(path.read_text()) if path.exists() else None, 'job': job}


def train_classifier_curves(body: CurveRequest):
    directory = _curve_directory(body.run_id, body.model_id)
    def run(job_id):
        from CV.models.classifiers.learning_curves import learning_curves
        summary = json.loads((directory / 'summary.json').read_text())
        split = json.loads((directory / 'split.json').read_text())
        labels = np.array([s['label'] for s in summary['train']])
        groups = np.array([s['sha256'] for s in split['train']])
        with np.load(directory / 'features.npz') as data:
            features = data['train']
        manifest = directory / 'cv_folds.json'
        if manifest.exists():
            folds = [(np.array(f['train_indices']), np.array(f['validation_indices'])) for f in json.loads(manifest.read_text())]
        else:
            folds = list(StratifiedGroupKFold(summary['folds'], shuffle=True, random_state=summary['seed']).split(features, labels, groups))
        model = joblib.load(_model_file(directory, body.model_id))
        with threadpool_limits(limits=1):
            curve = learning_curves(model.pipeline, features, labels, groups, folds, summary['seed'],
                                    progress=lambda done, total: models._progress(job_id, 'Computing learning curves', done, total))
        curve.update(run_id=body.run_id, model_id=body.model_id)
        results.write_json(results.data_root('classifiers') / directory.name / f'{body.model_id}-learning-curves.json', curve)
        results.write_csv(results.data_root('classifiers') / directory.name / f'{body.model_id}-learning-curves.csv', curve['points'])
        return {'model_id': body.model_id}
    result = models._submit('classifier_curves', run)
    with models._lock:
        models._jobs[result['job_id']].update(curve_run=body.run_id, curve_model=body.model_id)
    return result


def get_classifier_patch_explanation(run_id: str, model_id: str, image_index: int):
    from CV.postprocessing import explain_patches
    from CV.models.feature_sets import feature_signature, extract_model_features
    from CV.models.classifiers.base import FEATURE_SET
    from .os_helpers import image_path
    import cv2
    directory = _curve_directory(run_id, model_id)
    summary = json.loads((directory / 'summary.json').read_text())
    inventory = summary['train'] + summary['test']
    if not 0 <= image_index < len(inventory):
        raise HTTPException(404, 'Image not found')
    sample = inventory[image_index]
    snapshot = results.data_root('classifiers') / directory.name / 'explanations' / f'{model_id}-{image_index}.json.gz'
    if snapshot.exists():
        return results.read_json(snapshot)
    model = joblib.load(_model_file(directory, model_id))
    if model.signature != feature_signature(FEATURE_SET):
        raise HTTPException(409, 'Feature pipeline changed; retrain before generating patch explanations')
    image = cv2.imread(str(image_path(sample['path'])))
    if image is None:
        raise HTTPException(422, 'Could not decode image')
    try:
        patches = extract_model_features(image, sample['path'], model.patch_size, feature_set=FEATURE_SET)
        pooled = np.concatenate((patches.values.mean(axis=0), patches.values.std(axis=0)))
        training = image_index < len(summary['train'])
        index = image_index if training else image_index-len(summary['train'])
        with np.load(directory / 'features.npz') as data:
            saved = data['train' if training else 'test'][index]
        if not np.allclose(pooled, saved, rtol=1e-5, atol=1e-6):
            raise HTTPException(409, 'Image features differ from the saved run; retrain before explaining this prediction')
        with threadpool_limits(limits=1):
            result = explain_patches(model, patches.values, patches.records)
        output = {**result, 'width': image.shape[1], 'height': image.shape[0]}
        results.write_json(snapshot, output)
        return output
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
