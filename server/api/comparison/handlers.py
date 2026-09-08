"""Saved-model robustness comparisons on one shared, generated image sample."""
from datetime import datetime, timezone
from pathlib import Path
import time
import threading
import json
import uuid
from typing import Literal

import cv2
import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits

from CV.models.data_shuffle import collect_images
from ..anomaly_detection import models
from ..shared import result_store as results
from ..shared.os_helpers import folder_path


# Process-local predictors, reused until explicitly replaced or the server restarts.
_predictors = {}
_cache_lock = threading.RLock()


class LoadModelsRequest(BaseModel):
    model_ids: list[str] = Field(min_length=1)


def _model_version(entry):
    family, *ids = entry['id'].split(':')
    root = results.saved_root(family) / ids[0]
    files = [root / 'model.npz', root / 'model.json'] if family == 'anomaly' else [root / (f'{ids[1]}.joblib' if family == 'classifiers' else 'model.pt')]
    return (json.dumps(entry['config'], sort_keys=True), tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files))


def _is_loaded(entry):
    try:
        with _cache_lock:
            cached = _predictors.get(entry['id'])
            return bool(entry['available'] and cached and cached['version'] == _model_version(entry))
    except OSError:
        return False


def _selected_entries(ids):
    available = {m['id']: m for m in _saved_models()}
    selected = list(dict.fromkeys(ids))
    if any(i not in available or not available[i]['available'] for i in selected):
        raise HTTPException(422, 'Choose available saved models; refresh the list if a model was removed or changed')
    return [available[i] for i in selected]


def load_comparison_models(body: LoadModelsRequest):
    entries = _selected_entries(body.model_ids)
    def load(job_id):
        errors = []
        with _cache_lock:
            # Release unchecked models when loading a new selection to limit memory use.
            for key in list(_predictors):
                if key not in body.model_ids:
                    del _predictors[key]
        for index, entry in enumerate(entries):
            models._progress(job_id, f"Loading {entry['name']}", index, len(entries))
            if _is_loaded(entry):
                continue
            with _cache_lock:
                _predictors.pop(entry['id'], None)
            try:
                version = _model_version(entry)
                predictor = Predictor(entry)
                with _cache_lock:
                    _predictors[entry['id']] = dict(version=version, predictor=predictor, warmed=False)
            except Exception as error:
                errors.append(f"{entry['name']}: {error}")
        if errors:
            raise ValueError('; '.join(errors))
        return {'loaded_model_ids': [entry['id'] for entry in entries]}
    return models._submit('comparison_load', load)


class ComparisonRequest(BaseModel):
    model_ids: list[str] = Field(min_length=1)
    mode: Literal['set', 'single'] = 'set'
    count: int = Field(default=12, ge=1, le=50)


def _saved_metrics(evaluation):
    """Compact saved test metrics; collapse defect classes only for binary measures."""
    if not evaluation:
        return None
    matrix = evaluation.get('confusion_matrix')
    names = evaluation.get('classes', evaluation.get('class_names', []))
    if matrix is not None and 'good' in names:
        matrix = np.asarray(matrix)
        g = names.index('good')
        tn = int(matrix[g, g])
        fp = int(matrix[g].sum()) - tn
        fn = int(matrix[:, g].sum()) - tn
        tp = int(matrix.sum()) - tn - fp - fn
        accuracy = evaluation.get('accuracy')
    elif all(k in evaluation for k in ('tp', 'fp', 'tn', 'fn')):
        tp, fp, tn, fn = (evaluation[k] for k in ('tp', 'fp', 'tn', 'fn'))
        accuracy = None
    else:
        return None
    total = tp + fp + tn + fn
    return dict(images=total, binary_accuracy=(tp + tn)/total if total else None,
                class_accuracy=accuracy, defect_recall=tp/(tp + fn) if tp + fn else None,
                false_positive_rate=fp/(fp + tn) if fp + tn else None, missed_defects=fn,
                macro_f1=evaluation.get('report', {}).get('macro avg', {}).get('f1-score'))


def _read_optional_json(path):
    try:
        return results.read_json(path)
    except (OSError, ValueError):
        return {}


def _saved_models():
    """Only enumerate opt-in model copies; never fall back to temporary artifacts."""
    rows = []
    for directory in sorted(results.saved_root('anomaly').glob('*')):
        if not directory.is_dir() or not (directory / 'model.npz').exists():
            continue
        try:
            report = results.read_json(directory / 'report.json')
            from CV.models.anomaly_detection.inputs import input_signature
            config = report['config']
            compatible = config['pipeline_signature'] == input_signature(config.get('feature_set', 'lab_sobel'), config.get('preprocessing', 'full'))
            kind = 'One-class SVM' if config.get('model_type') == 'one_class_svm' else 'PCA anomaly'
            rows.append(dict(id=f'anomaly:{directory.name}', family='anomaly', name=f'{kind} · {directory.name[:8]}',
                             config=config, metrics=_saved_metrics(_read_optional_json(results.data_root('anomaly') / directory.name / 'evaluation.json')), available=compatible, reason=None if compatible else 'Preprocessing changed; retrain and save this model again.'))
        except (OSError, ValueError, KeyError):
            rows.append(dict(id=f'anomaly:{directory.name}', family='anomaly', name=directory.name[:8], config={}, available=False, reason='Saved model metadata is missing or unreadable.'))
    for path in sorted(results.saved_root('classifiers').glob('*/*.joblib')):
        summary_path = results.data_root('classifiers') / path.parent.name / 'summary.json'
        summary = _read_optional_json(summary_path)
        report = next((m for m in summary.get('models', []) if m['id'] == path.stem), {})
        config = report.get('config', {})
        rows.append(dict(id=f'classifiers:{path.parent.name}:{path.stem}', family='classifiers',
                         name=f"{config.get('kind', 'Classifier').upper()} · {path.stem[:8]}", config=config, metrics=_saved_metrics(report.get('evaluation', {})), available=True, reason=None))
    for path in sorted(results.saved_root('cnn').glob('*/model.pt')):
        summary_path = results.data_root('cnn') / path.parent.name / 'summary.json'
        try:
            from ..cnn.handlers import _input_signature
            summary = results.read_json(summary_path)
            compatible = summary.get('input_signature') == _input_signature()
            config = summary['config']
            names = {'standard': 'Whole-plate ResNet', 'defect_weighted': 'Weighted ResNet', 'patch': 'Binary patch ResNet', 'patch_multiclass': 'Four-class patch ResNet'}
            rows.append(dict(id=f'cnn:{path.parent.name}', family='cnn', name=f"{names.get(config.get('variant', 'standard'), 'ResNet')} · {path.parent.name[:8]}",
                             config=config, metrics=_saved_metrics(summary.get('test', {})), available=compatible, reason=None if compatible else 'CNN input pipeline changed; retrain and save again.'))
        except (OSError, ValueError, KeyError):
            rows.append(dict(id=f'cnn:{path.parent.name}', family='cnn', name=f'ResNet · {path.parent.name[:8]}', config={}, available=False, reason='Saved run metadata is missing or unreadable.'))
    return rows


def get_comparison_models():
    with models._lock:
        active = next((dict(j) for j in models._jobs.values() if j['status'] in ('queued', 'running')), None)
    runs = []
    for path in results.data_root('comparison').glob('*/result.json'):
        data = results.read_json(path)
        runs.append({k: data[k] for k in ('run_id', 'created_at', 'status', 'mode')})
    return {'models': [dict(entry, loaded=_is_loaded(entry)) for entry in _saved_models()], 'active_job': active, 'runs': sorted(runs, key=lambda r: r['created_at'], reverse=True)}


def rotate_image(image, angle):
    """Rotate on an expanded canvas, filling new pixels with median border color."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    width, height = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    matrix[:, 2] += ((width-w)/2, (height-h)/2)
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    fill = tuple(float(v) for v in np.median(border, axis=0))
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def choose_samples(inventory, mode, count, rng):
    if mode == 'single':
        # Independent 50/50 class-group choice, then image choice, then transform choice.
        defect = bool(rng.random() >= .5)
        pool = [s for s in inventory if (s.label != 'good') == defect]
        if not pool:
            raise ValueError('Need both good and defective images for balanced random single-image sampling')
        selected = [pool[int(rng.integers(len(pool)))]]
    else:
        selected = [inventory[i] for i in rng.choice(len(inventory), size=min(count, len(inventory)), replace=False)]
    return [(sample, float(rng.choice([-90, -45, -30, -15, 15, 30, 45, 90, 180])) if rng.random() >= .5 else 0.) for sample in selected]


class Predictor:
    """Load once; accept only an image at inference, never its ground-truth label."""
    def __init__(self, entry):
        self.entry = entry
        self.exposure = {}
        family, *ids = entry['id'].split(':')
        if family == 'anomaly':
            from CV.models import PCAAnomalyDetector, OneClassSVMDetector
            root = results.saved_root('anomaly') / ids[0]
            cls = OneClassSVMDetector if entry['config'].get('model_type') == 'one_class_svm' else PCAAnomalyDetector
            self.model = cls.load(root)
            report = results.read_json(root / 'report.json')
            self.exposure = {r['image_id']: r['split'] for r in report.get('manifest', [])}
            self.classification = False
        elif family == 'classifiers':
            import joblib
            from CV.models.classifiers.base import FEATURE_SET
            from CV.features.feature_sets import feature_signature
            self.model = joblib.load(results.saved_root('classifiers') / ids[0] / f'{ids[1]}.joblib')
            if self.model.signature != feature_signature(FEATURE_SET):
                raise ValueError('Classifier features changed; retrain and save this model again')
            path = results.data_root('classifiers') / ids[0] / 'summary.json'
            if path.exists():
                summary = results.read_json(path)
                self.exposure = {r['path']: 'training' for r in summary['train']}
                self.exposure.update({r['path']: 'holdout' for r in summary['test']})
            self.classification = True
        else:
            import torch
            from CV.models.object_detection.resnet import PlateResNet
            from CV.models.object_detection.patch_resnet import PatchResNet
            from CV.models.object_detection.multiclass_patch_resnet import MulticlassPatchResNet
            checkpoint = torch.load(results.saved_root('cnn') / ids[0] / 'model.pt', map_location='cpu', weights_only=True)
            variant = entry['config'].get('variant', 'standard')
            if variant == 'patch_multiclass':
                self.model = MulticlassPatchResNet(checkpoint['class_names'], weights=None)
            elif variant == 'patch':
                self.model = PatchResNet(weights=None)
            else:
                self.model = PlateResNet(checkpoint['class_names'], weights=None)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            split = results.data_root('cnn') / ids[0] / 'split.json'
            if split.exists():
                inventory = results.read_json(split)
                self.exposure = {r['path']: 'training' if k == 'train' else 'holdout' for k, rows in inventory.items() for r in rows}
            self.classification = variant != 'patch'

    def predict(self, image):
        family = self.entry['family']
        if family == 'anomaly':
            from CV.models.anomaly_detection.inputs import extract_anomaly_features
            c = self.model.config
            patches = extract_anomaly_features(image, patch_size=c['patch_size'], min_coverage=c['min_coverage'],
                                              feature_set=c.get('feature_set', 'lab_sobel'), preprocessing=c.get('preprocessing', 'full'))
            # Production decision only: omit expensive explanation/nearest-reference plots.
            scores = self.model._project(patches.values)[2]
            score = float(scores.max())
            return {'prediction': 'defect' if score > self.model.plate_threshold else 'good', 'score': score,
                    'score_kind': 'Anomaly score', 'threshold': self.model.plate_threshold}
        if family == 'classifiers':
            from CV.features.feature_sets import extract_model_features
            from CV.models.classifiers.base import FEATURE_SET
            patches = extract_model_features(image, patch_size=self.model.patch_size, feature_set=FEATURE_SET)
            pooled = np.concatenate((patches.values.mean(axis=0), patches.values.std(axis=0)))[None]
            return {'prediction': str(self.model.predict(pooled)[0])}
        import torch
        from PIL import Image
        from CV.preprocessing import segment_plate
        from CV.postprocessing.patch_geometry import patch_boxes
        from CV.models.object_detection.multiclass_patch_resnet import aggregate_patches
        segmented = segment_plate(image)
        if not segmented.plate_mask.any():
            raise ValueError('No plate found after segmentation')
        plate = segmented.plate
        variant = self.entry['config'].get('variant', 'standard')
        names = self.model.class_names
        with torch.inference_mode():
            if variant not in ('patch', 'patch_multiclass'):
                tensor = self.model.transform(Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)))
                p = self.model(tensor[None]).softmax(1)[0].tolist()
            else:
                size = self.entry['config'].get('patch_size', 64)
                boxes = patch_boxes(segmented.plate_mask, size, size//2)
                if not boxes:
                    raise ValueError('No eligible patches')
                probabilities = []
                for start in range(0, len(boxes), 16):
                    batch = [self.model.transform(Image.fromarray(cv2.cvtColor(plate[b['top']:b['bottom'], b['left']:b['right']], cv2.COLOR_BGR2RGB))) for b in boxes[start:start+16]]
                    probabilities.extend(self.model(torch.stack(batch)).softmax(1).tolist())
                if variant == 'patch':
                    score = max(row[1] for row in probabilities)
                    return {'prediction': 'defect' if score >= .5 else 'good', 'score': score, 'score_kind': 'Maximum patch defect score', 'threshold': .5}
                records = [dict(probabilities=row, anomalous=names[int(np.argmax(row))] != 'good', score=max(row)) for row in probabilities]
                p = aggregate_patches(records, names)
        index = int(np.argmax(p))
        return {'prediction': names[index], 'score': p[index], 'score_kind': 'Class score'}


def summarize(rows):
    successful = [r for r in rows if not r.get('error')]
    good = [r for r in successful if r['actual'] == 'good']
    bad = [r for r in successful if r['actual'] != 'good']
    tp = sum(r['prediction'] != 'good' for r in bad)
    fp = sum(r['prediction'] != 'good' for r in good)
    tn, fn = len(good)-fp, len(bad)-tp
    times = [r['elapsed_ms'] for r in successful]
    typed = [r for r in successful if r['supports_classes']]
    return dict(total=len(rows), evaluated=len(successful), errors=len(rows)-len(successful), tp=tp, fp=fp, tn=tn, fn=fn,
                binary_accuracy=(tp+tn)/len(successful) if successful else None,
                defect_recall=tp/len(bad) if bad else None, false_positive_rate=fp/len(good) if good else None,
                defect_precision=tp/(tp+fp) if tp+fp else None,
                class_accuracy=sum(r['prediction']==r['actual'] for r in typed)/len(typed) if typed else None,
                mean_ms=float(np.mean(times)) if times else None, median_ms=float(np.median(times)) if times else None,
                p95_ms=float(np.percentile(times,95)) if times else None)


def run_comparison(body: ComparisonRequest):
    entries = _selected_entries(body.model_ids)
    if any(not _is_loaded(entry) for entry in entries):
        raise HTTPException(409, 'Load the selected models before running predictions; the server may have restarted or a saved model changed')
    run_id = uuid.uuid4().hex
    def run(job_id):
        directory = results.data_root('comparison') / run_id
        output = dict(run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(), mode=body.mode,
                      status='running', source_split='test', models=entries, samples=[], predictions=[], summaries=[])
        def persist():
            output['summaries'] = []
            for entry in entries:
                rows = [r for r in output['predictions'] if r['model_id'] == entry['id']]
                if rows:
                    output['summaries'].append(dict(model_id=entry['id'], **summarize(rows),
                        original=summarize([r for r in rows if not r['transformed']]),
                        transformed=summarize([r for r in rows if r['transformed']])))
            results.write_json(directory / 'result.json', output)
        try:
            inventory = [sample for sample in collect_images(folder_path('metal_plate')) if sample.original_split == 'test']
            if not inventory:
                raise ValueError('No labeled images found in the test folder')
            rng = np.random.default_rng()
            chosen = choose_samples(inventory, body.mode, body.count, rng)
            models._progress(job_id, 'Generating shared random image set', 0, len(chosen))
            for index, (sample, angle) in enumerate(chosen):
                image = cv2.imread(sample.path)
                if image is None:
                    raise ValueError(f'Could not read {sample.path}')
                if angle:
                    image = rotate_image(image, angle)
                destination = directory / 'images' / f'{index}.png'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), image):
                    raise OSError('Could not save comparison image')
                relative = Path(sample.path).relative_to(folder_path('metal_plate').parent).as_posix()
                output['samples'].append(dict(id=index, path=relative, actual=sample.label, original_split=sample.original_split,
                                               transformed=bool(angle), rotation_degrees=angle))
            persist()
            # CPU-only, one compute thread per model for a consistent timing basis.
            import torch
            old_threads = torch.get_num_threads()
            old_cv_threads = cv2.getNumThreads()
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            try:
                with threadpool_limits(limits=1):
                    for model_index, entry in enumerate(entries):
                        with _cache_lock:
                            cached = _predictors[entry['id']]
                        predictor = cached['predictor']
                        if not cached['warmed']:
                            models._progress(job_id, f"Warming up {entry['name']}", model_index*len(chosen), len(entries)*len(chosen))
                            warmup = cv2.imread(str(directory / 'images/0.png'))
                            try:
                                predictor.predict(warmup)
                                cached['warmed'] = True
                            except Exception:
                                pass
                        entry['warmup_succeeded'] = cached['warmed']
                        for sample in output['samples']:
                            image = cv2.imread(str(directory / 'images' / f"{sample['id']}.png"))
                            row = dict(model_id=entry['id'], sample_id=sample['id'], actual=sample['actual'], transformed=sample['transformed'],
                                       supports_classes=predictor.classification, exposure=predictor.exposure.get(sample['path'], 'unknown'))
                            models._progress(job_id, f"Predicting with {entry['name']}", model_index*len(chosen)+sample['id'], len(entries)*len(chosen))
                            start = time.perf_counter()
                            try:
                                row.update(predictor.predict(image))
                                row['elapsed_ms'] = (time.perf_counter()-start)*1000
                                row['correct'] = (row['prediction'] == row['actual']) if predictor.classification else ((row['prediction'] == 'good') == (row['actual'] == 'good'))
                            except Exception as error:
                                row.update(error=str(error), elapsed_ms=(time.perf_counter()-start)*1000)
                            output['predictions'].append(row)
                            persist()
                        del predictor
            finally:
                torch.set_num_threads(old_threads)
                cv2.setNumThreads(old_cv_threads)
            output['status'] = 'complete'
            persist()
            results.write_csv(directory / 'predictions.csv', output['predictions'])
            results.write_csv(directory / 'summary.csv', output['summaries'])
            return {'comparison_id': run_id}
        except Exception as error:
            output.update(status='failed', error=str(error))
            persist()
            raise
    return {**models._submit('comparison', run), 'run_id': run_id}


def get_comparison_run(run_id: str):
    results.validate_id(run_id)
    path = results.data_root('comparison') / run_id / 'result.json'
    if not path.exists():
        raise HTTPException(404, 'Comparison is still preparing or does not exist')
    return results.read_json(path)


def get_comparison_image(run_id: str, image_index: int):
    results.validate_id(run_id)
    if image_index < 0:
        raise HTTPException(404, 'Image not found')
    path = results.data_root('comparison') / run_id / 'images' / f'{image_index}.png'
    if not path.exists():
        raise HTTPException(404, 'Image not found')
    return FileResponse(path, media_type='image/png')
