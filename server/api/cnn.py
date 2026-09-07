"""ResNet jobs and durable, inspectable CNN results (torch loaded on demand)."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid
from typing import Literal

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from . import models, result_store as results
from .features import _png
from .os_helpers import folder_path, image_path

ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / 'artifacts/cnn'


class TrainRequest(BaseModel):
    variant: Literal['standard', 'defect_weighted', 'patch', 'patch_multiclass'] = 'standard'
    num_workers: int = Field(default=2, ge=0, le=8)
    patch_size: int = Field(default=64, ge=16, le=256)
    epochs: int = Field(default=5, ge=1, le=200)
    batch_size: int = Field(default=16, ge=2, le=128)
    learning_rate: float = Field(default=.0001, gt=0, le=.1)
    seed: int = Field(default=42, ge=0, le=2**32-1)


class RunRequest(BaseModel):
    run_id: str


def _directory(run_id):
    results.validate_id(run_id)
    directory = results.data_root('cnn') / run_id
    if not (directory / 'summary.json').exists():
        raise HTTPException(404, 'CNN run not found')
    return directory


def _checkpoint(run_id):
    for path in (ARTIFACT_ROOT / run_id / 'model.pt', results.saved_root('cnn') / run_id / 'model.pt'):
        if path.exists():
            return path
    raise HTTPException(410, 'Fitted CNN was deleted. Saved predictions remain available; retrain to test again.')


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def _input_signature():
    """Avoid recomputing old predictions with a changed segmentation pipeline."""
    import hashlib
    root = Path(__file__).resolve().parents[2] / 'CV'
    digest = hashlib.sha256()
    for path in sorted((root / 'preprocessing').glob('*.py')) + [root / 'models/object_detection/resnet.py']:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def get_cnn():
    runs = []
    for path in sorted(results.data_root('cnn').glob('*/summary.json'), reverse=True):
        summary = results.read_json(path)
        run_id = path.parent.name
        summary['model_saved'] = (results.saved_root('cnn') / run_id / 'model.pt').exists()
        summary['model_available'] = summary['model_saved'] or (ARTIFACT_ROOT / run_id / 'model.pt').exists()
        runs.append(summary)
    runs.sort(key=lambda r: r['created_at'], reverse=True)
    with models._lock:
        active = next((dict(job) for job in models._jobs.values() if job['status'] in ('queued', 'running')), None)
    return {'runs': runs, 'active_job': active}


def _weight_view(model):
    kernels = model.network.conv1.weight.detach().cpu().numpy().transpose(0, 2, 3, 1)
    limit = float(np.max(np.abs(kernels))) or 1.
    tiles = np.clip((kernels/limit + 1)*127.5, 0, 255).astype(np.uint8)
    # One shared symmetric scale: gray is zero, bright/dark are signed weights.
    mosaic = np.zeros((8*32, 8*32, 3), np.uint8)
    for i, tile in enumerate(tiles):
        mosaic[i//8*32:(i//8+1)*32, i%8*32:(i%8+1)*32] = cv2.resize(tile[:, :, ::-1], (32, 32), interpolation=cv2.INTER_NEAREST)
    weights = model.network.fc.weight.detach().cpu().numpy()
    return {'conv1_filters': _png(mosaic), 'conv1_scale': limit,
            'classifier_weights': weights.tolist(), 'classifier_bias': model.network.fc.bias.detach().cpu().tolist(),
            'layers': [{'name': name, 'parameters': value.numel(),
                        'mean': float(value.detach().float().mean().cpu()),
                        'std': float(value.detach().float().std(unbiased=False).cpu())}
                       for name, value in model.named_parameters() if name.endswith('weight')]}


def _export_evaluation(directory, split, evaluation):
    root = folder_path('metal_plate').resolve().parent
    for row in evaluation['predictions']:
        if Path(row['path']).is_absolute():
            row['path'] = Path(row['path']).resolve().relative_to(root).as_posix()
    results.write_json(directory / f'{split}.json', evaluation)
    results.write_csv(directory / f'{split}_predictions.csv', evaluation['predictions'])
    results.write_csv(directory / f'{split}_metrics.csv', [
        {'class': label, **score} for label, score in evaluation['report'].items() if isinstance(score, dict)])
    results.write_csv(directory / f'{split}_overall.csv', [{
        'accuracy': evaluation['accuracy'], 'cross_entropy': evaluation['loss'],
        **{f'macro_{key}': value for key, value in evaluation['report']['macro avg'].items()}}])
    return {k: v for k, v in evaluation.items() if k != 'predictions'}


def train_cnn(body: TrainRequest):
    def run(job_id):
        from CV.models.object_detection.cached_data import RunData
        with RunData(body.num_workers) as data:
            return run_cached(job_id, data)

    def run_cached(job_id, data):
        import torch
        from CV.models.object_detection.resnet import PlateResNet, get_data_loaders, train, evaluate
        from CV.models.object_detection.cached_data import CachedPlateDataset, CachedPatchDataset, CachedMulticlassPatchDataset
        cache_progress = lambda done, total: models._progress(job_id, 'Caching segmented plates', done, total)
        torch.manual_seed(body.seed)
        models._progress(job_id, 'Preparing shared 80/20 split')
        models._progress(job_id, 'Loading pretrained ResNet-18 (first use downloads weights)')
        if body.variant in ('patch', 'patch_multiclass'):
            from CV.models.object_detection.patch_resnet import PatchResNet, CLASSES
            from CV.models.object_detection.cnn import get_train_test_data
            split_train, split_test = get_train_test_data(folder_path('metal_plate'), seed=body.seed)
            dataset_options = {'cache_dir': data.cache_dir, 'progress': cache_progress}
            PatchDataset = CachedPatchDataset
            if body.variant == 'patch_multiclass':
                from CV.models.object_detection.multiclass_patch_resnet import MulticlassPatchResNet
                classes = tuple(sorted({s.label for s in split_train + split_test}))
                model = MulticlassPatchResNet(classes)
                PatchDataset = CachedMulticlassPatchDataset
                dataset_options['class_names'] = classes
            else:
                model = PatchResNet()
                classes = CLASSES
            datasets = [PatchDataset(samples, folder_path('metal_plate'), model.transform,
                                     body.patch_size, body.patch_size//2, **dataset_options) for samples in (split_train, split_test)]
            if {r['target'] for r in datasets[0].records} != set(range(len(classes))):
                raise ValueError('Training needs annotated patches for every output class')
            training = data.loader(datasets[0], batch_size=body.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(body.seed))
            testing = data.loader(datasets[1], batch_size=body.batch_size)
        else:
            training, testing, classes = get_data_loaders(folder_path('metal_plate'), body.batch_size, body.seed)
            model = PlateResNet(classes)
            training = data.loader(CachedPlateDataset(training.dataset.samples, classes, model.transform,
                                                     data.cache_dir, cache_progress), batch_size=body.batch_size,
                                   shuffle=True, generator=torch.Generator().manual_seed(body.seed))
        device = _device()
        trainer = train
        class_weights = [1.0] * len(classes)
        if body.variant == 'defect_weighted':
            from CV.models.object_detection.weighted_resnet import train_defect_weighted, defect_class_weights
            trainer = train_defect_weighted
            class_weights = defect_class_weights(classes).tolist()
        history = trainer(model, training, body.epochs, body.learning_rate, device,
                        progress=lambda epoch, batch, total: models._progress(
                            job_id, f'Training epoch {epoch+1}/{body.epochs} on {device}', epoch*total+batch, body.epochs*total))
        run_id = uuid.uuid4().hex
        directory = results.data_root('cnn') / run_id
        artifact = ARTIFACT_ROOT / run_id
        artifact.mkdir(parents=True, exist_ok=True)
        temporary = artifact / 'model.tmp'
        torch.save({'state_dict': model.cpu().state_dict(), 'class_names': list(classes),
                    'variant': body.variant, 'class_weights': class_weights}, temporary)
        temporary.replace(artifact / 'model.pt')
        root = folder_path('metal_plate').resolve().parent
        inventory = {name: [{**asdict(s), 'path': Path(s.path).relative_to(root).as_posix()} for s in data.dataset.samples]
                     for name, data in [('train', training), ('test', testing)]}
        results.write_json(directory / 'split.json', inventory)
        results.write_csv(directory / 'split.csv', [{'membership': name, **s} for name, rows in inventory.items() for s in rows])
        data.close_workers()
        models._progress(job_id, 'Evaluating training images')
        if body.variant in ('patch', 'patch_multiclass'):
            from CV.models.object_detection.patch_resnet import evaluate_patches
            if body.variant == 'patch_multiclass':
                from CV.models.object_detection.multiclass_patch_resnet import evaluate_multiclass_patches as evaluate_patches
            evaluation, patch_records = evaluate_patches(model, data.loader(training.dataset, batch_size=body.batch_size), device)
            results.write_csv(directory / 'training_patches.csv', patch_records)
        else:
            evaluation = evaluate(model, data.loader(training.dataset, batch_size=body.batch_size), device)
        training_metrics = _export_evaluation(directory, 'training', evaluation)
        weights = _weight_view(model)
        results.write_json(directory / 'weights.json', weights)
        results.write_csv(directory / 'layer_weights.csv', weights['layers'])
        results.write_csv(directory / 'classifier_weights.csv', [
            {'class': label, 'bias': weights['classifier_bias'][i],
             **{f'channel_{j}': value for j, value in enumerate(weights['classifier_weights'][i])}}
            for i, label in enumerate(classes)])
        results.write_csv(directory / 'history.csv', history)
        summary = {'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                   'config': body.model_dump(), 'input_signature': _input_signature(),
                   'class_names': list(classes), 'class_weights': class_weights, 'history': history,
                   'split_counts': {name: dict(Counter(('good' if s['label'] == 'good' else 'defect') if body.variant == 'patch' else s['label'] for s in rows)) for name, rows in inventory.items()},
                   'training': training_metrics, 'test': None}
        results.write_json(directory / 'summary.json', summary)
        return {'run_id': run_id}
    return models._submit('cnn_train', run)


def _load_model(run_id):
    import torch
    from CV.models.object_detection.resnet import PlateResNet
    checkpoint = torch.load(_checkpoint(run_id), map_location='cpu', weights_only=True)
    if checkpoint.get('variant') == 'patch_multiclass':
        from CV.models.object_detection.multiclass_patch_resnet import MulticlassPatchResNet
        model = MulticlassPatchResNet(checkpoint['class_names'], weights=None)
    elif checkpoint.get('variant') == 'patch':
        from CV.models.object_detection.patch_resnet import PatchResNet
        model = PatchResNet(weights=None)
    else:
        model = PlateResNet(checkpoint['class_names'], weights=None)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def _inspection(model, tensor, row):
    inspection = model.inspect(tensor[None])
    rgb = tensor.cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip((rgb * np.array([.229, .224, .225]) + np.array([.485, .456, .406]))*255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cam = inspection['activation_maps'][0].numpy()
    heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = cam[:, :, None]*.55
    overlay = np.clip(bgr*(1-alpha)+heat*alpha, 0, 255).astype(np.uint8)
    features = inspection['features'][0].numpy()
    maps = inspection['feature_maps'][0].numpy()
    top = np.argsort(features)[-8:][::-1]
    return {**row, 'input_image': _png(bgr), 'activation_image': _png(overlay),
            'features': features.tolist(),
            'feature_maps': [{'channel': int(i), 'activation': float(features[i]),
                              'image': _png(cv2.resize(cv2.applyColorMap(
                                  np.clip(maps[i]/max(float(maps[i].max()), 1e-8)*255, 0, 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO), (112, 112), interpolation=cv2.INTER_NEAREST))} for i in top]}


def test_cnn(body: RunRequest):
    directory = _directory(body.run_id)
    _checkpoint(body.run_id)
    def run(job_id):
        from CV.models.object_detection.cached_data import RunData
        config = results.read_json(directory / 'summary.json')['config']
        with RunData(config.get('num_workers', 2)) as data:
            return run_cached(job_id, data)

    def run_cached(job_id, data):
        from CV.models.object_detection.cached_data import CachedPlateDataset
        from CV.models.data_shuffle import ImageSample
        from CV.models.object_detection.resnet import evaluate
        model = _load_model(body.run_id)
        summary = results.read_json(directory / 'summary.json')
        if summary.get('input_signature') != _input_signature():
            raise ValueError('CNN input pipeline changed; retrain before computing new predictions')
        records = results.read_json(directory / 'split.json')['test']
        samples = [ImageSample(**{**r, 'path': str(image_path(r['path']))}) for r in records]
        # Testing always uses this run's saved holdout inventory, never a new split.
        import hashlib
        if any(hashlib.sha256(Path(s.path).read_bytes()).hexdigest() != s.sha256 for s in samples):
            raise ValueError('A held-out image changed since training; retrain before testing')
        if summary['config'].get('variant') in ('patch', 'patch_multiclass'):
            from .cnn_patch import test_patch_run
            return test_patch_run(directory, model, summary, samples, job_id, data)
        dataset = CachedPlateDataset(samples, model.class_names, model.transform, data.cache_dir,
                                     lambda done, total: models._progress(job_id, 'Caching segmented plates', done, total))
        loader = data.loader(dataset, batch_size=summary['config']['batch_size'])
        models._progress(job_id, 'Evaluating held-out images')
        evaluation = evaluate(model, loader, _device())
        feature_rows = []
        for index, row in enumerate(evaluation['predictions']):
            models._progress(job_id, 'Saving prediction activation maps', index, len(dataset))
            tensor, _, _ = dataset[index]
            row['path'] = records[index]['path']
            inspection = _inspection(model, tensor, row)
            results.write_json(directory / 'inspections' / f'{index}.json.gz', inspection)
            feature_rows.append({'path': row['path'], 'actual': row['actual'], 'prediction': row['prediction'],
                                 **{f'channel_{j}': value for j, value in enumerate(inspection['features'])}})
        results.write_csv(directory / 'test_features.csv', feature_rows)
        summary['test'] = _export_evaluation(directory, 'test', evaluation)
        results.write_json(directory / 'summary.json', summary)
        return {'run_id': body.run_id}
    return models._submit('cnn_test', run)


def cnn_weights(run_id: str):
    return results.read_json(_directory(run_id) / 'weights.json')


def cnn_predictions(run_id: str):
    path = _directory(run_id) / 'test.json'
    return results.read_json(path)['predictions'] if path.exists() else []


def cnn_inspect(run_id: str, image_index: int):
    path = _directory(run_id) / 'inspections' / f'{image_index}.json.gz'
    if image_index < 0 or not path.exists():
        raise HTTPException(404, 'Prediction not available; test this run first')
    return results.read_json(path)


def save_cnn(body: RunRequest):
    _directory(body.run_id)
    with results._save_lock:
        destination = results.saved_root('cnn') / body.run_id / 'model.pt'
        if not destination.exists():
            source = _checkpoint(body.run_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix('.tmp')
            try:
                shutil.copy2(source, temp)
                temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)
    return {'saved': True}
