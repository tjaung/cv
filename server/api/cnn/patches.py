"""Persist binary patch predictions and overlays in original image coordinates."""
import cv2
import numpy as np
from CV.models.object_detection.patch_resnet import evaluate_patches
from CV.models.object_detection.cached_data import CachedPatchDataset, CachedMulticlassPatchDataset
from . import handlers as cnn
from ..anomaly_detection import models
from ..shared import result_store as results
from ..features.handlers import _png
from ..shared.os_helpers import folder_path


def test_patch_run(directory, model, summary, samples, job_id, data):
    size = summary['config']['patch_size']
    dataset_class, evaluator = CachedPatchDataset, evaluate_patches
    options = {'cache_dir': data.cache_dir, 'progress': lambda done, total: models._progress(job_id, 'Caching segmented plates', done, total)}
    if summary['config']['variant'] == 'patch_multiclass':
        from CV.models.object_detection.multiclass_patch_resnet import evaluate_multiclass_patches
        dataset_class, evaluator = CachedMulticlassPatchDataset, evaluate_multiclass_patches
        options['class_names'] = model.class_names
    dataset = dataset_class(samples, folder_path('metal_plate'), model.transform, size, size//2, **options)
    evaluation, patches = evaluator(model, data.loader(dataset, batch_size=summary['config']['batch_size']), cnn._device())
    cnn._export_evaluation(directory, 'test', evaluation)
    results.write_csv(directory / 'test_patches.csv', [
        {'path': evaluation['predictions'][p['image_index']]['path'], **p} for p in patches])
    for i, row in enumerate(evaluation['predictions']):
        models._progress(job_id, 'Saving localized patch predictions', i, len(samples))
        plate, mask = dataset.plate(i)
        overlay = plate.copy()
        local = [p for p in patches if p['image_index'] == i]
        for p in local:
            if p['anomalous']:
                colors = {'major_rust': (0, 140, 255), 'scratches': (255, 160, 0), 'total_rust': (180, 0, 220)}
                color = colors.get(p.get('prediction'), (0, 0, 255))
                cv2.rectangle(overlay, (p['left'], p['top']), (p['right']-1, p['bottom']-1), color, 2)
        results.write_json(directory / 'inspections' / f'{i}.json.gz', {
            **row, 'input_image': _png(plate), 'activation_image': _png(overlay),
            'patches': local, 'width': plate.shape[1], 'height': plate.shape[0],
            'features': [], 'feature_maps': [],
            'plate_coverage': float(np.mean(mask > 0))})
    summary['test'] = {k: v for k, v in evaluation.items() if k != 'predictions'}
    results.write_json(directory / 'summary.json', summary)
    return {'run_id': directory.name}
