import csv
import json
from pathlib import Path

import numpy as np

from ..patch_features import FEATURE_NAMES, FEATURE_VERSION, PatchData, extract_patch_features, pipeline_signature


class PCAAnomalyDetector:

    def __init__(self, variance_target=0.95, patch_size=64, min_coverage=0.5, distance_metric="squared_l2"):
        if not 0 < variance_target < 1:
            raise ValueError('Variance target must be between 0 and 1')
        if distance_metric not in ('l1', 'l2', 'mahalanobis', 'squared_l2'):
            raise ValueError('Unknown distance metric')

        self.config = {'variance_target': variance_target, 'patch_size': patch_size,
                       'distance_metric': distance_metric, 'min_coverage': min_coverage, 'quantile': .99, 'source': 'normalized_segmented_before_glare',
                       'feature_version': FEATURE_VERSION, 'pipeline_signature': pipeline_signature()}
        self.fitted = False

    def fit(self, training: PatchData, calibration: PatchData):
        x, c = training.values, calibration.values
        for values, records in [(x, training.records), (c, calibration.records)]:
            if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) != len(records) or not np.isfinite(values).all():
                raise ValueError('Invalid feature matrix or records')
        if len(x) < 3 or not len(c):
            raise ValueError('Need at least three training patches and held-out calibration patches')

        training_ids = {r['image_id'] for r in training.records}
        calibration_ids = {r['image_id'] for r in calibration.records}

        if training_ids & calibration_ids:
            raise ValueError('Training and calibration images must be disjoint')

        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.scale = np.where(std > 1e-8, std, 1)
        self.train_raw, self.cal_raw = x.copy(), c.copy()
        self.train_records, self.cal_records = list(training.records), list(calibration.records)
        standardized = (x - self.mean) / self.scale
        self.pca_mean = standardized.mean(axis=0)
        _, singular, vt = np.linalg.svd(standardized - self.pca_mean, full_matrices=False)
        variance = singular ** 2 / (len(x) - 1)

        if variance.sum() <= 1e-12:
            raise ValueError('Training features have no variation; PCA cannot learn a subspace')

        self.explained_ratio = variance / variance.sum()
        rank = int(np.count_nonzero(variance > max(variance[0] * 1e-10, 1e-12)))
        keep = int(np.searchsorted(np.cumsum(self.explained_ratio), self.config['variance_target']) + 1)
        keep = max(1, min(keep, rank, x.shape[1] - 1))
        self.components, self.eigenvalues = vt[:keep].copy(), variance[:keep].copy()

        residual = standardized - ((standardized - self.pca_mean) @ self.components.T @ self.components + self.pca_mean)
        covariance = residual.T @ residual / (len(x) - 1)
        ridge = max(float(np.trace(covariance) / x.shape[1]) * .01, 1e-8)
        self.residual_precision = np.linalg.inv(covariance + ridge * np.eye(x.shape[1]))
        self.config['mahalanobis_ridge'] = ridge
        self.fitted = True
        self.train_z, _, self.train_errors = self._project(x)
        self.cal_z, _, self.cal_errors = self._project(c)
        self.patch_threshold = float(np.quantile(self.cal_errors, .99, method='higher'))
        plate_errors = [float(self.cal_errors[[r['image_id'] == name for r in self.cal_records]].max()) for name in sorted(calibration_ids)]
        self.plate_threshold = float(np.quantile(plate_errors, .99, method='higher'))
        self.cal_plate_errors = plate_errors

        return self

    def _project(self, values):
        if not self.fitted:
            raise ValueError('Model has not been trained')

        standardized = (values - self.mean) / self.scale
        z = (standardized - self.pca_mean) @ self.components.T
        reconstruction = z @ self.components + self.pca_mean
        residual = standardized - reconstruction
        metric = self.config.get('distance_metric', 'squared_l2')

        if metric == 'l1':
            errors = np.abs(residual).sum(axis=1)
        elif metric == 'mahalanobis':
            errors = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', residual, self.residual_precision, residual), 0))
        else:
            errors = np.sum(residual ** 2, axis=1)
            if metric == 'l2':
                errors = np.sqrt(errors)

        return z, reconstruction, errors

    def score(self, patches: PatchData):
        values = patches.values

        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not len(values) or len(patches.records) != len(values) or not np.isfinite(values).all():
            raise ValueError('Invalid patch feature vectors')

        z, reconstructed, errors = self._project(values)
        nearest_distances, nearest_indices = [], []

        reference_norms = np.sum(self.train_z ** 2, axis=1)
        for start in range(0, len(z), 128):
            block = z[start:start + 128]
            squared = np.maximum(np.sum(block ** 2, axis=1)[:, None] + reference_norms - 2 * block @ self.train_z.T, 0)
            indices = np.argmin(squared, axis=1)
            nearest_indices.extend(indices.tolist())
            nearest_distances.extend(np.sqrt(squared[np.arange(len(indices)), indices]).tolist())
        mahalanobis = np.sqrt(np.sum(z ** 2 / np.maximum(self.eigenvalues, 1e-12), axis=1))
        rows = [{**record, 'error': float(errors[i]), 'anomalous': bool(errors[i] > self.patch_threshold),
                 'scores': z[i].tolist(), 'nearest_distance': nearest_distances[i],
                 'nearest_patch': self.train_records[nearest_indices[i]], 'mahalanobis': float(mahalanobis[i])}
                for i, record in enumerate(patches.records)]

        return {'prediction': 'BAD' if errors.max() > self.plate_threshold else 'GOOD',
                'plate_score': float(errors.max()), 'patch_threshold': self.patch_threshold,
                'plate_threshold': self.plate_threshold, 'anomalous_patches': int(np.sum(errors > self.patch_threshold)),
                'patches': rows}, z, reconstructed, errors

    def test_image(self, image, image_id=''):
        if self.config['pipeline_signature'] != pipeline_signature():
            raise ValueError('Preprocessing or feature code changed; retrain the model')

        patches = extract_patch_features(image, image_id, self.config['patch_size'], self.config['min_coverage'])
        result, z, reconstructed, errors = self.score(patches)

        return result, patches, z, reconstructed, errors

    def summary(self):
        return {'config': self.config, 'features': len(FEATURE_NAMES), 'feature_names': FEATURE_NAMES,
                'components': len(self.components), 'retained_variance': float(self.explained_ratio[:len(self.components)].sum()),
                'explained_variance_ratio': self.explained_ratio.tolist(),
                'patch_threshold': self.patch_threshold, 'plate_threshold': self.plate_threshold,
                'training_images': len({r['image_id'] for r in self.train_records}),
                'calibration_images': len({r['image_id'] for r in self.cal_records}),
                'training_patches': len(self.train_raw), 'calibration_patches': len(self.cal_raw),
                'calibration_plate_errors': self.cal_plate_errors,
                'training_errors': self.train_errors.tolist(), 'calibration_errors': self.cal_errors.tolist(),
                'training_points': [{'image_id': r['image_id'], 'patch_id': r['patch_id'], 'scores': z.tolist(), 'error': float(error)}
                                    for r, z, error in zip(self.train_records, self.train_z, self.train_errors)],
                'calibration_points': [{'image_id': r['image_id'], 'patch_id': r['patch_id'], 'scores': z.tolist(), 'error': float(error)}
                                       for r, z, error in zip(self.cal_records, self.cal_z, self.cal_errors)]}

    def export_features(self, path, patches, split):
        z, reconstruction, errors = self._project(patches.values)
        standardized = (patches.values - self.mean) / self.scale
        columns = list(patches.records[0]) + ['split', 'error'] + FEATURE_NAMES
        columns += ['scaled_' + name for name in FEATURE_NAMES] + ['reconstructed_' + name for name in FEATURE_NAMES]
        columns += [f'PC{i + 1}' for i in range(z.shape[1])]
        with Path(path).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for i, row in enumerate(patches.records):
                writer.writerow(list(row.values()) + [split, errors[i]] + patches.values[i].tolist()
                                + standardized[i].tolist() + reconstruction[i].tolist() + z[i].tolist())

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in ('mean', 'scale', 'pca_mean', 'components', 'eigenvalues',
                  'explained_ratio', 'train_raw', 'cal_raw', 'train_z', 'cal_z', 'train_errors', 'cal_errors')}
        if hasattr(self, 'residual_precision'):
            arrays['residual_precision'] = self.residual_precision

        np.savez_compressed(directory / 'model.npz', **arrays)
        metadata = {'config': self.config, 'feature_names': FEATURE_NAMES, 'train_records': self.train_records,
                    'cal_records': self.cal_records, 'patch_threshold': self.patch_threshold,
                    'plate_threshold': self.plate_threshold, 'cal_plate_errors': self.cal_plate_errors}
        (directory / 'model.json').write_text(json.dumps(metadata))
        self.export_features(directory / 'training_features.csv', PatchData(self.train_raw, self.train_records), 'training_good')
        self.export_features(directory / 'calibration_features.csv', PatchData(self.cal_raw, self.cal_records), 'calibration_good')

        with (directory / 'components.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['component', 'eigenvalue'] + FEATURE_NAMES)
            for i, component in enumerate(self.components):
                writer.writerow([f'PC{i + 1}', self.eigenvalues[i]] + component.tolist())

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / 'model.json').read_text())

        if metadata['feature_names'] != FEATURE_NAMES or metadata['config']['feature_version'] != FEATURE_VERSION:
            raise ValueError('Saved model feature schema is incompatible')

        model = cls()
        for name in ('config', 'train_records', 'cal_records', 'patch_threshold', 'plate_threshold', 'cal_plate_errors'):
            setattr(model, name, metadata[name])
        with np.load(directory / 'model.npz', allow_pickle=False) as data:
            for name in data.files:
                setattr(model, name, data[name])
        model.fitted = True

        return model
