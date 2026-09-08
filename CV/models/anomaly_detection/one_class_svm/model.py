from pathlib import Path

import numpy as np

from ..pca import PCAAnomalyDetector


class OneClassSVMDetector(PCAAnomalyDetector):
    def __init__(self, variance_target=.95, patch_size=64, nu=.05, kernel='rbf', gamma='scale', feature_set='lab_sobel', preprocessing='full'):
        super().__init__(variance_target, patch_size, feature_set=feature_set, preprocessing=preprocessing)
        if not 0 < nu <= 1 or kernel not in ('rbf', 'linear'):
            raise ValueError('Invalid one-class SVM nu or kernel')
        if gamma != 'scale' and (not isinstance(gamma, (float, int)) or gamma <= 0):
            raise ValueError('Gamma must be scale or a positive number')
        self.config.update(model_type='one_class_svm', nu=nu, kernel=kernel,
                           gamma=gamma if kernel == 'rbf' else 'not_used', distance_metric='svm_margin')

    def fit(self, training, calibration):
        from sklearn.svm import OneClassSVM

        # A refit must not reuse the previous decision function during PCA.fit.
        if hasattr(self, 'support_vectors'):
            del self.support_vectors
        super().fit(training, calibration)
        estimator = OneClassSVM(nu=self.config['nu'], kernel=self.config['kernel'],
                               gamma=self.config['gamma'] if self.config['kernel'] == 'rbf' else 'scale',
                               cache_size=200, max_iter=100000)
        estimator.fit(self.train_z)
        if estimator.fit_status_ != 0:
            raise ValueError('One-class SVM did not converge; change parameters')
        self.support_vectors = estimator.support_vectors_.copy()
        self.dual_coef = estimator.dual_coef_[0].copy()
        self.intercept = np.asarray(estimator.intercept_)
        self.config.update(effective_gamma=float(estimator._gamma), support_vectors=len(self.support_vectors))
        self.train_errors = self._project(training.values)[2]
        self.cal_errors = self._project(calibration.values)[2]
        self.patch_threshold = float(np.quantile(self.cal_errors, .99, method='higher'))
        self.cal_plate_errors = [float(self.cal_errors[[r['image_id'] == name for r in self.cal_records]].max())
                                 for name in sorted({r['image_id'] for r in self.cal_records})]
        self.plate_threshold = float(np.quantile(self.cal_plate_errors, .99, method='higher'))
        return self

    def _project(self, values):
        z, reconstruction, errors = super()._project(values)
        if hasattr(self, 'support_vectors'):
            chunks = []
            for start in range(0, len(z), 256):
                block = z[start:start + 256]
                kernel = block @ self.support_vectors.T
                if self.config['kernel'] == 'rbf':
                    squared = np.maximum((block ** 2).sum(axis=1)[:, None]
                                         + (self.support_vectors ** 2).sum(axis=1) - 2 * kernel, 0)
                    kernel = np.exp(-self.config['effective_gamma'] * squared)
                chunks.append(-(kernel @ self.dual_coef + self.intercept[0]))
            errors = np.concatenate(chunks) if chunks else np.empty(0)
        return z, reconstruction, errors

    def save(self, directory):
        super().save(directory)
        path = Path(directory) / 'model.npz'
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        arrays.update(support_vectors=self.support_vectors, dual_coef=self.dual_coef, intercept=self.intercept)
        np.savez_compressed(path, **arrays)
