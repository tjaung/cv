import csv
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from CV.models.feature_sets import extract_model_features, feature_names, feature_signature

FEATURE_SET = 'lab_sobel_hog_frangi'

class ImageClassifier:
    def __init__(self, estimator, *, variance_target=.95, patch_size=64):
        if variance_target is not None and not 0 < variance_target < 1:
            raise ValueError('variance_target must be in (0, 1), or None')
        if not isinstance(patch_size, int) or not 16 <= patch_size <= 256:
            raise ValueError('patch_size must be an integer from 16 to 256')
        self.patch_size = patch_size
        self.signature = feature_signature(FEATURE_SET)
        self.feature_names = [f'{stat}_{name}' for stat in ('mean', 'std') for name in feature_names(FEATURE_SET)]
        steps = [('scaler', StandardScaler())]
        if variance_target is not None:
            steps.append(('pca', PCA(n_components=variance_target, svd_solver='full')))
        self.pipeline = Pipeline(steps + [('classifier', estimator)])

    def extract_features(self, samples):
        if self.signature != feature_signature(FEATURE_SET):
            raise ValueError('Feature pipeline changed; create and train a new classifier')

        vectors = []
        for sample in samples:
            image = cv2.imread(sample.path)
            if image is None:
                raise ValueError(f'Could not decode {sample.path}')
            try:
                patches = extract_model_features(image, sample.path, self.patch_size, feature_set=FEATURE_SET)
            except ValueError as error:
                raise ValueError(f'{sample.path}: {error}') from error
            vectors.append(np.concatenate((patches.values.mean(axis=0), patches.values.std(axis=0))))

        if not vectors:
            raise ValueError('No images supplied')
        
        return np.vstack(vectors)

    def fit(self, features, labels):
        x, y = np.asarray(features, dtype=float), np.asarray(labels)
        if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all() or y.ndim != 1 or len(x) != len(y):
            raise ValueError('Expected finite feature rows and one class label per row')
        if len(np.unique(y)) < 2:
            raise ValueError('Need at least two training classes')

        neighbors = getattr(self.pipeline['classifier'], 'n_neighbors', 1)

        if neighbors > len(x):
            raise ValueError('n_neighbors exceeds the number of training images')
        self.pipeline.fit(x, y)

        return self

    def fit_images(self, samples, csv_path=None):
        samples = tuple(samples)
        x = self.extract_features(samples)
        self.fit(x, [s.label for s in samples])
        self.training_hashes = {s.sha256 for s in samples}

        if csv_path is not None:
            self.export_features(csv_path, samples, x)

        return self

    def predict(self, features):
        check_is_fitted(self.pipeline)
        return self.pipeline.predict(features)

    def predict_images(self, samples):
        return self.predict(self.extract_features(samples))

    def test(self, features, labels):
        predicted = self.predict(features)
        labels = np.asarray(labels)
        classes = sorted(set(labels.tolist()) | set(predicted.tolist()) | set(self.pipeline.classes_.tolist()))

        return {'accuracy': float(accuracy_score(labels, predicted)), 'classes': classes,
                'confusion_matrix': confusion_matrix(labels, predicted, labels=classes).tolist(),
                'report': classification_report(labels, predicted, labels=classes, output_dict=True, zero_division=0),
                'predictions': predicted.tolist()}

    def test_images(self, samples):
        samples = tuple(samples)
        if getattr(self, 'training_hashes', set()) & {s.sha256 for s in samples}:
            raise ValueError('Test images overlap training content')

        return self.test(self.extract_features(samples), [s.label for s in samples])

    def export_features(self, path, samples, features):
        with Path(path).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_path', 'class', 'original_split'] + self.feature_names)
            for sample, row in zip(samples, features, strict=True):
                writer.writerow([sample.path, sample.label, sample.original_split, *row])

    def save(self, path):
        check_is_fitted(self.pipeline)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path):
        """Load only trusted local artifacts: joblib uses pickle serialization."""
        model = joblib.load(path)

        if not isinstance(model, cls):
            raise ValueError('Wrong classifier artifact type')

        return model
