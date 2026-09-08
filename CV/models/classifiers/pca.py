from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
from .base import ImageClassifier


class PCAClassifier(ImageClassifier):
    def __init__(self, variance_target=.95, patch_size=64, metric='euclidean'):
        if metric not in ('euclidean', 'manhattan'):
            raise ValueError('metric must be euclidean or manhattan')
        if variance_target is None:
            raise ValueError('PCAClassifier requires a variance target')

        super().__init__(NearestCentroid(metric=metric, priors='uniform'), variance_target=variance_target, patch_size=patch_size)

    def project(self, features):
        return self.pipeline[:-1].transform(features)

    def distances(self, features):
        classifier = self.pipeline['classifier']
        return pairwise_distances(self.project(features), classifier.centroids_, metric=classifier.metric)

    def pca_summary(self):
        pca = self.pipeline['pca']
        return {'components': pca.components_.tolist(), 'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'retained_variance': float(pca.explained_variance_ratio_.sum()), 'classes': self.pipeline.classes_.tolist()}
