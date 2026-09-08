from sklearn.neighbors import KNeighborsClassifier
from .base import ImageClassifier


class KNNClassifier(ImageClassifier):
    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean', variance_target=.95, patch_size=64):
        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric),
                         variance_target=variance_target, patch_size=patch_size)
