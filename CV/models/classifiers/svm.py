from sklearn.svm import SVC
from .base import ImageClassifier


class SVMClassifier(ImageClassifier):
    def __init__(self, C=1., kernel='rbf', gamma='scale', class_weight='balanced', variance_target=.95, patch_size=64):
        super().__init__(SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight),
                         variance_target=variance_target, patch_size=patch_size)
