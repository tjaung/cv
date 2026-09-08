"""Four-class patch classification with mask-derived local defect-type labels."""
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix

from .patch_resnet import PatchResNet, PatchDataset


def validate_classes(class_names):
    names = tuple(class_names)
    if len(names) != 4 or len(set(names)) != 4 or 'good' not in names:
        raise ValueError('Expected good and three distinct defect classes')
    return names


class MulticlassPatchResNet(PatchResNet):
    def __init__(self, class_names, weights=ResNet18_Weights.DEFAULT):
        super().__init__(weights=weights)
        self.class_names = validate_classes(class_names)
        self.network.fc = nn.Linear(self.network.fc.in_features, 4)


class MulticlassPatchDataset(PatchDataset):
    def __init__(self, samples, root, transform, patch_size=64, stride=32, *, class_names):
        self.class_names = validate_classes(class_names)
        if any(s.label not in self.class_names for s in samples):
            raise ValueError('Sample label is missing from the shared class mapping')
        super().__init__(samples, root, transform, patch_size, stride)
        for record in self.records:
            label = self.samples[record['image_index']].label if record['target'] else 'good'
            record['target'] = self.class_names.index(label)


def multiclass_metrics(targets, probabilities, class_names):
    truth = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities)
    predicted = p.argmax(1)
    indices = list(range(len(class_names)))
    return {'accuracy': float(np.mean(truth == predicted)),
            'loss': float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-7, 1)).mean()),
            'report': classification_report(truth, predicted, labels=indices, target_names=class_names, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(truth, predicted, labels=indices).tolist()}


def aggregate_patches(records, class_names):
    """Allow up to two bad patches when mean good probability is highest.

    Use the mean vector for this good override; otherwise retain the strongest
    non-good patch. Ties do not override a defect prediction. These aggregated
    scores are not calibrated whole-image probabilities.
    """
    if not records:
        raise ValueError('Cannot classify an image without scored patches')
    flagged = [r for r in records if r['anomalous']]
    good = class_names.index('good')
    mean = np.mean([r['probabilities'] for r in records], axis=0)
    if len(flagged) <= 2 and mean[good] > np.max(np.delete(mean, good)):
        return mean.tolist()
    if flagged:
        return max(flagged, key=lambda r: r['score'])['probabilities']
    return min(records, key=lambda r: r['probabilities'][good])['probabilities']


@torch.inference_mode()
def evaluate_multiclass_patches(model, loader, device='cpu'):
    model.to(device).eval()
    names = model.class_names
    good = names.index('good')
    records = []
    for images, labels, indices in loader:
        probabilities = model(images.to(device)).softmax(1).cpu().tolist()
        for index, target, p in zip(indices.tolist(), labels.tolist(), probabilities):
            predicted = int(np.argmax(p))
            records.append({**loader.dataset.records[index], 'target': target,
                            'prediction': names[predicted], 'actual': names[target],
                            'probabilities': p, 'score': p[predicted], 'anomalous': predicted != good})
    rows = []
    for i, sample in enumerate(loader.dataset.samples):
        p = aggregate_patches([r for r in records if r['image_index'] == i], names)
        prediction = names[int(np.argmax(p))]
        rows.append({'path': sample.path, 'actual': sample.label, 'prediction': prediction,
                     'correct': prediction == sample.label, 'probabilities': p, 'source_class': sample.label})
    return {**multiclass_metrics([names.index(r['actual']) for r in rows], [r['probabilities'] for r in rows], names),
            'class_names': list(names), 'predictions': rows,
            'patch_metrics': multiclass_metrics([r['target'] for r in records], [r['probabilities'] for r in records], names)}, records
