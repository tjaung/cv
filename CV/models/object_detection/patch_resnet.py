"""Binary patch ResNet, supervised by pixel masks after an image-level split."""
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix

from CV.preprocessing import segment_plate
from CV.models.patch_geometry import patch_boxes
from .resnet import PlateResNet

CLASSES = ('good', 'defect')


class PatchResNet(PlateResNet):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        nn.Module.__init__(self)
        self.class_names = CLASSES
        self.network = resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, 2)
        # Preserve all patch pixels, including scratches at a patch boundary.
        self.transform = ResNet18_Weights.DEFAULT.transforms(resize_size=224, crop_size=224)



def region(image, box):
    return image[box['top']:box['bottom'], box['left']:box['right']]


class PatchDataset(Dataset):
    def __init__(self, samples, root, transform, patch_size=64, stride=32):
        self.samples, self.root, self.transform = tuple(samples), Path(root), transform
        self.records = []
        for index, sample in enumerate(self.samples):
            plate, mask = self.plate(index)
            if sample.label == 'good':
                truth = np.zeros(mask.shape, np.uint8)
            else:
                if sample.original_split != 'test':
                    raise ValueError(f'No supported ground-truth mask for {sample.path}')
                path = self.root / 'ground_truth' / sample.label / f'{Path(sample.path).stem}_mask.png'
                truth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if truth is None or truth.shape != mask.shape:
                    raise ValueError(f'Missing or mismatched defect mask: {path}')
            boxes = patch_boxes(mask, patch_size, stride)
            if not boxes:
                raise ValueError(f'No eligible plate patches: {sample.path}')
            for box in boxes:
                # Any annotated defect pixel inside the plate makes a positive patch.
                positive = bool(np.any((region(truth, box) > 0) & (region(mask, box) > 0)))
                self.records.append({'image_index': index, **box, 'target': int(positive)})

    @lru_cache(maxsize=8)
    def plate(self, index):
        image = cv2.imread(self.samples[index].path)
        if image is None:
            raise ValueError(f'Could not read {self.samples[index].path}')
        result = segment_plate(image)
        return result.plate, result.plate_mask

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        plate, _ = self.plate(record['image_index'])
        rgb = cv2.cvtColor(region(plate, record), cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb)), record['target'], index


def metrics(targets, probabilities):
    truth = np.asarray(targets)
    p = np.asarray(probabilities)
    predicted = (p >= .5).astype(int)
    assigned = np.where(truth == 1, p, 1-p)
    return {'accuracy': float(np.mean(truth == predicted)),
            'loss': float(-np.log(np.clip(assigned, 1e-7, 1)).mean()),
            'report': classification_report(truth, predicted, labels=[0, 1], target_names=CLASSES, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(truth, predicted, labels=[0, 1]).tolist()}


@torch.inference_mode()
def evaluate_patches(model, loader, device='cpu'):
    model.to(device).eval()
    records = []
    for images, labels, indices in loader:
        probabilities = model(images.to(device)).softmax(1)[:, 1].cpu().tolist()
        for index, label, probability in zip(indices.tolist(), labels.tolist(), probabilities):
            records.append({**loader.dataset.records[index], 'target': label, 'score': probability,
                            'anomalous': probability >= .5})
    rows = []
    for i, sample in enumerate(loader.dataset.samples):
        score = max(r['score'] for r in records if r['image_index'] == i)
        actual = 'good' if sample.label == 'good' else 'defect'
        prediction = 'defect' if score >= .5 else 'good'
        rows.append({'path': sample.path, 'actual': actual, 'prediction': prediction,
                     'correct': prediction == actual, 'probabilities': [1-score, score], 'source_class': sample.label})
    return {**metrics([r['actual'] == 'defect' for r in rows], [r['probabilities'][1] for r in rows]),
            'class_names': list(CLASSES), 'predictions': rows,
            'patch_metrics': metrics([r['target'] for r in records], [r['score'] for r in records])}, records
