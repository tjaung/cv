"""Four-class ResNet baseline on segmented color plates."""

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from CV.preprocessing import segment_plate
from .cnn import get_train_test_data


class PlateDataset(Dataset):
    def __init__(self, samples, class_names, transform):
        self.samples = tuple(samples)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        if any(s.label not in self.class_to_idx for s in self.samples):
            raise ValueError('Sample label is missing from the shared class mapping')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(sample.path)
        if image is None:
            raise ValueError(f'Could not read image: {sample.path}')
        segmented = segment_plate(image)
        if not segmented.plate_mask.any():
            raise ValueError(f'No plate found: {sample.path}')
        rgb = cv2.cvtColor(segmented.plate, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb)), self.class_to_idx[sample.label], sample.path


def get_data_loaders(root, batch_size=16, seed=42, num_workers=0):
    """Use the same 80/20 inventory and seed as the classical classifiers."""
    train, test = get_train_test_data(root, test_size=.2, seed=seed)
    class_names = tuple(sorted({sample.label for sample in train + test}))
    if len(class_names) != 4:
        raise ValueError(f'Expected four classes, found {class_names}')
    transform = ResNet18_Weights.DEFAULT.transforms()
    training = PlateDataset(train, class_names, transform)
    testing = PlateDataset(test, class_names, transform)
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(training, batch_size=batch_size, shuffle=True,
                       generator=generator, num_workers=num_workers),
            DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            class_names)


class PlateResNet(nn.Module):
    def __init__(self, class_names, weights=ResNet18_Weights.DEFAULT):
        super().__init__()
        self.class_names = tuple(class_names)
        if len(self.class_names) != 4 or len(set(self.class_names)) != 4:
            raise ValueError('Supply four distinct class names in the dataset mapping order')
        self.network = resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, 4)
        self.transform = ResNet18_Weights.DEFAULT.transforms()

    def feature_maps(self, images):
        net = self.network
        x = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        return net.layer4(net.layer3(net.layer2(net.layer1(x))))

    def forward(self, images):
        maps = self.feature_maps(images)
        vectors = self.network.avgpool(maps).flatten(1)
        return self.network.fc(vectors)

    @torch.inference_mode()
    def inspect(self, images, class_index=None):
        """Return logits, 512-D features, feature maps and class activation maps.

        CAMs show positive class evidence, not defect masks. Their coordinates
        correspond to the weights-transformed input (including its center crop).
        """
        was_training = self.training
        self.eval()
        try:
            images = images.to(next(self.parameters()).device)
            maps = self.feature_maps(images)
            features = self.network.avgpool(maps).flatten(1)
            logits = self.network.fc(features)
            predicted = logits.argmax(1)
            if class_index is not None and not 0 <= class_index < 4:
                raise ValueError('class_index must be between 0 and 3')
            targets = predicted if class_index is None else torch.full_like(predicted, class_index)
            cams = torch.einsum('bc,bchw->bhw', self.network.fc.weight[targets], maps).relu()
            cams = nn.functional.interpolate(cams[:, None], size=images.shape[-2:],
                                              mode='bilinear', align_corners=False)[:, 0]
            cams = cams / cams.flatten(1).amax(1)[:, None, None].clamp_min(1e-8)
            return {key: value.cpu() for key, value in {
                'logits': logits, 'probabilities': logits.softmax(1), 'predictions': predicted,
                'features': features, 'feature_maps': maps, 'activation_maps': cams,
                'target_classes': targets}.items()}
        finally:
            self.train(was_training)


def train(model, loader, epochs=5, learning_rate=1e-4, device='cpu', progress=None):
    """Fine-tune the entire network; the held-out loader is never used here."""
    if epochs < 1 or learning_rate <= 0 or not len(loader.dataset):
        raise ValueError('Require positive epochs/rate and nonempty training data')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = correct = count = 0
        for batch, (images, labels, _) in enumerate(loader):
            if progress:
                progress(epoch, batch, len(loader))
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            count += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        history.append({'epoch': epoch + 1, 'loss': loss_sum/count, 'accuracy': correct/count})
    return history


@torch.inference_mode()
def evaluate(model, loader, device='cpu'):
    """Report held-out per-class precision/recall/F1 and individual predictions."""
    from sklearn.metrics import classification_report, confusion_matrix
    model.to(device)
    was_training = model.training
    model.eval()
    truth, predictions, rows = [], [], []
    loss_sum = 0.
    try:
        for images, labels, paths in loader:
            logits = model(images.to(device))
            loss_sum += nn.functional.cross_entropy(logits, labels.to(device), reduction='sum').item()
            probabilities = logits.softmax(1).cpu()
            predicted = probabilities.argmax(1).tolist()
            truth.extend(labels.tolist())
            predictions.extend(predicted)
            rows.extend({'path': path, 'actual': model.class_names[label],
                         'prediction': model.class_names[pred], 'correct': label == pred,
                         'probabilities': probability.tolist()}
                        for path, label, pred, probability in zip(paths, labels.tolist(), predicted, probabilities))
        if not truth:
            raise ValueError('No evaluation samples')
        return {'class_names': list(model.class_names), 'loss': loss_sum/len(truth),
                'accuracy': float(np.mean(np.asarray(truth) == predictions)),
                'report': classification_report(truth, predictions, labels=list(range(4)),
                                                 target_names=model.class_names, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(truth, predictions, labels=list(range(4))).tolist(),
                'predictions': rows}
    finally:
        model.train(was_training)
