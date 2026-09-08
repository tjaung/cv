"""Defect-weighted training for the same four-class ResNet architecture."""
import torch
from torch import nn


def defect_class_weights(class_names, device='cpu'):
    """Map weights by label so a different output order cannot weight good as bad."""
    if len(class_names) != 4 or len(set(class_names)) != 4 or 'good' not in class_names:
        raise ValueError('Expected good and three distinct defect classes')
    return torch.tensor([1.0 if name == 'good' else 2.0 for name in class_names],
                        dtype=torch.float32, device=device)


def train_defect_weighted(model, loader, epochs=5, learning_rate=1e-4, device='cpu', progress=None):
    """Fine-tune with twice the relative loss weight for defect examples."""
    if epochs < 1 or learning_rate <= 0 or not len(loader.dataset):
        raise ValueError('Require positive epochs/rate and nonempty training data')
    model.to(device)
    weights = defect_class_weights(model.class_names, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = weight_sum = correct = count = 0
        for batch, (images, labels, _) in enumerate(loader):
            if progress:
                progress(epoch, batch, len(loader))
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # CrossEntropyLoss's weighted mean divides by target weights,
            # not sample count. Use that same denominator for the epoch curve.
            batch_weight = weights[labels].sum().item()
            loss_sum += loss.item() * batch_weight
            weight_sum += batch_weight
            count += len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        history.append({'epoch': epoch + 1, 'loss': loss_sum/weight_sum, 'accuracy': correct/count})
    return history
