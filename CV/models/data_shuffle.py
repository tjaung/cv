from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


@dataclass(frozen=True)
class ImageSample:
    path: str
    label: str
    original_split: str
    sha256: str


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[ImageSample, ...]
    test: tuple[ImageSample, ...]
    seed: int

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def collect_images(root):
    root = Path(root).resolve()

    records = []
    for split in ('train', 'test'):
        directory = root / split
        if not directory.is_dir():
            raise ValueError(f'Missing image split directory: {directory}')
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(directory)
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS or any(p.startswith('.') for p in relative.parts):
                continue
            if len(relative.parts) < 2:
                raise ValueError(f'Image must be inside a class folder: {path}')
            records.append(ImageSample(str(path.resolve()), relative.parts[0], split, hashlib.sha256(path.read_bytes()).hexdigest()))
    
    if not records:
        raise ValueError('No labeled images found')

    return tuple(records)


def sample_images(images, test_size=.2, seed=42):
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1')

    groups, labels, paths = {}, {}, set()
    for sample in images:
        if sample.path in paths:
            raise ValueError(f'Duplicate inventory path: {sample.path}')
        paths.add(sample.path)
        if sample.sha256 in labels and labels[sample.sha256] != sample.label:
            raise ValueError('Identical image content has conflicting class labels')
        labels[sample.sha256] = sample.label
        groups.setdefault(sample.label, {}).setdefault(sample.sha256, []).append(sample)

    if len(groups) < 2:
        raise ValueError('Classification requires at least two classes')
    rng = np.random.default_rng(seed)

    train, test = [], []
    for label, content in sorted(groups.items()):
        keys = sorted(content)
        if len(keys) < 2:
            raise ValueError(f'Class {label!r} needs at least two distinct images')
        order = rng.permutation(len(keys))
        count = min(len(keys) - 1, max(1, round(len(keys) * test_size)))
        for i, index in enumerate(order):
            (test if i < count else train).extend(sorted(content[keys[index]], key=lambda s: s.path))

    rng.shuffle(train)
    rng.shuffle(test)

    return DatasetSplit(tuple(train), tuple(test), seed)


class ImageStore:
    def __init__(self, root):
        self.images = collect_images(root)

    def sample(self, test_size=.2, seed=42):
        return sample_images(self.images, test_size, seed)
