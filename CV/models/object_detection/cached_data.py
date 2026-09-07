"""Per-job segmented-plate cache and spawn-safe parallel batch loading."""
from collections import OrderedDict
from pathlib import Path
import tempfile
import uuid

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from . import resnet, patch_resnet
from .resnet import PlateDataset
from .patch_resnet import PatchDataset
from .multiclass_patch_resnet import MulticlassPatchDataset


class PlateCache:
    """Store lossless BGR + mask arrays; workers open them as read-only memmaps."""
    def __init__(self, samples, directory, segmenter, progress=None):
        self.directory = Path(directory) / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.loaded = OrderedDict()
        for index, sample in enumerate(samples):
            if progress:
                progress(index, len(samples))
            image = cv2.imread(sample.path)
            if image is None:
                raise ValueError(f'Could not read image: {sample.path}')
            result = segmenter(image)
            if not result.plate_mask.any():
                raise ValueError(f'No plate found: {sample.path}')
            np.save(self.directory / f'{index}.npy',
                    np.dstack((result.plate, result.plate_mask)), allow_pickle=False)

    def get(self, index):
        if index not in self.loaded:
            self.loaded[index] = np.load(self.directory / f'{index}.npy', mmap_mode='r', allow_pickle=False)
            if len(self.loaded) > 16:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        array = self.loaded[index]
        return array[:, :, :3], array[:, :, 3]

    def __getstate__(self):
        # Never pickle loaded image buffers into macOS spawned workers.
        return {'directory': self.directory, 'loaded': OrderedDict()}


class CachedPlateDataset(PlateDataset):
    def __init__(self, samples, class_names, transform, cache_dir, progress=None):
        super().__init__(samples, class_names, transform)
        self.cache = PlateCache(self.samples, cache_dir, resnet.segment_plate, progress)

    def __getitem__(self, index):
        sample = self.samples[index]
        plate, _ = self.cache.get(index)
        rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb)), self.class_to_idx[sample.label], sample.path


class CachedPatchDataset(PatchDataset):
    def __init__(self, samples, root, transform, patch_size, stride, *, cache_dir, progress=None):
        self.cache = PlateCache(samples, cache_dir, patch_resnet.segment_plate, progress)
        super().__init__(samples, root, transform, patch_size, stride)

    def plate(self, index):
        return self.cache.get(index)


class CachedMulticlassPatchDataset(MulticlassPatchDataset):
    def __init__(self, samples, root, transform, patch_size, stride, *, class_names, cache_dir, progress=None):
        self.cache = PlateCache(samples, cache_dir, patch_resnet.segment_plate, progress)
        super().__init__(samples, root, transform, patch_size, stride, class_names=class_names)

    def plate(self, index):
        return self.cache.get(index)


def initialize_worker(_worker_id):
    # Each process loads/transforms images; avoid nested CPU thread pools.
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


class RunData:
    """Keep workers alive across epochs, then stop them before removing cache."""
    def __init__(self, num_workers=2):
        if not 0 <= num_workers <= 8:
            raise ValueError('num_workers must be between 0 and 8')
        self.num_workers = num_workers
        self.loaders = []

    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='plate-cnn-cache-')
        self.cache_dir = Path(self.temporary.name)
        return self

    def loader(self, dataset, batch_size, shuffle=False, generator=None):
        options = {}
        if self.num_workers:
            options = dict(multiprocessing_context='spawn', persistent_workers=True,
                           prefetch_factor=2, worker_init_fn=initialize_worker)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            generator=generator, num_workers=self.num_workers, **options)
        self.loaders.append(loader)
        return loader

    def close_workers(self):
        for loader in self.loaders:
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                # DataLoader has no public close for persistent worker processes.
                iterator._shutdown_workers()
                loader._iterator = None
        self.loaders.clear()

    def __exit__(self, *_exc):
        try:
            self.close_workers()
        finally:
            self.temporary.cleanup()
