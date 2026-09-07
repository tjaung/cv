from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights

from CV.models.data_shuffle import ImageSample
from CV.models.object_detection.cached_data import CachedPlateDataset, CachedPatchDataset, RunData


class CachedDataTests(unittest.TestCase):
    def test_spawned_batches_match_and_segmentation_runs_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples = []
            for i in range(3):
                path = Path(temporary) / f'{i}.png'
                cv2.imwrite(str(path), np.full((48, 48, 3), (20+i, 50, 80), np.uint8))
                samples.append(ImageSample(str(path), 'good', 'train', str(i)))
            transform = ResNet18_Weights.DEFAULT.transforms()
            def segment(image):
                return SimpleNamespace(plate=image, plate_mask=np.ones(image.shape[:2], np.uint8))
            with patch('CV.models.object_detection.resnet.segment_plate', side_effect=segment) as segmentation:
                # Exercise spawn from a background job thread, like the server.
                def job():
                    with RunData(2) as data:
                        directory = data.cache_dir
                        dataset = CachedPlateDataset(samples, ('good', 'a', 'b', 'c'), transform, directory)
                        expected = torch.stack([dataset[i][0] for i in range(3)])
                        loader = data.loader(dataset, batch_size=2)
                        first = list(loader)
                        processes = list(loader._iterator._workers)
                        second = list(loader)
                        self.assertEqual([p.pid for p in processes], [p.pid for p in loader._iterator._workers])
                        torch.testing.assert_close(torch.cat([batch[0] for batch in first]), expected)
                        torch.testing.assert_close(torch.cat([batch[0] for batch in second]), expected)
                        self.assertEqual([p for batch in first for p in batch[2]], [s.path for s in samples])
                        self.assertEqual(segmentation.call_count, 3)
                    self.assertFalse(directory.exists())
                    self.assertTrue(all(not p.is_alive() for p in processes))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(job).result(timeout=60)

    def test_patch_revisits_and_failed_job_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'good.png'
            image = np.full((96, 96, 3), (20, 50, 80), np.uint8)
            cv2.imwrite(str(path), image)
            sample = ImageSample(str(path), 'good', 'train', 'example')
            with patch('CV.models.object_detection.patch_resnet.segment_plate', return_value=SimpleNamespace(plate=image, plate_mask=np.ones((96, 96), np.uint8))) as segmentation:
                with self.assertRaisesRegex(RuntimeError, 'job failed'):
                    with RunData(0) as data:
                        directory = data.cache_dir
                        dataset = CachedPatchDataset([sample], temporary, ResNet18_Weights.DEFAULT.transforms(), 64, 32, cache_dir=directory)
                        for _ in range(3):
                            for i in range(len(dataset)):
                                dataset[i]
                        self.assertEqual(segmentation.call_count, 1)
                        self.assertEqual(len(dataset), 4)
                        raise RuntimeError('job failed')
                self.assertFalse(directory.exists())
