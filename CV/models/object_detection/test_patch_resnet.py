import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from CV.models.data_shuffle import collect_images, sample_images
from CV.models.object_detection.patch_resnet import PatchDataset, PatchResNet, patch_boxes, evaluate_patches


class PatchCNNTests(unittest.TestCase):
    def test_patch_labels_borders_and_binary_predictions(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ('train', 'test'):
                for label in ('good', 'scratches'):
                    folder = root / split / label
                    folder.mkdir(parents=True)
                    # Defective images are only in the original test folder.
                    if split == 'train' and label != 'good':
                        continue
                    for i in range(2):
                        image = np.full((96, 96, 3), 30 + i + (50 if label == 'scratches' else 0) + (10 if split == 'test' else 0), np.uint8)
                        cv2.imwrite(str(folder / f'{i}.png'), image)
            mask_dir = root / 'ground_truth/scratches'
            mask_dir.mkdir(parents=True)
            for i in range(2):
                mask = np.zeros((96, 96), np.uint8)
                mask[3, 3] = 255  # Tiny scratch must not be lost to an area threshold.
                cv2.imwrite(str(mask_dir / f'{i}_mask.png'), mask)
            split = sample_images(collect_images(root))
            self.assertFalse({s.sha256 for s in split.train} & {s.sha256 for s in split.test})
            model = PatchResNet(weights=None)
            with patch('CV.models.object_detection.patch_resnet.segment_plate', side_effect=lambda image: SimpleNamespace(plate=image, plate_mask=np.ones(image.shape[:2], np.uint8))):
                data = PatchDataset(split.train, root, model.transform, patch_size=64, stride=32)
                bad_index = next(i for i, sample in enumerate(data.samples) if sample.label == 'scratches')
                bad_rows = [r for r in data.records if r['image_index'] == bad_index]
                self.assertEqual({r['target'] for r in bad_rows}, {0, 1})
                self.assertEqual(sum(r['target'] for r in bad_rows), 1)
                self.assertEqual(tuple(data[0][0].shape), (3, 224, 224))
                evaluation, predictions = evaluate_patches(model, DataLoader(data, batch_size=2))
                self.assertEqual(np.shape(evaluation['confusion_matrix']), (2, 2))
                self.assertEqual(len(predictions), len(data))
                for i, row in enumerate(evaluation['predictions']):
                    self.assertAlmostEqual(row['probabilities'][1], max(p['score'] for p in predictions if p['image_index'] == i))
                # Labels affect metrics, never inference scores.
                for record in data.records:
                    record['target'] = 1-record['target']
                _, changed = evaluate_patches(model, DataLoader(data, batch_size=2))
                self.assertEqual([p['score'] for p in predictions], [p['score'] for p in changed])
            boxes = patch_boxes(np.ones((101, 99), np.uint8), 64, 32)
            self.assertTrue(any(b['right'] == 99 and b['bottom'] == 101 for b in boxes))
