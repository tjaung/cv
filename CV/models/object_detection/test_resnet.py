import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from CV.models.data_shuffle import ImageSample
from CV.models.object_detection.resnet import PlateDataset, PlateResNet, evaluate
from CV.preprocessing import segment_plate, preprocess_plate


class ResNetTests(unittest.TestCase):
    def test_defect_weight_mapping_and_epoch_loss(self):
        from CV.models.object_detection.weighted_resnet import defect_class_weights, train_defect_weighted
        from torch.utils.data import DataLoader, TensorDataset
        names = ['scratches', 'total_rust', 'good', 'major_rust']
        weights = defect_class_weights(names)
        torch.testing.assert_close(weights, torch.tensor([2., 2., 1., 2.]))
        model = torch.nn.Linear(2, 4)
        model.class_names = names
        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        labels = torch.tensor([2, 2, 0])
        data = DataLoader(TensorDataset(x, labels, torch.arange(3)), batch_size=2)
        expected = torch.nn.functional.cross_entropy(model(x), labels, weight=weights).item()
        # Keep parameters fixed to verify epoch weighting across uneven batches.
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        with patch('CV.models.object_detection.weighted_resnet.torch.optim.AdamW', return_value=optimizer):
            history = train_defect_weighted(model, data, epochs=1)
        self.assertAlmostEqual(history[0]['loss'], expected, places=6)
        with self.assertRaises(ValueError):
            defect_class_weights(['a', 'b', 'c', 'd'])

    def test_segment_stage_matches_full_pipeline(self):
        image = np.full((140, 140, 3), 220, np.uint8)
        image[20:120, 20:120] = (90, 40, 20)
        segmented = segment_plate(image)
        full = preprocess_plate(image)
        np.testing.assert_array_equal(segmented.plate, full.plate)
        np.testing.assert_array_equal(segmented.plate_mask, full.plate_mask)

    def test_dataset_color_and_empty_mask(self):
        sample = ImageSample('plate.png', 'good', 'train', 'example')
        plate = np.full((8, 8, 3), (20, 40, 90), np.uint8)
        result = SimpleNamespace(plate=plate, plate_mask=np.ones((8, 8), np.uint8))
        data = PlateDataset([sample], ['good', 'partial', 'scratch', 'total'], np.asarray)
        with patch('CV.models.object_detection.resnet.cv2.imread', return_value=plate), \
             patch('CV.models.object_detection.resnet.segment_plate', return_value=result):
            rgb, label, path = data[0]
            self.assertEqual((label, path), (0, 'plate.png'))
            np.testing.assert_array_equal(rgb[0, 0], [90, 40, 20])
            result.plate_mask[:] = 0
            with self.assertRaisesRegex(ValueError, 'No plate found'):
                data[0]

    def test_four_class_features_and_cam_without_download(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        model = PlateResNet(['good', 'partial', 'scratch', 'total'], weights=None)
        model.eval()
        images = torch.randn(2, 3, 224, 224)
        inspection = model.inspect(images)
        self.assertEqual(tuple(inspection['features'].shape), (2, 512))
        self.assertEqual(tuple(inspection['feature_maps'].shape), (2, 512, 7, 7))
        self.assertEqual(tuple(inspection['activation_maps'].shape), (2, 224, 224))
        torch.testing.assert_close(model(images), inspection['logits'])
        torch.testing.assert_close(inspection['probabilities'].sum(1), torch.ones(2))
        self.assertFalse(model.training)
        report = evaluate(model, [(images, torch.tensor([0, 1]), ['a', 'b'])])
        self.assertEqual(len(report['predictions']), 2)
        self.assertEqual(np.asarray(report['confusion_matrix']).shape, (4, 4))


if __name__ == '__main__':
    unittest.main()
