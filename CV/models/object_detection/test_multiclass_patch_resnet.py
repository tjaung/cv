import unittest
import numpy as np
from unittest.mock import patch

from CV.models.data_shuffle import ImageSample
from CV.models.object_detection.multiclass_patch_resnet import MulticlassPatchDataset, aggregate_patches


class MulticlassPatchTests(unittest.TestCase):
    def test_clean_regions_remain_good_and_defects_keep_their_type(self):
        names = ('scratches', 'good', 'total_rust', 'major_rust')
        samples = [ImageSample(str(i), name, 'test', str(i)) for i, name in enumerate(names)]
        def initialize(data, samples, *args):
            data.samples = samples
            data.records = [{'image_index': i, 'target': positive} for i in range(4) for positive in (0, 1)]
        with patch('CV.models.object_detection.multiclass_patch_resnet.PatchDataset.__init__', initialize):
            dataset = MulticlassPatchDataset(samples, '.', None, class_names=names)
        for row in dataset.records[::2]:
            self.assertEqual(row['target'], names.index('good'))
        self.assertEqual([r['target'] for r in dataset.records[1::2]], [0, 1, 2, 3])

    def test_aggregation_keeps_defect_decision_and_vector_consistent(self):
        names = ('good', 'major_rust', 'scratches', 'total_rust')
        clean = {'anomalous': False, 'score': .5, 'probabilities': [.5, .4, .05, .05]}
        scratch = {'anomalous': True, 'score': .35, 'probabilities': [.3, .2, .35, .15]}
        rust = {'anomalous': True, 'score': .8, 'probabilities': [.1, .05, .05, .8]}
        np.testing.assert_allclose(aggregate_patches([clean, scratch], names), [.4, .3, .2, .1])
        self.assertEqual(aggregate_patches([clean, scratch, rust], names), rust['probabilities'])
        self.assertEqual(aggregate_patches([clean], names), clean['probabilities'])

    def test_good_override_requires_count_and_highest_mean_probability(self):
        names = ('good', 'major_rust', 'scratches', 'total_rust')
        clean = {'anomalous': False, 'score': .9, 'probabilities': [.9, .04, .03, .03]}
        bad = {'anomalous': True, 'score': .8, 'probabilities': [.1, .8, .05, .05]}
        # Good has the highest mean, but only two bad patches are tolerated.
        self.assertEqual(np.argmax(aggregate_patches([clean]*10 + [bad]*2, names)), 0)
        self.assertEqual(np.argmax(aggregate_patches([clean]*10 + [bad]*3, names)), 1)
        # Having at most two bad patches alone is insufficient.
        self.assertEqual(np.argmax(aggregate_patches([bad], names)), 1)
        tie_good = {'anomalous': False, 'score': .8, 'probabilities': [.8, .1, .05, .05]}
        self.assertEqual(np.argmax(aggregate_patches([tie_good, bad], names)), 1)
        with self.assertRaises(ValueError):
            aggregate_patches([], names)
