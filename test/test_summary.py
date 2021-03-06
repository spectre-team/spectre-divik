import unittest

from functools import partial

import numpy as np
import numpy.testing as npt

import spdivik.rejection as rj
import spdivik.summary as sm
import spdivik.types as ty


DUMMY_RESULT = ty.DivikResult(
    centroids=np.zeros((3, 1)),
    quality=3.,
    partition=np.array([0] * 10 + [1] * 5 + [2] * 10, dtype=int),
    filters={},
    thresholds={},
    merged=np.array([0] * 10 + [1] * 15, dtype=int),
    subregions=[
        None,
        ty.DivikResult(
            centroids=np.zeros((2, 1)),
            quality=2.,
            partition=np.array([0] + [1] * 4, dtype=int),
            filters={},
            thresholds={},
            merged=np.array([0] + [1] * 4, dtype=int),
            subregions=[None, None]
        ),
        ty.DivikResult(
            centroids=np.zeros((3, 1)),
            quality=2.,
            partition=np.array([0] * 2 + [1] * 3 + [2] * 5, dtype=int),
            filters={},
            thresholds={},
            merged=np.array([0] * 2 + [1] * 3 + [2] * 5, dtype=int),
            subregions=[None, None, None]
        )
    ]
)


class DepthTest(unittest.TestCase):
    def test_resolves_tree_depth(self):
        self.assertEqual(sm.depth(DUMMY_RESULT), 3)


class MergeTest(unittest.TestCase):
    def test_merges_disjoint_regions(self):
        partition = sm.merged_partition(DUMMY_RESULT)
        regions, counts = np.unique(partition, return_counts=True)
        npt.assert_equal([10, 1, 4, 2, 3, 5], counts)
        npt.assert_equal(np.arange(6), regions)


class RejectionTest(unittest.TestCase):
    def test_without_rejection_updates_merged_and_nothing_else(self):
        filtered = sm.reject_split(DUMMY_RESULT, [])
        self.assertEqual(filtered.quality, DUMMY_RESULT.quality)
        self.assertEqual(sm.depth(filtered), sm.depth(DUMMY_RESULT))
        npt.assert_equal(filtered.merged, sm.merged_partition(DUMMY_RESULT))

    def test_rejects_splits(self):
        rejection_conditions = [
            partial(rj.reject_if_clusters_smaller_than, size=2)]
        filtered = sm.reject_split(DUMMY_RESULT, rejection_conditions)
        self.assertIsNone(filtered.subregions[1])
