import math
import unittest

from thesis.official_evaluation_core import (
    OfficialEvaluationError,
    build_filter_maps,
    fit_validation_temperature,
    mean_multiclass_nll,
    query_probability_records,
)


class OfficialEvaluationCoreTest(unittest.TestCase):
    def test_filter_maps_use_only_supplied_splits(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the cluster environment")
        train = np.asarray([[0, 0, 1]], dtype=np.int64)
        valid = np.asarray([[0, 0, 2]], dtype=np.int64)
        test = np.asarray([[0, 0, 3]], dtype=np.int64)
        tail, head = build_filter_maps(train, valid)
        self.assertEqual(tail[(0, 0)], {1, 2})
        self.assertNotIn(3, tail[(0, 0)])
        test_tail, test_head = build_filter_maps(train, valid, test)
        self.assertEqual(test_tail[(0, 0)], {1, 2, 3})
        self.assertEqual(head[(0, 1)], {0})
        self.assertEqual(test_head[(0, 3)], {0})

    def test_float64_nll_matches_hand_calculation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is tested in the cluster environment")
        scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        targets = torch.tensor([2.0, 2.0], dtype=torch.float32)
        observed = mean_multiclass_nll(scores, targets, 1.0)
        expected = math.log(math.exp(2.0) + 1.0) - 2.0
        self.assertAlmostEqual(observed, expected, places=14)

    def test_bounded_temperature_is_deterministic_and_retains_boundary(self):
        try:
            import scipy  # noqa: F401
            import torch
        except ImportError:
            self.skipTest("PyTorch and SciPy are tested in the cluster environment")
        scores = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        targets = torch.tensor([3.0, 3.0], dtype=torch.float32)
        first = fit_validation_temperature(scores, targets)
        second = fit_validation_temperature(scores, targets)
        self.assertEqual(first, second)
        self.assertEqual(first["boundary_status"], "lower")
        self.assertAlmostEqual(first["fitted_temperature"], 0.001, places=15)
        self.assertLessEqual(
            first["validation_mean_multiclass_nll"],
            first["raw_temperature_one_validation_mean_multiclass_nll"],
        )

    def test_invalid_temperature_fails(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is tested in the cluster environment")
        scores = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        targets = torch.tensor([1.0], dtype=torch.float32)
        with self.assertRaises(OfficialEvaluationError):
            mean_multiclass_nll(scores, targets, 0.0)

    def test_query_records_use_realistic_ties_and_smallest_top_index(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is tested in the cluster environment")
        scores = torch.tensor(
            [[2.0, 2.0, 0.0], [3.0, -torch.inf, 1.0]], dtype=torch.float32
        )
        target_scores = torch.tensor([2.0, 3.0], dtype=torch.float32)
        target_indices = torch.tensor([1, 0], dtype=torch.long)
        counts = torch.tensor([3, 2], dtype=torch.long)
        observed = query_probability_records(
            scores, target_scores, target_indices, counts, 2.0
        )
        self.assertEqual(observed["predicted_index"], [0, 0])
        self.assertEqual(observed["top_score_tie_count"], [2, 1])
        self.assertEqual(observed["correct"], [False, True])
        self.assertEqual(observed["rank"], [1.5, 1.0])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in observed["calibrated_confidence"]))


if __name__ == "__main__":
    unittest.main()
