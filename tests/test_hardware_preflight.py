import unittest

from thesis.hardware_preflight import (
    HardwarePreflightError,
    build_fixed_step_schedule,
    validate_config,
)


class HardwarePreflightTest(unittest.TestCase):
    def test_maximum_shape_is_mandatory(self):
        config = {
            "schema_version": 1,
            "status": "development_hardware_feasibility_preflight_configuration",
            "official_result": False,
            "datasets": ["FB15k-237", "WN18RR", "CoDEx-M"],
            "models": [
                {"name": "TransE", "transe_distance_norm": 2},
                {"name": "DistMult", "transe_distance_norm": 1},
            ],
            "training_steps": 3,
            "embedding_dimension": 384,
            "batch_size": 1024,
            "negatives_per_positive": 64,
            "weight_decay": 0.0,
            "development_seed_label": "test",
        }
        validate_config(config)
        config["negatives_per_positive"] = 16
        with self.assertRaises(HardwarePreflightError):
            validate_config(config)

    def test_fixed_schedule_has_exact_shape_and_rejects_truth(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the cluster environment")
        triples = np.asarray(
            [[head, 0, (head + 1) % 20] for head in range(20)] * 4,
            dtype=np.int64,
        )
        schedule, digest, population, truth = build_fixed_step_schedule(
            triples,
            steps=2,
            batch_size=8,
            negatives_per_positive=4,
            batch_seed=3,
            negative_seed=5,
            maximum_attempts=1000,
        )
        self.assertEqual(2, len(schedule))
        self.assertEqual(64, len(digest))
        self.assertEqual(20, len(population))
        for positives, negatives in schedule:
            self.assertEqual((8, 3), positives.shape)
            self.assertEqual((8, 4, 3), negatives.shape)
            self.assertFalse(
                any(tuple(map(int, row)) in truth for row in negatives.reshape(-1, 3))
            )


if __name__ == "__main__":
    unittest.main()
