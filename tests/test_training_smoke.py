import unittest

from thesis.training_smoke import make_valid_shuffle, seed32


class TrainingSmokeTest(unittest.TestCase):
    def test_seed_namespaces_are_stable_and_distinct(self):
        self.assertEqual(seed32("a"), seed32("a"))
        self.assertNotEqual(seed32("a"), seed32("b"))

    def test_shuffle_is_deranged_and_avoids_equal_hashes(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the cluster environment")
        evidence = ["a", "b", "b", "c", "d"]
        vectors = ["A", "B", "B", "C", "D"]
        permutation, attempt, digest = make_valid_shuffle(
            evidence, vectors, seed=17, maximum_attempts=1000
        )
        self.assertGreaterEqual(attempt, 1)
        self.assertEqual(64, len(digest))
        self.assertTrue(np.all(permutation != np.arange(len(permutation))))
        for target, source in enumerate(permutation):
            self.assertNotEqual(evidence[target], evidence[source])
            self.assertNotEqual(vectors[target], vectors[source])


if __name__ == "__main__":
    unittest.main()
