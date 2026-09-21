import hashlib
import unittest

import numpy as np

from thesis.official_shuffle import OfficialShuffleError, verify_permutation
from thesis.training_smoke import make_valid_shuffle


class OfficialShuffleTest(unittest.TestCase):
    def test_frozen_algorithm_is_deterministic_and_strict(self):
        evidence = ["a", "b", "b", "c", "d", "e"]
        vectors = ["0", "1", "2", "3", "4", "5"]
        first, attempt_first, digest_first = make_valid_shuffle(
            evidence, vectors, seed=12345, maximum_attempts=1000
        )
        second, attempt_second, digest_second = make_valid_shuffle(
            evidence, vectors, seed=12345, maximum_attempts=1000
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(attempt_first, attempt_second)
        self.assertEqual(digest_first, digest_second)
        observed = verify_permutation(first, evidence, vectors)
        self.assertEqual(observed["self_assignments"], 0)
        self.assertEqual(observed["equal_evidence_hash_assignments"], 0)
        self.assertEqual(observed["equal_vector_hash_assignments"], 0)
        self.assertTrue(observed["correct_and_shuffled_vector_multisets_identical"])
        self.assertEqual(
            observed["permutation_raw_sha256"],
            hashlib.sha256(first.astype("<i8").tobytes()).hexdigest(),
        )

    def test_non_bijection_and_equal_hash_assignment_fail(self):
        with self.assertRaises(OfficialShuffleError):
            verify_permutation(
                np.asarray([1, 1, 0]), ["a", "b", "c"], ["0", "1", "2"]
            )
        with self.assertRaises(OfficialShuffleError):
            verify_permutation(
                np.asarray([1, 0, 3, 2]),
                ["same", "same", "c", "d"],
                ["0", "1", "2", "3"],
            )


if __name__ == "__main__":
    unittest.main()
