import unittest

from thesis.kg_core import (
    KGCoreError,
    realistic_filtered_rank,
    sample_negative_triples,
    sample_negative_triples_vectorized,
    triple_keys,
)


class KGCorePureTest(unittest.TestCase):
    def test_realistic_rank_exact_tie_and_filtering(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the cluster environment")
        scores = np.asarray([2.0, 1.0, 1.0, 3.0], dtype=np.float32)
        self.assertEqual(
            3.5,
            realistic_filtered_rank(scores, target_index=1, filtered_indices=[]),
        )
        self.assertEqual(
            2.5,
            realistic_filtered_rank(scores, target_index=1, filtered_indices=[3]),
        )
        scores[0] = 1.0 + 1e-6
        self.assertEqual(
            2.5,
            realistic_filtered_rank(scores, target_index=1, filtered_indices=[3]),
        )

    def test_negative_sampler_is_deterministic_and_rejects_train(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the cluster environment")
        positives = np.asarray([[0, 0, 1], [1, 0, 2]], dtype=np.int64)
        true_train = {(0, 0, 1), (1, 0, 2), (2, 0, 1)}
        kwargs = dict(
            positives=positives,
            negatives_per_positive=8,
            replacement_entities=np.asarray([0, 1, 2, 3]),
            true_training_triples=true_train,
            maximum_attempts=1000,
        )
        first = sample_negative_triples(
            rng=np.random.default_rng(7), **kwargs
        )
        second = sample_negative_triples(
            rng=np.random.default_rng(7), **kwargs
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(any(tuple(row) in true_train for row in first.reshape(-1, 3)))

    def test_vectorized_sampler_is_deterministic_and_rejects_train(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the cluster environment")
        positives = np.asarray([[0, 0, 1], [1, 0, 2]], dtype=np.int64)
        true_train = np.asarray([[0, 0, 1], [1, 0, 2], [2, 0, 1]])
        keys = np.sort(
            triple_keys(true_train, entity_count=4, relation_count=1)
        )
        kwargs = dict(
            positives=positives,
            negatives_per_positive=16,
            replacement_entities=np.arange(4, dtype=np.int64),
            sorted_true_training_keys=keys,
            entity_count=4,
            relation_count=1,
            maximum_attempts=1000,
        )
        first = sample_negative_triples_vectorized(
            rng=np.random.default_rng(11), **kwargs
        )
        second = sample_negative_triples_vectorized(
            rng=np.random.default_rng(11), **kwargs
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(
            any(
                tuple(map(int, row)) in {tuple(map(int, value)) for value in true_train}
                for row in first.reshape(-1, 3)
            )
        )
        repeated = np.repeat(positives, 16, axis=0)
        flattened = first.reshape(-1, 3)
        self.assertTrue(
            np.all(np.sum(flattened != repeated, axis=1) == 1)
        )


class KGCoreTorchTest(unittest.TestCase):
    def test_hand_scores_and_dense_chunked_candidates_match(self):
        try:
            import torch
            from thesis.kg_core import score_candidates, score_triples
        except ImportError:
            self.skipTest("PyTorch is tested in the cluster environment")
        entities = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        relations = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        triples = torch.tensor([[0, 0, 2]], dtype=torch.long)
        self.assertAlmostEqual(
            -1.0,
            float(score_triples("TransE", entities, relations, triples)[0]),
        )
        self.assertAlmostEqual(
            0.5,
            float(score_triples("DistMult", entities, relations, triples)[0]),
        )
        for model in ("TransE", "DistMult"):
            for direction in ("head", "tail"):
                dense = score_candidates(
                    model,
                    entities,
                    relations,
                    (0, 0, 2),
                    direction=direction,
                )
                chunked = score_candidates(
                    model,
                    entities,
                    relations,
                    (0, 0, 2),
                    direction=direction,
                    chunk_size=2,
                )
                self.assertTrue(torch.equal(dense, chunked))

    def test_invalid_model_fails(self):
        try:
            import torch
            from thesis.kg_core import score_triples
        except ImportError:
            self.skipTest("PyTorch is tested in the cluster environment")
        with self.assertRaises(KGCoreError):
            score_triples(
                "Unknown",
                torch.ones((2, 2)),
                torch.ones((1, 2)),
                torch.tensor([[0, 0, 1]]),
            )


if __name__ == "__main__":
    unittest.main()
