import unittest

from thesis.hpo_design import (
    HPO_DESIGN_SEEDS,
    PREFIX,
    TUNING_SEEDS,
    derive_seed,
    generate_candidates,
    stream_derivations,
)


SEARCH = {
    "learning_rate": {
        "distribution": "log_uniform",
        "minimum": 0.0001,
        "maximum": 0.003,
    },
    "batch_size": [256, 512, 1024],
    "negatives_per_positive": [1, 16, 64],
    "regularisation_coefficient": [0.0, 0.0001, 0.001, 0.01, 0.1],
    "transe_distance_norm": [1, 2],
    "transe_exact_count_per_norm": 6,
}


class HPODesignTest(unittest.TestCase):
    def test_protocol_seeds_are_reproducible(self):
        for replicate, expected in enumerate(TUNING_SEEDS, 1):
            self.assertEqual(
                expected,
                derive_seed(PREFIX + "|tuning|" + str(replicate))["seed"],
            )
        for (dataset, model), expected in HPO_DESIGN_SEEDS.items():
            self.assertEqual(
                expected,
                derive_seed(
                    PREFIX + "|hpo-design|" + dataset + "|" + model
                )["seed"],
            )

    def test_candidate_design_is_deterministic_distinct_and_bounded(self):
        first = generate_candidates(
            "FB15k-237", "TransE", HPO_DESIGN_SEEDS[("FB15k-237", "TransE")], SEARCH, 12
        )
        second = generate_candidates(
            "FB15k-237", "TransE", HPO_DESIGN_SEEDS[("FB15k-237", "TransE")], SEARCH, 12
        )
        self.assertEqual(first, second)
        self.assertEqual(12, len({item["candidate_id"] for item in first}))
        self.assertEqual(
            {1: 6, 2: 6},
            {
                norm: sum(
                    item["parameters"]["transe_distance_norm"] == norm
                    for item in first
                )
                for norm in (1, 2)
            },
        )
        for candidate in first:
            parameters = candidate["parameters"]
            self.assertGreaterEqual(parameters["learning_rate"], 0.0001)
            self.assertLessEqual(parameters["learning_rate"], 0.003)
            self.assertIn(parameters["batch_size"], SEARCH["batch_size"])
            self.assertIn(
                parameters["negatives_per_positive"],
                SEARCH["negatives_per_positive"],
            )

    def test_substream_namespaces_are_distinct(self):
        records = stream_derivations("WN18RR", "DistMult")
        self.assertEqual(15, len(records))
        self.assertEqual(15, len({record["seed"] for record in records}))


if __name__ == "__main__":
    unittest.main()
