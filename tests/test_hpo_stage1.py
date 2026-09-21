import hashlib
import unittest

from thesis.hpo_stage1 import (
    HPOStage1Error,
    build_filter_maps,
    evaluate_validation,
    initialisation_statistics,
    random_unit_array,
    rank_direction,
    select_array_task,
)
from thesis.kg_core import realistic_filtered_rank, score_candidates
from thesis.source_lock import canonical_json_bytes


def minimal_design():
    units = []
    for unit_index in range(6):
        candidates = []
        for candidate_index in range(12):
            parameters = {
                "learning_rate": 0.001 + candidate_index * 1e-8,
                "batch_size": 256,
                "negatives_per_positive": 1,
                "regularisation_coefficient": 0.0,
            }
            fingerprint = hashlib.sha256(
                canonical_json_bytes(parameters)
            ).hexdigest()
            candidates.append(
                {
                    "candidate_id": "u{}-c{}".format(
                        unit_index, candidate_index
                    ),
                    "configuration_sha256": fingerprint,
                    "parameters": parameters,
                }
            )
        units.append({"unit_id": "u{}".format(unit_index), "candidates": candidates})
    return {
        "schema_version": 1,
        "status": "hpo_design_frozen_pending_stage_1",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "selection_metric": "combined_filtered_validation_mrr",
        "stage_plan": {
            "stage_1": {
                "candidates": 12,
                "tuning_replicates": [1],
                "epochs": 25,
                "validation_epochs": [25],
                "advance": 4,
            }
        },
        "units": units,
    }


class HPOStage1PureTest(unittest.TestCase):
    def test_array_index_selects_exact_candidate(self):
        design = minimal_design()
        unit, candidate = select_array_task(design, 0)
        self.assertEqual("u0", unit["unit_id"])
        self.assertEqual("u0-c0", candidate["candidate_id"])
        unit, candidate = select_array_task(design, 71)
        self.assertEqual("u5", unit["unit_id"])
        self.assertEqual("u5-c11", candidate["candidate_id"])
        with self.assertRaises(HPOStage1Error):
            select_array_task(design, 72)

    def test_random_unit_initialisation_is_deterministic(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the cluster environment")
        first = random_unit_array(8, 5, 17)
        second = random_unit_array(8, 5, 17)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(
            np.allclose(np.linalg.norm(first.astype(np.float64), axis=1), 1.0)
        )
        statistics = initialisation_statistics(first)
        self.assertLessEqual(
            statistics["maximum_absolute_deviation_from_unit_norm"], 1e-6
        )


class HPOStage1TorchTest(unittest.TestCase):
    def test_blockwise_filtered_ranking_matches_expected(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("PyTorch is tested in the cluster environment")
        entity = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        relation = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        train = np.asarray([[0, 0, 2], [3, 0, 2]], dtype=np.int64)
        valid = np.asarray([[0, 0, 1]], dtype=np.int64)
        metrics, table = evaluate_validation(
            "DistMult",
            entity,
            relation,
            train,
            valid,
            np.asarray([0, 2, 3], dtype=np.int64),
            transe_distance_norm=1,
            query_batch_size=1,
            candidate_chunk_size=2,
        )
        self.assertEqual(2, metrics["combined"]["queries"])
        self.assertEqual(1, metrics["strata"]["seen_target"]["head"]["queries"])
        self.assertEqual(0, metrics["strata"]["unseen_target"]["head"]["queries"])
        self.assertEqual(1, metrics["strata"]["unseen_target"]["tail"]["queries"])
        self.assertIn(b"validation_row\tdirection", table)
        tail_map, head_map = build_filter_maps(train, valid)
        expected = []
        for direction, filter_map, target in (
            ("head", head_map, 0),
            ("tail", tail_map, 1),
        ):
            scores = score_candidates(
                "DistMult",
                entity,
                relation,
                valid[0],
                direction=direction,
            ).numpy()
            key = (0, 0) if direction == "tail" else (0, 1)
            expected.append(
                realistic_filtered_rank(
                    scores,
                    target_index=target,
                    filtered_indices=filter_map[key],
                )
            )
        observed_head = rank_direction(
            "DistMult",
            entity,
            relation,
            valid,
            head_map,
            direction="head",
            transe_distance_norm=1,
            query_batch_size=1,
            candidate_chunk_size=2,
        )
        observed_tail = rank_direction(
            "DistMult",
            entity,
            relation,
            valid,
            tail_map,
            direction="tail",
            transe_distance_norm=1,
            query_batch_size=1,
            candidate_chunk_size=2,
        )
        self.assertEqual(expected, observed_head + observed_tail)
        self.assertAlmostEqual(
            sum(1.0 / rank for rank in expected) / 2.0,
            metrics["combined"]["mrr"],
        )


if __name__ == "__main__":
    unittest.main()
