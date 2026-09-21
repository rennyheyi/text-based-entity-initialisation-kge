import unittest

from thesis.hpo_stage2_selection import (
    HPOStage2SelectionError,
    candidate_mean,
    select_advancing,
    validate_attempt_lock,
)


def candidate(candidate_id, mean_mrr, negatives, batch):
    return {
        "candidate_id": candidate_id,
        "mean_validation_mrr": mean_mrr,
        "parameters": {
            "negatives_per_positive": negatives,
            "batch_size": batch,
        },
    }


def attempt_lock():
    return {
        "schema_version": 1,
        "status": "hpo_stage_2_attempts_frozen_pending_selection",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "protocol_sha256": "98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc",
        "source_hpo_design_sha256": "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93",
        "stage_2_design_sha256": "f28776481ecff19142f8174245c4ab11b95aeaeffd74f11188c827e41848ffd9",
        "canonical_attempt_ranges": [
            {
                "array_job_id": "440800",
                "first_array_index": 0,
                "last_array_index": 0,
            },
            {
                "array_job_id": "440801",
                "first_array_index": 1,
                "last_array_index": 47,
            },
        ],
        "excluded_completed_attempt_ranges": [
            {
                "array_job_id": "440859",
                "first_array_index": 1,
                "last_array_index": 7,
                "reason": "accidental duplicate",
            }
        ],
        "cancelled_duplicate_submission": {
            "array_job_id": "440859",
            "requested_first_array_index": 1,
            "requested_last_array_index": 47,
            "maximum_concurrent_tasks": 2,
            "completed_report_indices": list(range(1, 8)),
            "cancelled_running_indices_without_reports": [8, 9],
            "cancelled_pending_ranges_without_reports": [
                {"first_array_index": 10, "last_array_index": 47}
            ],
            "reason": "cancelled after discovery",
        },
        "selection_rule": "canonical only",
    }


class HPOStage2SelectionTest(unittest.TestCase):
    def test_attempt_lock_separates_canonical_completed_and_cancelled(self):
        canonical, excluded = validate_attempt_lock(attempt_lock())
        self.assertEqual("hpo_stage2_440800_0.json", canonical[0])
        self.assertEqual("hpo_stage2_440801_47.json", canonical[47])
        self.assertEqual(7, len(excluded))
        self.assertIn("hpo_stage2_440859_7.json", excluded)

    def test_candidate_mean_requires_exact_replicates(self):
        records = [
            {"candidate_id": "c1", "replicate": 1, "validation_mrr": 0.2},
            {"candidate_id": "c1", "replicate": 2, "validation_mrr": 0.4},
        ]
        self.assertAlmostEqual(0.3, candidate_mean(records))
        records[1]["replicate"] = 1
        with self.assertRaises(HPOStage2SelectionError):
            candidate_mean(records)

    def test_primary_order_is_full_precision_mean_mrr(self):
        records = [
            candidate("a", 0.2, 1, 256),
            candidate("b", 0.20000000000000004, 64, 1024),
            candidate("c", 0.1, 1, 256),
        ]
        selected = select_advancing(records)
        self.assertEqual(["b", "a"], [item["candidate_id"] for item in selected])

    def test_exact_tie_breaks_by_k_batch_then_identifier(self):
        records = [
            candidate("z", 0.5, 1, 256),
            candidate("b", 0.5, 1, 256),
            candidate("a", 0.5, 1, 512),
            candidate("c", 0.5, 16, 256),
        ]
        selected = select_advancing(records)
        self.assertEqual(["b", "z"], [item["candidate_id"] for item in selected])

    def test_fewer_than_two_valid_candidates_terminates_unit(self):
        with self.assertRaises(HPOStage2SelectionError):
            select_advancing([candidate("a", 0.1, 1, 256)])


if __name__ == "__main__":
    unittest.main()
