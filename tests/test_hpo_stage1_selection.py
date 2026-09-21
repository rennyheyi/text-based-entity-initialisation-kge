import tempfile
import unittest
from pathlib import Path

from thesis.hpo_stage1_selection import (
    HPOStage1SelectionError,
    audit_rank_table,
    select_advancing,
    validate_attempt_lock,
)
from thesis.kg_core import ranking_metrics


def record(candidate_id, mrr, negatives, batch):
    return {
        "candidate_id": candidate_id,
        "validation_mrr": mrr,
        "parameters": {
            "negatives_per_positive": negatives,
            "batch_size": batch,
        },
    }


class HPOStage1SelectionTest(unittest.TestCase):
    def test_attempt_lock_separates_canonical_and_excluded_attempts(self):
        attempt_lock = {
            "schema_version": 1,
            "status": "hpo_stage_1_attempts_frozen_pending_selection",
            "official_result": False,
            "test_data_accessed": False,
            "text_conditions_accessed": False,
            "protocol_sha256": "98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc",
            "hpo_design_sha256": "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93",
            "canonical_attempt_ranges": [
                {
                    "array_job_id": "10",
                    "first_array_index": 0,
                    "last_array_index": 0,
                },
                {
                    "array_job_id": "11",
                    "first_array_index": 1,
                    "last_array_index": 71,
                },
            ],
            "excluded_attempts": [
                {
                    "array_job_id": "12",
                    "array_index": 0,
                    "reason": "accidental duplicate",
                }
            ],
            "selection_rule": "canonical only",
        }
        canonical, excluded = validate_attempt_lock(attempt_lock)
        self.assertEqual("hpo_stage1_10_0.json", canonical[0])
        self.assertEqual("hpo_stage1_11_71.json", canonical[71])
        self.assertIn("hpo_stage1_12_0.json", excluded)

    def test_primary_order_is_full_precision_mrr(self):
        records = [
            record("a", 0.2, 64, 1024),
            record("b", 0.3, 64, 1024),
            record("c", 0.1, 1, 256),
            record("d", 0.4, 64, 1024),
            record("e", 0.5, 64, 1024),
        ]
        selected = select_advancing(records)
        self.assertEqual(["e", "d", "b", "a"], [x["candidate_id"] for x in selected])

    def test_exact_tie_breaks_by_k_batch_then_identifier(self):
        records = [
            record("z", 0.5, 1, 256),
            record("b", 0.5, 1, 256),
            record("a", 0.5, 1, 512),
            record("c", 0.5, 16, 256),
            record("d", 0.4, 1, 256),
        ]
        selected = select_advancing(records)
        self.assertEqual(["b", "z", "a", "c"], [x["candidate_id"] for x in selected])

    def test_fewer_than_four_valid_runs_terminates_unit(self):
        with self.assertRaises(HPOStage1SelectionError):
            select_advancing(
                [
                    record("a", 0.1, 1, 256),
                    record("b", 0.2, 1, 256),
                    record("c", 0.3, 1, 256),
                ]
            )

    def test_rank_table_recomputes_stored_metrics(self):
        rows = (
            "validation_row\tdirection\thead\trelation\ttail\ttarget_entity\trank\n"
            "0\thead\t2\t0\t3\t2\t1.0\n"
            "0\ttail\t2\t0\t3\t3\t2.5\n"
            "1\thead\t4\t1\t5\t4\t3.0\n"
            "1\ttail\t4\t1\t5\t5\t4.0\n"
        )
        validation = {
            "head": ranking_metrics([1.0, 3.0]),
            "tail": ranking_metrics([2.5, 4.0]),
            "combined": ranking_metrics([1.0, 3.0, 2.5, 4.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranks.tsv"
            path.write_text(rows, encoding="utf-8")
            audit_rank_table(
                path, validation_triples=2, validation=validation
            )
            validation["combined"]["mrr"] += 1e-12
            with self.assertRaises(HPOStage1SelectionError):
                audit_rank_table(
                    path, validation_triples=2, validation=validation
                )


if __name__ == "__main__":
    unittest.main()
