import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from thesis.official_evaluation_core import query_probability_records
from thesis.official_test_evaluation import (
    OfficialTestEvaluationError,
    aggregate_query_metrics,
    binary_auroc,
    ece_table,
    load_official_test_data,
    risk_coverage_table,
    strict_tsv_bytes,
)


def row(query_id, rank, correct, raw, calibrated):
    return {
        "query_id": query_id,
        "direction": "head" if query_id in {"a", "b"} else "tail",
        "realistic_target_rank": rank,
        "top_label_correct": correct,
        "raw_confidence": raw,
        "calibrated_confidence": calibrated,
        "raw_nll": -math.log(0.25 if correct else 0.1),
        "calibrated_nll": -math.log(0.5 if correct else 0.2),
        "raw_normalised_entropy": 0.8,
        "calibrated_normalised_entropy": 0.6,
        "filtered_candidate_count": 4,
        "target_seen_status": "seen",
        "triple_endpoint_seen_status": "both_seen",
    }


class OfficialTestEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row("a", 1.0, True, 0.9, 0.8),
            row("b", 2.0, False, 0.8, 0.2),
            row("c", 3.0, False, 0.2, 0.1),
            row("d", 1.0, True, 0.7, 0.9),
        ]

    def test_auroc_uses_average_ranks_for_exact_ties(self):
        self.assertEqual(binary_auroc([0.5, 0.5, 0.2, 0.8], [True, False, False, True]), 0.875)
        self.assertIsNone(binary_auroc([0.2, 0.3], [True, True]))

    def test_ece_has_fifteen_fixed_bins_and_includes_one(self):
        rows = [dict(self.rows[0], raw_confidence=1.0)]
        table, ece = ece_table(rows, "raw")
        self.assertEqual(len(table), 15)
        self.assertEqual(table[-1]["queries"], 1)
        self.assertTrue(table[-1]["final_bin_includes_one"])
        self.assertEqual(ece, 0.0)

    def test_risk_coverage_ties_break_by_query_identifier(self):
        rows = [dict(self.rows[0], raw_confidence=0.5), dict(self.rows[1], raw_confidence=0.5)]
        table, values = risk_coverage_table(rows, "raw")
        self.assertEqual(table[0]["query_id_at_boundary"], "a")
        self.assertEqual(values["aurc"], 0.25)
        self.assertEqual(values["eaurc"], 0.0)

    def test_aggregate_reproduces_primary_metrics(self):
        metrics, ece, risk = aggregate_query_metrics(self.rows, 1.5)
        self.assertEqual(metrics["queries"], 4)
        self.assertAlmostEqual(metrics["metric_vector"]["combined_filtered_test_mrr"], (1 + .5 + 1/3 + 1) / 4)
        self.assertIn("calibrated_test_eaurc", metrics["metric_vector"])
        self.assertEqual(len(ece), 30)
        self.assertEqual(len(risk), 8)

    def test_strict_tsv_rejects_embedded_delimiter(self):
        with self.assertRaises(OfficialTestEvaluationError):
            strict_tsv_bytes(("value",), [{"value": "unsafe\tvalue"}])

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

    def test_official_test_loader_requires_frozen_test_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "data" / "processed" / "Toy"
            manifests = root / "data" / "manifests"
            processed.mkdir(parents=True)
            manifests.mkdir(parents=True)
            test_bytes = b"0\t0\t1\n"
            (processed / "test.tsv").write_bytes(test_bytes)
            manifest_path = manifests / "Toy_dataset.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "processed_sha256": {
                            "test.tsv": hashlib.sha256(test_bytes).hexdigest()
                        }
                    }
                ),
                encoding="utf-8",
            )
            base = {
                "manifest_path": manifest_path,
                "train": None,
                "valid": None,
                "entity_count": 2,
                "relation_count": 1,
            }
            fake_hpo_stage1 = types.ModuleType("thesis.hpo_stage1")
            fake_hpo_stage1.load_stage_data = mock.Mock(return_value=base)
            fake_hpo_stage1.load_indexed_tsv = lambda path: np.loadtxt(
                path, delimiter="\t", dtype="int64", ndmin=2
            )
            fake_hpo_stage1.triple_keys = lambda values, *, entity_count, relation_count: (
                (values[:, 0] * relation_count + values[:, 1]) * entity_count
                + values[:, 2]
            )
            with mock.patch.dict(sys.modules, {"thesis.hpo_stage1": fake_hpo_stage1}):
                observed = load_official_test_data(root, "Toy", "unused-by-mock")
            self.assertEqual(observed["test"].tolist(), [[0, 0, 1]])
            (processed / "test.tsv").write_bytes(b"1\t0\t0\n")
            with mock.patch.dict(sys.modules, {"thesis.hpo_stage1": fake_hpo_stage1}):
                with self.assertRaises(OfficialTestEvaluationError):
                    load_official_test_data(root, "Toy", "unused-by-mock")


if __name__ == "__main__":
    unittest.main()
