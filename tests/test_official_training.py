import json
import unittest
from pathlib import Path

from thesis.official_design import build_official_design
from thesis.official_training import OfficialTrainingError, select_official_run


ROOT = Path(__file__).resolve().parents[1]


class OfficialTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        claims = json.loads(
            (ROOT / "configs/official_claims_lock.json").read_text(encoding="utf-8")
        )
        final = json.loads(
            (ROOT / "configs/final_hyperparameters.json").read_text(encoding="utf-8")
        )
        cls.design = build_official_design(final, claims)

    def test_array_mapping_covers_all_frozen_runs(self):
        selected = [select_official_run(self.design, index) for index in range(90)]
        self.assertEqual([row["array_index"] for row in selected], list(range(90)))
        self.assertEqual(len({row["run_id"] for row in selected}), 90)
        with self.assertRaises(OfficialTrainingError):
            select_official_run(self.design, -1)
        with self.assertRaises(OfficialTrainingError):
            select_official_run(self.design, 90)

    def test_paired_conditions_share_training_streams_and_configuration(self):
        by_block = {}
        for row in self.design["runs"]:
            by_block.setdefault(row["paired_block_id"], []).append(row)
        self.assertEqual(len(by_block), 30)
        for rows in by_block.values():
            self.assertEqual({row["condition"] for row in rows}, {
                "random", "correct_text", "shuffled_text"
            })
            self.assertEqual(
                len({json.dumps(row["selected_configuration"], sort_keys=True) for row in rows}),
                1,
            )
            for namespace in ("relation_initialisation", "batch_order", "negative_sampling"):
                self.assertEqual(
                    len({row["rng_streams"][namespace]["seed"] for row in rows}), 1
                )

    def test_no_official_run_contains_test_or_validation_outcomes(self):
        for row in self.design["runs"]:
            encoded = json.dumps(row, sort_keys=True)
            self.assertNotIn("test_mrr", encoded)
            self.assertNotIn("validation_mrr", encoded)
            self.assertNotIn("temperature", encoded)


if __name__ == "__main__":
    unittest.main()
