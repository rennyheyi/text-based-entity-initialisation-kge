import copy
import json
import unittest
from pathlib import Path

from thesis.official_calibration import (
    OfficialCalibrationError,
    select_run_and_checkpoint,
    validate_evaluation_lock,
)
from thesis.official_design import build_official_design


ROOT = Path(__file__).resolve().parents[1]


class OfficialCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        claims = json.loads(
            (ROOT / "configs/official_claims_lock.json").read_text(encoding="utf-8")
        )
        final = json.loads(
            (ROOT / "configs/final_hyperparameters.json").read_text(encoding="utf-8")
        )
        cls.design = build_official_design(final, claims)
        cls.evaluation_lock = json.loads(
            (ROOT / "configs/official_evaluation_lock.json").read_text(encoding="utf-8")
        )

    def checkpoint_lock(self):
        rows = []
        for run in self.design["runs"]:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "array_index": run["array_index"],
                    "dataset": run["dataset"],
                    "model": run["model"],
                    "condition": run["condition"],
                    "official_replicate": run["official_replicate"],
                    "official_seed": run["official_seed"],
                    "run_configuration_sha256": run["run_configuration_sha256"],
                    "selected_configuration": run["selected_configuration"],
                }
            )
        return {
            "status": "official_checkpoints_frozen_pending_calibration_and_test_evaluation",
            "official_result": False,
            "test_data_accessed": False,
            "validation_metrics_computed": False,
            "test_metrics_computed": False,
            "counts": {"official_runs": 90},
            "runs": rows,
        }

    def test_evaluation_lock_is_exact(self):
        validate_evaluation_lock(self.evaluation_lock)
        changed = copy.deepcopy(self.evaluation_lock)
        changed["temperature_scaling"]["test_queries_used_for_fitting"] = True
        with self.assertRaises(OfficialCalibrationError):
            validate_evaluation_lock(changed)

    def test_array_mapping_covers_all_checkpoint_identities(self):
        lock = self.checkpoint_lock()
        selected = [
            select_run_and_checkpoint(self.design, lock, index)[0]["run_id"]
            for index in range(90)
        ]
        self.assertEqual(len(selected), 90)
        self.assertEqual(len(set(selected)), 90)
        with self.assertRaises(OfficialCalibrationError):
            select_run_and_checkpoint(self.design, lock, 90)

    def test_checkpoint_metadata_drift_fails(self):
        lock = self.checkpoint_lock()
        lock["runs"][0]["official_seed"] += 1
        with self.assertRaises(OfficialCalibrationError):
            select_run_and_checkpoint(self.design, lock, 0)


if __name__ == "__main__":
    unittest.main()
