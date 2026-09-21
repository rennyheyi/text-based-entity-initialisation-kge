import copy
import json
import math
import unittest
from pathlib import Path

from thesis.official_calibration_audit import (
    OfficialCalibrationAuditError,
    validate_calibration_report,
)
from thesis.official_design import build_official_design


ROOT = Path(__file__).resolve().parents[1]


class OfficialCalibrationAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        claims = json.loads(
            (ROOT / "configs/official_claims_lock.json").read_text(encoding="utf-8")
        )
        final = json.loads(
            (ROOT / "configs/final_hyperparameters.json").read_text(encoding="utf-8")
        )
        cls.design_run = build_official_design(final, claims)["runs"][0]
        cls.locked = {
            "checkpoint": {
                "path": cls.design_run["expected_checkpoint"],
                "sha256": "a" * 64,
            }
        }

    def report(self):
        run = self.design_run
        return {
            "schema_version": 1,
            "status": "official_validation_calibration_completed_pending_temperature_freeze",
            "official_result": False,
            "validation_data_accessed": True,
            "test_data_accessed": False,
            "test_metrics_computed": False,
            "array_index": run["array_index"],
            "run_id": run["run_id"],
            "paired_block_id": run["paired_block_id"],
            "dataset": run["dataset"],
            "model": run["model"],
            "condition": run["condition"],
            "official_replicate": run["official_replicate"],
            "official_seed": run["official_seed"],
            "run_configuration_sha256": run["run_configuration_sha256"],
            "checkpoint": self.locked["checkpoint"],
            "official_design_sha256": "48a82e2bf835942f1d76ec40ce0b9b52499aa3a0c16f40ab31c04652215a6ee5",
            "official_checkpoint_lock_sha256": "220e4dd3d74a7a3699a32c64fd763f99b6af97dbcf9d1fedbd2ddebbe27f7a86",
            "official_evaluation_lock_sha256": "ffc14e3084a63d336e6da4c8b6dbe950e8f41cb6558b7e244809be7d9cccf4fa",
            "validation": {
                "triples": 2,
                "head_queries": 2,
                "tail_queries": 2,
                "combined_queries": 4,
                "candidate_vocabulary": 10,
                "minimum_filtered_candidate_count": 8,
                "maximum_filtered_candidate_count": 10,
                "filtering_splits": ["train", "validation"],
                "test_split_loaded": False,
            },
            "temperature": {
                "method": "scipy.optimize.minimize_scalar_bounded",
                "parameterisation": "natural_log_temperature",
                "temperature_bounds": [0.001, 1000.0],
                "xatol_log_temperature": 1e-6,
                "maximum_iterations": 100,
                "fitted_temperature": 2.0,
                "fitted_log_temperature": math.log(2.0),
                "validation_mean_multiclass_nll": 0.8,
                "raw_temperature_one_validation_mean_multiclass_nll": 1.0,
                "success": True,
                "scipy_success": True,
                "boundary_status": "interior",
                "boundary_objectives": {"lower": 2.0, "upper": 1.5},
            },
            "git": {"commit": "b" * 40, "tracked_worktree_clean": True},
        }

    def test_valid_report_passes(self):
        validate_calibration_report(self.report(), self.design_run, self.locked)

    def test_test_access_or_boundary_drift_fails(self):
        report = self.report()
        report["test_data_accessed"] = True
        with self.assertRaises(OfficialCalibrationAuditError):
            validate_calibration_report(report, self.design_run, self.locked)
        report = self.report()
        report["temperature"]["fitted_temperature"] = 1001.0
        with self.assertRaises(OfficialCalibrationAuditError):
            validate_calibration_report(report, self.design_run, self.locked)

    def test_worse_than_boundary_fails(self):
        report = self.report()
        report["temperature"]["validation_mean_multiclass_nll"] = 3.0
        with self.assertRaises(OfficialCalibrationAuditError):
            validate_calibration_report(report, self.design_run, self.locked)


if __name__ == "__main__":
    unittest.main()
