import copy
import json
import unittest
from pathlib import Path

from thesis.official_design import build_official_design
from thesis.official_training_audit import (
    OfficialTrainingAuditError,
    paired_block_record,
    validate_report_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class OfficialTrainingAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        claims = json.loads(
            (ROOT / "configs/official_claims_lock.json").read_text(encoding="utf-8")
        )
        final = json.loads(
            (ROOT / "configs/final_hyperparameters.json").read_text(encoding="utf-8")
        )
        cls.design = build_official_design(final, claims)

    def report_for(self, run):
        epochs = run["selected_configuration"]["training_epochs"]
        return {
            "schema_version": 1,
            "status": "official_training_completed_pending_calibration_and_test_evaluation",
            "official_result": False,
            "test_data_accessed": False,
            "validation_metrics_computed": False,
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
            "selected_configuration": run["selected_configuration"],
            "rng_streams": run["rng_streams"],
            "official_design_sha256": "48a82e2bf835942f1d76ec40ce0b9b52499aa3a0c16f40ab31c04652215a6ee5",
            "hpo_design_sha256": "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93",
            "hardware_preflight_report_sha256": "a" * 64,
            "numerical_policy": {
                "parameter_dtype": "float32",
                "automatic_mixed_precision": False,
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms_enabled": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
            "training": {
                "epochs": epochs,
                "batches": epochs,
                "training_sequence_sha256": "b" * 64,
                "epoch_history": [
                    {"epoch": epoch, "batches": 1} for epoch in range(1, epochs + 1)
                ],
                "globally_training_unseen_maximum_absolute_change": 0.0,
                "batch_loss_columns": [
                    "total_loss",
                    "ranking_loss",
                    "unweighted_batch_local_l2",
                ],
                "maximum_negative_sampling_attempts": 1000,
                "rng_algorithm": "NumPy Generator(PCG64)",
                "data_loader_workers": 0,
            },
            "initialisation": {
                "dimension": 384,
                "entity_matrix_raw_sha256": "c" * 64,
                "relation_seed": run["rng_streams"]["relation_initialisation"]["seed"],
                "relation_matrix_raw_sha256": "d" * 64,
            },
            "data": {
                "dataset_manifest_sha256": "f" * 64,
                "entities": 10,
                "relations": 2,
                "training_triples": 20,
                "training_seen_entities": 10,
                "globally_training_unseen_entities": 0,
            },
            "training_environment": {"device_name": "NVIDIA H200 NVL"},
            "git": {"commit": "e" * 40, "tracked_worktree_clean": True},
        }

    def test_exact_report_metadata_passes(self):
        run = self.design["runs"][0]
        validate_report_metadata(
            self.report_for(run),
            run,
            preflight_sha256="a" * 64,
            dataset_manifest_sha256="f" * 64,
        )

    def test_outcome_access_or_numerical_drift_fails(self):
        run = self.design["runs"][0]
        report = self.report_for(run)
        report["test_data_accessed"] = True
        with self.assertRaises(OfficialTrainingAuditError):
            validate_report_metadata(
                report,
                run,
                preflight_sha256="a" * 64,
                dataset_manifest_sha256="f" * 64,
            )
        report = self.report_for(run)
        report["numerical_policy"]["automatic_mixed_precision"] = True
        with self.assertRaises(OfficialTrainingAuditError):
            validate_report_metadata(
                report,
                run,
                preflight_sha256="a" * 64,
                dataset_manifest_sha256="f" * 64,
            )

    def test_epoch_or_unseen_entity_drift_fails(self):
        run = self.design["runs"][0]
        report = self.report_for(run)
        report["training"]["epoch_history"].pop()
        with self.assertRaises(OfficialTrainingAuditError):
            validate_report_metadata(
                report,
                run,
                preflight_sha256="a" * 64,
                dataset_manifest_sha256="f" * 64,
            )
        report = self.report_for(run)
        report["training"]["globally_training_unseen_maximum_absolute_change"] = 1e-8
        with self.assertRaises(OfficialTrainingAuditError):
            validate_report_metadata(
                report,
                run,
                preflight_sha256="a" * 64,
                dataset_manifest_sha256="f" * 64,
            )

    def test_hidden_outcome_field_fails(self):
        run = self.design["runs"][0]
        report = self.report_for(run)
        report["artifacts"] = {"test_mrr": 0.5}
        with self.assertRaises(OfficialTrainingAuditError):
            validate_report_metadata(
                report,
                run,
                preflight_sha256="a" * 64,
                dataset_manifest_sha256="f" * 64,
            )

    def paired_rows(self):
        block_id = self.design["paired_blocks"][0]["paired_block_id"]
        rows = []
        for run in [row for row in self.design["runs"] if row["paired_block_id"] == block_id]:
            rows.append(
                {
                    "paired_block_id": block_id,
                    "dataset": run["dataset"],
                    "model": run["model"],
                    "condition": run["condition"],
                    "official_replicate": run["official_replicate"],
                    "run_id": run["run_id"],
                    "selected_configuration": run["selected_configuration"],
                    "rng_streams": run["rng_streams"],
                    "training_sequence_sha256": "1" * 64,
                    "relation_matrix_raw_sha256": "2" * 64,
                }
            )
        return rows

    def test_paired_block_requires_shared_sequence_and_relation_initialisation(self):
        rows = self.paired_rows()
        result = paired_block_record(rows)
        self.assertEqual(len(result["run_ids"]), 3)
        changed = copy.deepcopy(rows)
        changed[1]["training_sequence_sha256"] = "3" * 64
        with self.assertRaises(OfficialTrainingAuditError):
            paired_block_record(changed)
        changed = copy.deepcopy(rows)
        changed[2]["relation_matrix_raw_sha256"] = "4" * 64
        with self.assertRaises(OfficialTrainingAuditError):
            paired_block_record(changed)

    def test_paired_block_requires_all_three_conditions(self):
        rows = self.paired_rows()
        rows[2]["condition"] = "correct_text"
        with self.assertRaises(OfficialTrainingAuditError):
            paired_block_record(rows)


if __name__ == "__main__":
    unittest.main()
