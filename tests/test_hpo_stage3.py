import copy
import unittest

from thesis.hpo_stage3 import (
    HPOStage3Error,
    select_array_task,
    validate_stage3_design,
)
from tests.test_hpo_stage2 import designs as stage2_designs


def designs():
    stage2, source = stage2_designs()
    units = []
    for stage2_unit in stage2["units"]:
        candidates = []
        for rank, candidate in enumerate(
            stage2_unit["advancing_candidates"][:2], 1
        ):
            candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_configuration_sha256": candidate[
                        "candidate_configuration_sha256"
                    ],
                    "parameters": candidate["parameters"],
                    "stage_1_rank": candidate["stage_1_rank"],
                    "stage_2_rank": rank,
                    "stage_2_mean_validation_mrr": 0.5,
                    "stage_2_replicate_validation_mrr": {
                        "1": 0.4,
                        "2": 0.6,
                    },
                    "stage_2_report_sha256_by_replicate": {
                        "1": "a" * 64,
                        "2": "b" * 64,
                    },
                }
            )
        units.append(
            {
                "dataset": stage2_unit["dataset"],
                "model": stage2_unit["model"],
                "unit_id": stage2_unit["unit_id"],
                "tuning_stream_derivations": stage2_unit[
                    "tuning_stream_derivations"
                ],
                "advancing_candidates": candidates,
            }
        )
    stage3 = {
        "schema_version": 1,
        "status": "hpo_stage_3_design_frozen_pending_runs",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": 3,
        "maximum_epochs": 200,
        "tuning_replicates": [1, 2, 3],
        "validation_epochs": list(range(10, 201, 10)),
        "candidates_per_unit": 2,
        "trajectories_per_unit": 6,
        "selection_metric": "mean_combined_filtered_validation_mrr_by_candidate_epoch",
        "selection_target": "candidate_epoch_pair",
        "tie_breaking": [
            "earlier_epoch",
            "smaller_negatives_per_positive",
            "smaller_batch_size",
            "lexicographically_smaller_candidate_id",
        ],
        "protocol_sha256": "98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc",
        "source_hpo_design_sha256": "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93",
        "source_stage_2_design_sha256": "f28776481ecff19142f8174245c4ab11b95aeaeffd74f11188c827e41848ffd9",
        "source_stage_2_attempt_lock_sha256": "cce80551e4997c4bc573eb5f82d551fecba8f3450094170458962558dd09f157",
        "source_stage_2_report_manifest_sha256": "c" * 64,
        "source_stage_2_runner_git_commit": "d" * 40,
        "fixed_training": source["fixed_training"],
        "units": units,
    }
    return stage3, stage2, source


class HPOStage3Test(unittest.TestCase):
    def test_array_mapping_covers_candidates_replicates_and_units(self):
        stage3, stage2, source = designs()
        unit, candidate, replicate = select_array_task(
            stage3, stage2, source, 0
        )
        self.assertEqual(
            ("u0", "u0-c0", 1),
            (unit["unit_id"], candidate["candidate_id"], replicate),
        )
        _, candidate, replicate = select_array_task(stage3, stage2, source, 2)
        self.assertEqual(("u0-c0", 3), (candidate["candidate_id"], replicate))
        _, candidate, replicate = select_array_task(stage3, stage2, source, 3)
        self.assertEqual(("u0-c1", 1), (candidate["candidate_id"], replicate))
        unit, candidate, replicate = select_array_task(
            stage3, stage2, source, 35
        )
        self.assertEqual(
            ("u5", "u5-c1", 3),
            (unit["unit_id"], candidate["candidate_id"], replicate),
        )

    def test_validation_schedule_is_exact(self):
        stage3, stage2, source = designs()
        validate_stage3_design(stage3, stage2, source)
        broken = copy.deepcopy(stage3)
        broken["validation_epochs"].remove(100)
        with self.assertRaises(HPOStage3Error):
            validate_stage3_design(broken, stage2, source)

    def test_stage2_replicate_mean_is_recomputed(self):
        stage3, stage2, source = designs()
        stage3["units"][0]["advancing_candidates"][0][
            "stage_2_mean_validation_mrr"
        ] = 0.5000000000000001
        with self.assertRaises(HPOStage3Error):
            validate_stage3_design(stage3, stage2, source)


if __name__ == "__main__":
    unittest.main()
