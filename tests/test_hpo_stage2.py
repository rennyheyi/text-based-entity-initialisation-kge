import hashlib
import unittest

from thesis.hpo_stage2 import (
    HPOStage2Error,
    select_array_task,
    stream_seed,
    validate_stage2_design,
)
from thesis.source_lock import canonical_json_bytes


def designs():
    units = []
    source_units = []
    for unit_index in range(6):
        source_candidates = []
        advancing = []
        derivations = []
        for replicate in range(1, 4):
            for namespace in (
                "random_entity_initialisation",
                "relation_initialisation",
                "batch_order",
                "negative_sampling",
                "data_loader_workers",
            ):
                value = "unit{}|{}|{}".format(unit_index, replicate, namespace)
                digest = hashlib.sha256(value.encode()).hexdigest()
                derivations.append(
                    {
                        "input": value,
                        "sha256": digest,
                        "seed": int.from_bytes(bytes.fromhex(digest)[:4], "big"),
                        "replicate": replicate,
                        "namespace": namespace,
                    }
                )
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
            source = {
                "candidate_id": "u{}-c{}".format(unit_index, candidate_index),
                "configuration_sha256": fingerprint,
                "parameters": parameters,
            }
            source_candidates.append(source)
            if candidate_index < 4:
                advancing.append(
                    {
                        "candidate_id": source["candidate_id"],
                        "candidate_configuration_sha256": fingerprint,
                        "parameters": parameters,
                        "stage_1_rank": candidate_index + 1,
                        "stage_1_validation_mrr": 0.5 - candidate_index * 0.01,
                        "stage_1_report_sha256": "a" * 64,
                    }
                )
        unit = {
            "dataset": ["FB15k-237", "WN18RR", "CoDEx-M"][unit_index // 2],
            "model": ["TransE", "DistMult"][unit_index % 2],
            "unit_id": "u{}".format(unit_index),
            "tuning_stream_derivations": derivations,
        }
        units.append({**unit, "advancing_candidates": advancing})
        source_units.append({**unit, "candidates": source_candidates})
    fixed = {
        "entity_embedding_dimension": 384,
        "relation_embedding_dimension": 384,
        "optimiser": "Adam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "objective": "mean_softplus_negative_minus_positive",
        "negative_sampler": "uniform_single_side_train_seen_reject_train_truth",
        "initialisation_condition": "random",
    }
    source = {
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
        "fixed_training": fixed,
        "units": source_units,
    }
    stage2 = {
        "schema_version": 1,
        "status": "hpo_stage_2_design_frozen_pending_runs",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": 2,
        "epochs": 75,
        "tuning_replicates": [1, 2],
        "validation_epochs": [75],
        "advance": 2,
        "selection_metric": "mean_combined_filtered_validation_mrr",
        "tie_breaking": [
            "smaller_negatives_per_positive",
            "smaller_batch_size",
            "lexicographically_smaller_candidate_id",
        ],
        "protocol_sha256": "98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc",
        "source_hpo_design_sha256": "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93",
        "source_stage_1_attempt_lock_sha256": "dee93cf7da08cff01b1d32dc295bd0bb64770380921b5e2e880f33f564bd5f20",
        "source_stage_1_report_manifest_sha256": "b" * 64,
        "source_stage_1_runner_git_commit": "c" * 40,
        "fixed_training": fixed,
        "units": units,
    }
    return stage2, source


class HPOStage2Test(unittest.TestCase):
    def test_array_mapping_covers_candidates_replicates_and_units(self):
        stage2, source = designs()
        unit, candidate, replicate = select_array_task(stage2, source, 0)
        self.assertEqual(("u0", "u0-c0", 1), (unit["unit_id"], candidate["candidate_id"], replicate))
        unit, candidate, replicate = select_array_task(stage2, source, 1)
        self.assertEqual(("u0-c0", 2), (candidate["candidate_id"], replicate))
        unit, candidate, replicate = select_array_task(stage2, source, 7)
        self.assertEqual(("u0-c3", 2), (candidate["candidate_id"], replicate))
        unit, candidate, replicate = select_array_task(stage2, source, 8)
        self.assertEqual(("u1", "u1-c0", 1), (unit["unit_id"], candidate["candidate_id"], replicate))
        unit, candidate, replicate = select_array_task(stage2, source, 47)
        self.assertEqual(("u5", "u5-c3", 2), (unit["unit_id"], candidate["candidate_id"], replicate))
        with self.assertRaises(HPOStage2Error):
            select_array_task(stage2, source, 48)

    def test_replicate_seed_derivation_is_exact(self):
        stage2, source = designs()
        validate_stage2_design(stage2, source)
        unit = stage2["units"][0]
        self.assertNotEqual(
            stream_seed(unit, 1, "batch_order"),
            stream_seed(unit, 2, "batch_order"),
        )


if __name__ == "__main__":
    unittest.main()
