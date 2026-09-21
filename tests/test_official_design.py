import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from thesis.official_design import (
    AMENDMENT_SHA256,
    CLAIMS_LOCK_SHA256,
    CONDITIONS,
    DATASETS,
    EXPECTED_OFFICIAL_DESIGN_SHA256,
    FINAL_HYPERPARAMETERS_SHA256,
    MODELS,
    OFFICIAL_SEEDS,
    PROTOCOL_SHA256,
    RANDOM_NAMESPACE,
    OfficialDesignError,
    build_official_design,
    canonical_json_bytes,
    derivation,
    require_file_sha256,
    sha256_file,
    validate_design_invariants,
)


ROOT = Path(__file__).resolve().parents[1]


class OfficialDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.claims = json.loads(
            (ROOT / "configs/official_claims_lock.json").read_text(encoding="utf-8")
        )
        cls.final = json.loads(
            (ROOT / "configs/final_hyperparameters.json").read_text(encoding="utf-8")
        )

    def build(self):
        return build_official_design(self.final, self.claims)

    def test_frozen_input_hashes_are_exact(self):
        self.assertEqual(sha256_file(ROOT / "docs/PROTOCOL.md"), PROTOCOL_SHA256)
        self.assertEqual(
            sha256_file(ROOT / "docs/PROTOCOL_AMENDMENT_001.md"),
            AMENDMENT_SHA256,
        )
        self.assertEqual(
            sha256_file(ROOT / "configs/official_claims_lock.json"),
            CLAIMS_LOCK_SHA256,
        )
        self.assertEqual(
            sha256_file(ROOT / "configs/final_hyperparameters.json"),
            FINAL_HYPERPARAMETERS_SHA256,
        )

    def test_official_seed_derivations_match_protocol(self):
        observed = [
            derivation("official", str(replicate))["seed"]
            for replicate in range(1, 6)
        ]
        self.assertEqual(observed, list(OFFICIAL_SEEDS))

    def test_matrix_has_exact_shape_and_order(self):
        design = self.build()
        self.assertEqual(len(design["runs"]), 90)
        self.assertEqual(len(design["paired_blocks"]), 30)
        self.assertEqual(len(design["shuffle_artifacts"]), 15)
        self.assertEqual(
            [row["array_index"] for row in design["runs"]], list(range(90))
        )
        expected = []
        for dataset in DATASETS:
            for model in MODELS:
                for replicate in range(1, 6):
                    for condition in CONDITIONS:
                        expected.append((dataset, model, replicate, condition))
        observed = [
            (
                row["dataset"],
                row["model"],
                row["official_replicate"],
                row["condition"],
            )
            for row in design["runs"]
        ]
        self.assertEqual(observed, expected)

    def test_paired_streams_are_shared_and_condition_scoped(self):
        design = self.build()
        by_block = {}
        for row in design["runs"]:
            by_block.setdefault(row["paired_block_id"], []).append(row)
        for rows in by_block.values():
            self.assertEqual({row["condition"] for row in rows}, set(CONDITIONS))
            for namespace in (
                "relation_initialisation",
                "batch_order",
                "negative_sampling",
                "data_loader_workers",
            ):
                self.assertEqual(
                    len({row["rng_streams"][namespace]["seed"] for row in rows}),
                    1,
                )
            for row in rows:
                self.assertEqual(
                    RANDOM_NAMESPACE in row["rng_streams"],
                    row["condition"] == "random",
                )
                self.assertEqual(
                    "shuffle" in row["rng_streams"],
                    row["condition"] == "shuffled_text",
                )

    def test_shuffle_is_shared_across_models_only_by_dataset_replicate(self):
        design = self.build()
        grouped = {}
        for row in design["runs"]:
            if row["condition"] == "shuffled_text":
                key = (row["dataset"], row["official_replicate"])
                grouped.setdefault(key, []).append(row)
        self.assertEqual(len(grouped), 15)
        for rows in grouped.values():
            self.assertEqual({row["model"] for row in rows}, set(MODELS))
            self.assertEqual(
                len({row["rng_streams"]["shuffle"]["seed"] for row in rows}),
                1,
            )
            self.assertEqual(len({row["shuffle_id"] for row in rows}), 1)

    def test_selected_configuration_is_identical_across_conditions(self):
        design = self.build()
        by_block = {}
        for row in design["runs"]:
            by_block.setdefault(row["paired_block_id"], []).append(row)
        for rows in by_block.values():
            serialised = {
                canonical_json_bytes(row["selected_configuration"]) for row in rows
            }
            self.assertEqual(len(serialised), 1)

    def test_initialisation_sources_are_condition_specific_and_frozen(self):
        design = self.build()
        for row in design["runs"]:
            value = row["entity_initialisation"]
            if row["condition"] == "random":
                self.assertEqual(value["kind"], "random_unit_norm_float32")
                self.assertEqual(
                    value["seed_derivation"],
                    row["rng_streams"][RANDOM_NAMESPACE],
                )
            elif row["condition"] == "correct_text":
                self.assertEqual(
                    value["kind"], "frozen_l2_normalised_correct_text"
                )
                self.assertRegex(value["normalised_embeddings_sha256"], r"^[0-9a-f]{64}$")
            else:
                self.assertEqual(
                    value["kind"], "frozen_l2_normalised_shuffled_text"
                )
                self.assertEqual(value["shuffle_id"], row["shuffle_id"])

    def test_design_is_deterministic_and_internally_hashed(self):
        first = self.build()
        second = self.build()
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        validate_design_invariants(first)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(first)).hexdigest(),
            EXPECTED_OFFICIAL_DESIGN_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(first["runs"])).hexdigest(),
            first["runs_sha256"],
        )

    def test_claim_lock_rejects_outcome_access_or_arbitrary_sesoi(self):
        changed = copy.deepcopy(self.claims)
        changed["information_boundary"]["test_data_accessed"] = True
        with self.assertRaises(OfficialDesignError):
            build_official_design(self.final, changed)

        changed = copy.deepcopy(self.claims)
        changed["smallest_effects_of_interest"][
            "calibrated_test_eaurc"
        ]["status"] = "numeric"
        with self.assertRaises(OfficialDesignError):
            build_official_design(self.final, changed)

    def test_final_hyperparameters_cannot_be_changed_after_freeze(self):
        changed = copy.deepcopy(self.final)
        changed["units"][0]["training_epochs"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_hyperparameters.json"
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaises(OfficialDesignError):
                require_file_sha256(
                    path,
                    FINAL_HYPERPARAMETERS_SHA256,
                    "Final hyperparameters",
                )

    def test_design_contains_no_results(self):
        design = self.build()
        self.assertIs(design["official_result"], False)
        self.assertIs(design["test_data_accessed"], False)
        self.assertIs(design["text_condition_results_accessed"], False)
        for row in design["runs"]:
            self.assertNotIn("test_mrr", row)
            self.assertNotIn("validation_mrr", row)
            self.assertNotIn("nll", row)
            self.assertNotIn("eaurc", row)


if __name__ == "__main__":
    unittest.main()
