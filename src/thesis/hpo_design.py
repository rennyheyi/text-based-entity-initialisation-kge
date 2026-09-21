"""Freeze the six validation-only random-baseline HPO candidate designs."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .source_audit import sha256_file
from .source_lock import canonical_json_bytes, read_json_object


class HPODesignError(RuntimeError):
    """Raised when an HPO design cannot be frozen mechanically."""


PREFIX = "Integrating External Evidence to Reduce Uncertainty in Link Prediction"
PROTOCOL_SHA256 = "98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc"
PREFLIGHT_CONFIGURATION_SHA256 = (
    "1768965fa38972b00504b18f11a7cf1dde0678dade59606c40eb4d1687c53053"
)
DATASETS = ("FB15k-237", "WN18RR", "CoDEx-M")
MODELS = ("TransE", "DistMult")
TUNING_SEEDS = (347869531, 2594257857, 3877759316)
HPO_DESIGN_SEEDS = {
    ("FB15k-237", "TransE"): 3687661691,
    ("FB15k-237", "DistMult"): 1578249420,
    ("WN18RR", "TransE"): 61651425,
    ("WN18RR", "DistMult"): 3751609374,
    ("CoDEx-M", "TransE"): 1196443311,
    ("CoDEx-M", "DistMult"): 2640021856,
}
NAMESPACES = (
    "random_entity_initialisation",
    "relation_initialisation",
    "batch_order",
    "negative_sampling",
    "data_loader_workers",
)


def derive_seed(value: str) -> Dict[str, Any]:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return {
        "input": value,
        "sha256": digest,
        "seed": int.from_bytes(bytes.fromhex(digest)[:4], "big", signed=False),
    }


def validate_search_space(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise HPODesignError("HPO search-space schema version is invalid")
    if (
        config.get("status") != "hpo_search_space_frozen_pending_design"
        or config.get("official_result") is not False
    ):
        raise HPODesignError("HPO search-space status is invalid")
    if tuple(config.get("datasets", ())) != DATASETS:
        raise HPODesignError("HPO dataset order/scope is invalid")
    if tuple(config.get("models", ())) != MODELS:
        raise HPODesignError("HPO model order/scope is invalid")
    if config.get("candidate_count_per_unit") != 12:
        raise HPODesignError("Each HPO unit must contain exactly twelve candidates")
    if config.get("protocol_sha256") != PROTOCOL_SHA256:
        raise HPODesignError("HPO design references the wrong protocol")
    if config.get("required_hardware_preflight") != {
        "configuration_sha256": PREFLIGHT_CONFIGURATION_SHA256,
        "status": "hardware_feasibility_preflight_passed_pending_hpo_design",
        "validation_or_test_metrics_computed": False,
    }:
        raise HPODesignError("Hardware-preflight prerequisite differs")
    if config.get("fixed_training") != {
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
    }:
        raise HPODesignError("Fixed HPO training settings differ")
    if config.get("stage_plan") != {
        "stage_1": {
            "candidates": 12,
            "tuning_replicates": [1],
            "epochs": 25,
            "validation_epochs": [25],
            "advance": 4,
        },
        "stage_2": {
            "candidates": 4,
            "tuning_replicates": [1, 2],
            "epochs": 75,
            "validation_epochs": [75],
            "advance": 2,
        },
        "stage_3": {
            "candidates": 2,
            "tuning_replicates": [1, 2, 3],
            "epochs": 200,
            "validation_every_epochs": 10,
            "advance": 1,
        },
    }:
        raise HPODesignError("HPO stage plan differs from the protocol")
    search = config.get("search_space")
    if not isinstance(search, dict):
        raise HPODesignError("HPO search space is missing")
    expected = {
        "batch_size": [256, 512, 1024],
        "negatives_per_positive": [1, 16, 64],
        "regularisation_coefficient": [0.0, 0.0001, 0.001, 0.01, 0.1],
        "transe_distance_norm": [1, 2],
        "transe_exact_count_per_norm": 6,
    }
    for key, value in expected.items():
        if search.get(key) != value:
            raise HPODesignError("Frozen HPO search field differs: " + key)
    learning_rate = search.get("learning_rate")
    if learning_rate != {
        "distribution": "log_uniform",
        "minimum": 0.0001,
        "maximum": 0.003,
    }:
        raise HPODesignError("Frozen learning-rate search space differs")
    seed_config = config.get("seed_derivation")
    if not isinstance(seed_config, dict) or seed_config.get("prefix") != PREFIX:
        raise HPODesignError("HPO seed prefix differs")
    if (
        seed_config.get("digest") != "SHA-256"
        or seed_config.get("integer_rule")
        != "first_four_digest_bytes_unsigned_big_endian"
        or seed_config.get("candidate_rng") != "NumPy Generator(PCG64)"
    ):
        raise HPODesignError("HPO seed/RNG algorithm differs")
    observed_tuning = tuple(
        int(item["seed"]) for item in seed_config.get("tuning_replicates", ())
    )
    if observed_tuning != TUNING_SEEDS:
        raise HPODesignError("Tuning seeds differ from the protocol")
    observed_design = {
        (str(item["dataset"]), str(item["model"])): int(item["seed"])
        for item in seed_config.get("hpo_design_seeds", ())
    }
    if observed_design != HPO_DESIGN_SEEDS:
        raise HPODesignError("HPO-design seeds differ from the protocol")
    if tuple(seed_config.get("training_namespaces", ())) != NAMESPACES:
        raise HPODesignError("Training RNG namespaces differ from the protocol")
    for replicate, expected_seed in enumerate(TUNING_SEEDS, 1):
        derived = derive_seed(PREFIX + "|tuning|" + str(replicate))
        if derived["seed"] != expected_seed:
            raise HPODesignError("A tuning seed cannot be reproduced")
    for (dataset, model), expected_seed in HPO_DESIGN_SEEDS.items():
        derived = derive_seed(PREFIX + "|hpo-design|" + dataset + "|" + model)
        if derived["seed"] != expected_seed:
            raise HPODesignError("An HPO-design seed cannot be reproduced")


def validate_preflight(
    report: Mapping[str, Any], search_config: Mapping[str, Any]
) -> None:
    required = search_config["required_hardware_preflight"]
    for key, value in required.items():
        if report.get(key) != value:
            raise HPODesignError("Hardware preflight prerequisite differs: " + key)
    shape = report.get("frozen_maximum_shape")
    if shape != {
        "batch_size": 1024,
        "embedding_dimension": 384,
        "negatives_per_positive": 64,
        "training_steps": 3,
    }:
        raise HPODesignError("Hardware preflight did not exercise the maximum shape")
    observed_units = {
        (str(dataset["dataset"]), str(model["model"]))
        for dataset in report.get("datasets", ())
        for model in dataset.get("models", ())
    }
    expected_units = {(dataset, model) for dataset in DATASETS for model in MODELS}
    if observed_units != expected_units:
        raise HPODesignError("Hardware preflight is missing a tuning unit")


def candidate_parameters(
    rng: Any, model: str, norm: Optional[int], search: Mapping[str, Any]
) -> Dict[str, Any]:
    import numpy as np

    learning_rate = search["learning_rate"]
    value = float(
        np.exp(
            rng.uniform(
                math.log(float(learning_rate["minimum"])),
                math.log(float(learning_rate["maximum"])),
            )
        )
    )

    def choice(name: str) -> Any:
        values = search[name]
        return values[int(rng.integers(0, len(values)))]

    parameters = {
        "learning_rate": value,
        "batch_size": int(choice("batch_size")),
        "negatives_per_positive": int(choice("negatives_per_positive")),
        "regularisation_coefficient": float(
            choice("regularisation_coefficient")
        ),
    }
    if model == "TransE":
        parameters["transe_distance_norm"] = int(norm)
    return parameters


def generate_candidates(
    dataset: str,
    model: str,
    design_seed: int,
    search: Mapping[str, Any],
    count: int,
) -> List[Dict[str, Any]]:
    import numpy as np

    if (dataset, model) not in HPO_DESIGN_SEEDS or count != 12:
        raise HPODesignError("Unknown HPO unit or candidate count")
    rng = np.random.Generator(np.random.PCG64(design_seed))
    norms: Sequence[Optional[int]]
    if model == "TransE":
        norms = [1] * 6 + [2] * 6
    else:
        norms = [None] * count
    seen = set()
    candidates = []
    for index, norm in enumerate(norms, 1):
        for _ in range(10000):
            parameters = candidate_parameters(rng, model, norm, search)
            fingerprint = hashlib.sha256(canonical_json_bytes(parameters)).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                break
        else:
            raise HPODesignError("Duplicate rejection exceeded its attempt limit")
        slug = dataset.lower().replace("-", "").replace("_", "")
        candidate_id = "{}-{}-c{:02d}-{}".format(
            slug, model.lower(), index, fingerprint[:12]
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "configuration_sha256": fingerprint,
                "parameters": parameters,
            }
        )
    if len({item["candidate_id"] for item in candidates}) != count:
        raise HPODesignError("HPO candidate identifiers are not unique")
    if model == "TransE":
        counts = {
            norm: sum(
                item["parameters"]["transe_distance_norm"] == norm
                for item in candidates
            )
            for norm in (1, 2)
        }
        if counts != {1: 6, 2: 6}:
            raise HPODesignError("TransE distance norms are not balanced")
    return candidates


def stream_derivations(dataset: str, model: str) -> List[Dict[str, Any]]:
    records = []
    for replicate in range(1, 4):
        for namespace in NAMESPACES:
            value = "|".join(
                (
                    PREFIX,
                    "stream",
                    "tuning",
                    str(replicate),
                    dataset,
                    model,
                    namespace,
                )
            )
            record = derive_seed(value)
            record.update(
                {
                    "replicate": replicate,
                    "namespace": namespace,
                }
            )
            records.append(record)
    if len({item["seed"] for item in records}) != len(records):
        raise HPODesignError("A tuning-unit RNG substream seed collided")
    return records


def freeze_design(
    search_path: Path,
    preflight_path: Path,
    protocol_path: Path,
    *,
    project_root: Path,
) -> Dict[str, Any]:
    import numpy as np

    search_config = read_json_object(search_path)
    preflight = read_json_object(preflight_path)
    validate_search_space(search_config)
    validate_preflight(preflight, search_config)
    if sha256_file(protocol_path) != search_config["protocol_sha256"]:
        raise HPODesignError("Protocol hash differs from the frozen HPO search space")
    git = git_state(project_root)
    units = []
    for dataset in DATASETS:
        for model in MODELS:
            design_seed = HPO_DESIGN_SEEDS[(dataset, model)]
            candidates = generate_candidates(
                dataset,
                model,
                design_seed,
                search_config["search_space"],
                int(search_config["candidate_count_per_unit"]),
            )
            units.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "unit_id": dataset.lower().replace("-", "")
                    + "-"
                    + model.lower(),
                    "hpo_design_seed": design_seed,
                    "hpo_design_seed_derivation": derive_seed(
                        PREFIX + "|hpo-design|" + dataset + "|" + model
                    ),
                    "tuning_stream_derivations": stream_derivations(
                        dataset, model
                    ),
                    "candidates": candidates,
                }
            )
    if len(units) != 6 or sum(len(unit["candidates"]) for unit in units) != 72:
        raise HPODesignError("Frozen HPO design does not contain 6 x 12 candidates")
    return {
        "schema_version": 1,
        "status": "hpo_design_frozen_pending_stage_1",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "selection_metric": "combined_filtered_validation_mrr",
        "protocol_sha256": search_config["protocol_sha256"],
        "search_space_sha256": sha256_file(search_path),
        "hardware_preflight_report_sha256": sha256_file(preflight_path),
        "hardware_preflight_job_id": preflight["execution"][
            "scheduler_job_id"
        ],
        "source_git": git,
        "generator_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "bit_generator": "PCG64",
        },
        "fixed_training": search_config["fixed_training"],
        "stage_plan": search_config["stage_plan"],
        "tuning_replicate_seeds": [
            {
                "replicate": replicate,
                **derive_seed(PREFIX + "|tuning|" + str(replicate)),
            }
            for replicate in range(1, 4)
        ],
        "units": units,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = freeze_design(
        args.search_space.resolve(),
        args.preflight_report.resolve(),
        args.protocol.resolve(),
        project_root=args.project_root.resolve(),
    )
    write_bytes_new_or_identical(
        args.output.resolve(), canonical_json_bytes(report), "local-hpo-design"
    )
    print("Frozen HPO design written to {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
