"""Exercise the largest frozen training shape without evaluating validation/test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .kg_core import sample_negative_triples, train_embeddings
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object
from .training_smoke import load_entity_ids, load_indexed_tsv, random_unit_matrix, seed32


class HardwarePreflightError(RuntimeError):
    """Raised when the frozen maximum-shape preflight is invalid."""


EXPECTED_DATASETS = ("FB15k-237", "WN18RR", "CoDEx-M")
EXPECTED_MODELS = ("TransE", "DistMult")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise HardwarePreflightError("Hardware-preflight schema version is invalid")
    if (
        config.get("status")
        != "development_hardware_feasibility_preflight_configuration"
        or config.get("official_result") is not False
    ):
        raise HardwarePreflightError("Hardware-preflight status is invalid")
    if tuple(config.get("datasets", ())) != EXPECTED_DATASETS:
        raise HardwarePreflightError("Hardware preflight must cover all dataset shapes")
    models = config.get("models")
    if not isinstance(models, list) or tuple(item.get("name") for item in models) != EXPECTED_MODELS:
        raise HardwarePreflightError("Hardware preflight must cover TransE and DistMult")
    required_exact = {
        "embedding_dimension": 384,
        "batch_size": 1024,
        "negatives_per_positive": 64,
        "training_steps": 3,
    }
    for key, expected in required_exact.items():
        if config.get(key) != expected:
            raise HardwarePreflightError(
                "Hardware-preflight {} must equal {}".format(key, expected)
            )
    if config.get("weight_decay") != 0.0:
        raise HardwarePreflightError("Optimiser weight decay must remain zero")
    if not isinstance(config.get("development_seed_label"), str):
        raise HardwarePreflightError("Development seed label is missing")


def verify_dataset_inputs(
    project_root: Path, dataset: str
) -> Tuple[Any, int, int, Mapping[str, Any]]:
    processed = project_root / "data" / "processed" / dataset
    manifest_path = project_root / "data" / "manifests" / (dataset + "_dataset.json")
    manifest = read_json_object(manifest_path)
    if manifest.get("dataset") != dataset:
        raise HardwarePreflightError("Dataset manifest identity mismatch")
    expected_hashes = manifest.get("processed_sha256")
    required = ("train.tsv", "entities.tsv", "relations.tsv")
    if not isinstance(expected_hashes, dict):
        raise HardwarePreflightError("Processed dataset hashes are absent")
    for filename in required:
        path = processed / filename
        if expected_hashes.get(filename) != sha256_file(path):
            raise HardwarePreflightError(
                "Processed {} hash mismatch for {}".format(filename, dataset)
            )
    train = load_indexed_tsv(processed / "train.tsv")
    entity_count = len(load_entity_ids(processed / "entities.tsv"))
    relation_count = len(load_entity_ids(processed / "relations.tsv"))
    if (
        int(train[:, [0, 2]].max()) >= entity_count
        or int(train[:, 1].max()) >= relation_count
    ):
        raise HardwarePreflightError("Training indices exceed the frozen vocabulary")
    return train, entity_count, relation_count, manifest


def build_fixed_step_schedule(
    triples: Any,
    *,
    steps: int,
    batch_size: int,
    negatives_per_positive: int,
    batch_seed: int,
    negative_seed: int,
    maximum_attempts: int,
) -> Tuple[List[Tuple[Any, Any]], str, Any, Set[Tuple[int, int, int]]]:
    import numpy as np

    triples = np.asarray(triples, dtype=np.int64)
    required = steps * batch_size
    if triples.ndim != 2 or triples.shape[1] != 3 or len(triples) < required:
        raise HardwarePreflightError("Training split is too small for fixed preflight steps")
    true_training = {tuple(map(int, row)) for row in triples}
    replacement_entities = np.unique(triples[:, [0, 2]])
    batch_rng = np.random.default_rng(batch_seed)
    negative_rng = np.random.default_rng(negative_seed)
    order = batch_rng.permutation(len(triples))[:required]
    selected = triples[order]
    digest = hashlib.sha256()
    digest.update(order.astype("<i8", copy=False).tobytes())
    schedule = []
    for step in range(steps):
        start = step * batch_size
        positives = selected[start : start + batch_size].copy()
        negatives = sample_negative_triples(
            positives,
            negatives_per_positive=negatives_per_positive,
            replacement_entities=replacement_entities,
            true_training_triples=true_training,
            rng=negative_rng,
            maximum_attempts=maximum_attempts,
        )
        digest.update(step.to_bytes(8, "big"))
        digest.update(positives.astype("<i8", copy=False).tobytes())
        digest.update(negatives.astype("<i8", copy=False).tobytes())
        schedule.append((positives, negatives))
    return schedule, digest.hexdigest(), replacement_entities, true_training


def run_preflight(
    config_path: Path, *, project_root: Path, scratch: Path
) -> Dict[str, Any]:
    import numpy as np
    import torch

    execution = require_scheduled_execution(scratch.resolve())
    config = read_json_object(config_path)
    validate_config(config)
    git = git_state(project_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise HardwarePreflightError("Exactly one scheduler-visible CUDA device is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda"
    dimension = int(config["embedding_dimension"])
    dataset_reports = []

    for dataset in config["datasets"]:
        train, entity_count, relation_count, dataset_manifest = verify_dataset_inputs(
            project_root, dataset
        )
        label = str(config["development_seed_label"])
        batch_seed = seed32(label + "|" + dataset + "|batch")
        negative_seed = seed32(label + "|" + dataset + "|negative")
        schedule, sequence_sha, replacement_entities, true_training = (
            build_fixed_step_schedule(
                train,
                steps=int(config["training_steps"]),
                batch_size=int(config["batch_size"]),
                negatives_per_positive=int(config["negatives_per_positive"]),
                batch_seed=batch_seed,
                negative_seed=negative_seed,
                maximum_attempts=int(config["maximum_negative_sampling_attempts"]),
            )
        )
        train_seen = set(map(int, replacement_entities))
        globally_unseen = sorted(set(range(entity_count)) - train_seen)
        model_reports = []

        for model_spec in config["models"]:
            model = str(model_spec["name"])
            entity_seed = seed32(label + "|" + dataset + "|" + model + "|entity")
            relation_seed = seed32(label + "|" + dataset + "|" + model + "|relation")
            torch.cuda.empty_cache()
            gc.collect()
            entity_initial = random_unit_matrix(
                entity_count, dimension, entity_seed, device
            )
            relation_initial = random_unit_matrix(
                relation_count, dimension, relation_seed, device
            )
            unseen_initial = (
                entity_initial[globally_unseen].clone() if globally_unseen else None
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            free_before, total_memory = torch.cuda.mem_get_info()
            started = time.perf_counter()
            final_entity, final_relation, losses = train_embeddings(
                model,
                entity_initial,
                relation_initial,
                schedule,
                learning_rate=float(config["learning_rate"]),
                regularisation_coefficient=float(
                    config["regularisation_coefficient"]
                ),
                transe_distance_norm=int(model_spec["transe_distance_norm"]),
                adam_beta1=float(config["adam_beta1"]),
                adam_beta2=float(config["adam_beta2"]),
                adam_epsilon=float(config["adam_epsilon"]),
                weight_decay=float(config["weight_decay"]),
            )
            torch.cuda.synchronize()
            runtime = time.perf_counter() - started
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            free_after, observed_total = torch.cuda.mem_get_info()
            if observed_total != total_memory or len(losses) != int(config["training_steps"]):
                raise HardwarePreflightError("GPU memory/step accounting is invalid")
            if not np.isfinite(losses).all():
                raise HardwarePreflightError("Hardware preflight produced non-finite losses")
            unseen_change = 0.0
            if globally_unseen:
                unseen_change = float(
                    torch.max(
                        torch.abs(final_entity[globally_unseen] - unseen_initial)
                    ).cpu()
                )
                if unseen_change != 0.0:
                    raise HardwarePreflightError(
                        "Globally training-unseen entity changed during preflight"
                    )
            model_reports.append(
                {
                    "model": model,
                    "transe_distance_norm": int(
                        model_spec["transe_distance_norm"]
                    ),
                    "entity_initialisation_seed": entity_seed,
                    "relation_initialisation_seed": relation_seed,
                    "training_sequence_sha256": sequence_sha,
                    "steps": len(losses),
                    "losses": [float(value) for value in losses],
                    "runtime_seconds": float(runtime),
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_reserved_fraction_of_device": float(
                        peak_reserved / total_memory
                    ),
                    "free_memory_before_training_bytes": int(free_before),
                    "free_memory_after_training_bytes": int(free_after),
                    "globally_training_unseen_entities": len(globally_unseen),
                    "globally_training_unseen_maximum_absolute_change": unseen_change,
                }
            )
            del (
                final_entity,
                final_relation,
                entity_initial,
                relation_initial,
                unseen_initial,
            )
            torch.cuda.empty_cache()
            gc.collect()

        if {item["model"] for item in model_reports} != set(EXPECTED_MODELS):
            raise HardwarePreflightError("A required dataset-model preflight is absent")
        dataset_reports.append(
            {
                "dataset": dataset,
                "dataset_manifest_sha256": sha256_file(
                    project_root
                    / "data"
                    / "manifests"
                    / (dataset + "_dataset.json")
                ),
                "entities": entity_count,
                "relations": relation_count,
                "training_triples": int(len(train)),
                "training_seen_entities": int(len(replacement_entities)),
                "batch_order_seed": batch_seed,
                "negative_sampling_seed": negative_seed,
                "training_sequence_sha256": sequence_sha,
                "true_training_triples": len(true_training),
                "models": model_reports,
            }
        )

    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "status": "hardware_feasibility_preflight_passed_pending_hpo_design",
        "official_result": False,
        "validation_or_test_metrics_computed": False,
        "configuration_sha256": sha256_file(config_path),
        "execution": execution,
        "git": git,
        "torch_runtime": {
            "torch_version": torch.__version__,
            "compiled_cuda_version": torch.version.cuda,
            "device_name": properties.name,
            "device_total_memory_bytes": int(properties.total_memory),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        },
        "frozen_maximum_shape": {
            "embedding_dimension": int(config["embedding_dimension"]),
            "batch_size": int(config["batch_size"]),
            "negatives_per_positive": int(config["negatives_per_positive"]),
            "training_steps": int(config["training_steps"]),
        },
        "datasets": dataset_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    try:
        report = run_preflight(
            args.config.resolve(),
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "hardware_feasibility_preflight_failed",
            "official_result": False,
            "validation_or_test_metrics_computed": False,
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_bytes_new_or_identical(
            output,
            canonical_json_bytes(failure),
            str(os.environ.get("SLURM_JOB_ID", "unknown")),
        )
        raise
    write_bytes_new_or_identical(
        output,
        canonical_json_bytes(report),
        str(report["execution"]["scheduler_job_id"]),
    )
    print("Hardware feasibility preflight recorded in {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
