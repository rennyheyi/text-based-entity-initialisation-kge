"""Run one frozen HPO Stage 3 random-baseline final-tuning trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .hpo_stage1 import (
    HPO_DESIGN_SHA256,
    evaluate_validation,
    initialisation_statistics,
    load_stage_data,
    random_unit_array,
    save_checkpoint,
    save_numpy_artifact,
    training_environment,
    validate_design,
)
from .hpo_stage2 import (
    NAMESPACES,
    PROTOCOL_SHA256,
    STAGE2_DESIGN_SHA256,
    stream_seed,
    validate_stage2_design,
)
from .kg_core import batch_training_loss, sample_negative_triples_vectorized
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object


class HPOStage3Error(RuntimeError):
    """Raised when a Stage 3 trajectory violates its frozen design."""


STAGE3_DESIGN_SHA256 = "f96ac9322f7c78049dc9064ed03adc2dec74910b78d831eabe6a3af63b5974bc"
STAGE2_ATTEMPT_LOCK_SHA256 = "cce80551e4997c4bc573eb5f82d551fecba8f3450094170458962558dd09f157"
EXPECTED_DEVICE = "NVIDIA H200 NVL"
EXPECTED_UNITS = 6
EXPECTED_CANDIDATES_PER_UNIT = 2
EXPECTED_REPLICATES = (1, 2, 3)
EXPECTED_ARRAY_TASKS = 36
STAGE = 3
MAXIMUM_EPOCHS = 200
VALIDATION_EPOCHS = tuple(range(10, MAXIMUM_EPOCHS + 1, 10))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _full_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise HPOStage3Error(label + " is not a full SHA-256 digest")
    return value


def _finite_unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HPOStage3Error(label + " is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise HPOStage3Error(label + " is outside [0,1]")
    return number


def validate_stage3_design(
    stage3: Mapping[str, Any],
    stage2: Mapping[str, Any],
    source_design: Mapping[str, Any],
) -> None:
    validate_design(source_design)
    validate_stage2_design(stage2, source_design)
    exact = {
        "schema_version": 1,
        "status": "hpo_stage_3_design_frozen_pending_runs",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": STAGE,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "tuning_replicates": [1, 2, 3],
        "validation_epochs": list(VALIDATION_EPOCHS),
        "candidates_per_unit": EXPECTED_CANDIDATES_PER_UNIT,
        "trajectories_per_unit": 6,
        "selection_metric": "mean_combined_filtered_validation_mrr_by_candidate_epoch",
        "selection_target": "candidate_epoch_pair",
        "tie_breaking": [
            "earlier_epoch",
            "smaller_negatives_per_positive",
            "smaller_batch_size",
            "lexicographically_smaller_candidate_id",
        ],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_hpo_design_sha256": HPO_DESIGN_SHA256,
        "source_stage_2_design_sha256": STAGE2_DESIGN_SHA256,
        "source_stage_2_attempt_lock_sha256": STAGE2_ATTEMPT_LOCK_SHA256,
        "fixed_training": source_design["fixed_training"],
    }
    for key, value in exact.items():
        if stage3.get(key) != value:
            raise HPOStage3Error("Stage 3 design differs in " + key)
    _full_sha256(
        stage3.get("source_stage_2_report_manifest_sha256"),
        "Stage 2 report-manifest hash",
    )
    commit = stage3.get("source_stage_2_runner_git_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise HPOStage3Error("Stage 2 runner commit is invalid")
    units = stage3.get("units")
    stage2_units = stage2.get("units")
    if (
        not isinstance(units, list)
        or len(units) != EXPECTED_UNITS
        or not isinstance(stage2_units, list)
        or len(stage2_units) != EXPECTED_UNITS
    ):
        raise HPOStage3Error("Stage 3 tuning-unit count differs")
    all_candidate_ids = []
    for unit, stage2_unit in zip(units, stage2_units):
        for key in ("dataset", "model", "unit_id", "tuning_stream_derivations"):
            if unit.get(key) != stage2_unit.get(key):
                raise HPOStage3Error("Stage 3 unit differs in " + key)
        candidates = unit.get("advancing_candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise HPOStage3Error("A Stage 3 unit does not contain two candidates")
        if [candidate.get("stage_2_rank") for candidate in candidates] != [1, 2]:
            raise HPOStage3Error("Stage 2 advancing ranks differ")
        stage2_candidates = {
            candidate["candidate_id"]: candidate
            for candidate in stage2_unit["advancing_candidates"]
        }
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            source = stage2_candidates.get(candidate_id)
            if source is None:
                raise HPOStage3Error("A Stage 3 candidate is absent from Stage 2")
            expected = {
                "candidate_id": source["candidate_id"],
                "candidate_configuration_sha256": source[
                    "candidate_configuration_sha256"
                ],
                "parameters": source["parameters"],
                "stage_1_rank": source["stage_1_rank"],
            }
            for key, value in expected.items():
                if candidate.get(key) != value:
                    raise HPOStage3Error("A Stage 3 candidate differs in " + key)
            _finite_unit_interval(
                candidate.get("stage_2_mean_validation_mrr"),
                "Stage 2 mean validation MRR",
            )
            replicate_mrr = candidate.get("stage_2_replicate_validation_mrr")
            replicate_hashes = candidate.get("stage_2_report_sha256_by_replicate")
            if (
                not isinstance(replicate_mrr, dict)
                or set(replicate_mrr) != {"1", "2"}
                or not isinstance(replicate_hashes, dict)
                or set(replicate_hashes) != {"1", "2"}
            ):
                raise HPOStage3Error("Stage 2 replicate provenance differs")
            for replicate in ("1", "2"):
                _finite_unit_interval(
                    replicate_mrr[replicate], "Stage 2 replicate validation MRR"
                )
                _full_sha256(
                    replicate_hashes[replicate], "Stage 2 source report hash"
                )
            observed_mean = math.fsum(
                float(replicate_mrr[str(replicate)]) for replicate in (1, 2)
            ) / 2.0
            if observed_mean != float(candidate["stage_2_mean_validation_mrr"]):
                raise HPOStage3Error("Stage 2 replicate mean differs")
            all_candidate_ids.append(candidate_id)
    if len(all_candidate_ids) != 12 or len(set(all_candidate_ids)) != 12:
        raise HPOStage3Error("Stage 3 candidate IDs are incomplete or duplicated")


def select_array_task(
    stage3: Mapping[str, Any],
    stage2: Mapping[str, Any],
    source_design: Mapping[str, Any],
    array_index: int,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], int]:
    validate_stage3_design(stage3, stage2, source_design)
    if not 0 <= array_index < EXPECTED_ARRAY_TASKS:
        raise HPOStage3Error("Stage 3 array index is outside 0..35")
    tasks_per_unit = EXPECTED_CANDIDATES_PER_UNIT * len(EXPECTED_REPLICATES)
    unit = stage3["units"][array_index // tasks_per_unit]
    within_unit = array_index % tasks_per_unit
    candidate = unit["advancing_candidates"][within_unit // 3]
    replicate = EXPECTED_REPLICATES[within_unit % 3]
    return unit, candidate, replicate


def run_stage3(
    stage3_design_path: Path,
    stage2_design_path: Path,
    source_design_path: Path,
    preflight_path: Path,
    protocol_path: Path,
    *,
    array_index: int,
    project_root: Path,
    scratch: Path,
    artifact_root: Path,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    execution = require_scheduled_execution(scratch.resolve())
    frozen_hashes = {
        "Stage 3 design": (stage3_design_path, STAGE3_DESIGN_SHA256),
        "Stage 2 design": (stage2_design_path, STAGE2_DESIGN_SHA256),
        "source HPO design": (source_design_path, HPO_DESIGN_SHA256),
        "protocol": (protocol_path, PROTOCOL_SHA256),
    }
    for label, (path, expected) in frozen_hashes.items():
        if sha256_file(path) != expected:
            raise HPOStage3Error(label + " hash differs")
    stage3_design = read_json_object(stage3_design_path)
    stage2_design = read_json_object(stage2_design_path)
    source_design = read_json_object(source_design_path)
    unit, candidate, replicate = select_array_task(
        stage3_design, stage2_design, source_design, array_index
    )
    if sha256_file(preflight_path) != source_design["hardware_preflight_report_sha256"]:
        raise HPOStage3Error("Hardware-preflight report hash differs")
    preflight = read_json_object(preflight_path)
    source_git = git_state(project_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise HPOStage3Error("Exactly one scheduler-visible CUDA device is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    environment = training_environment(preflight)
    if environment["device_name"] != EXPECTED_DEVICE:
        raise HPOStage3Error("Stage 3 did not run on the frozen H200 device")
    dataset = str(unit["dataset"])
    model = str(unit["model"])
    candidate_id = str(candidate["candidate_id"])
    parameters = candidate["parameters"]
    preflight_dataset = [
        item for item in preflight["datasets"] if item["dataset"] == dataset
    ]
    if len(preflight_dataset) != 1:
        raise HPOStage3Error("Dataset is absent from hardware preflight")
    data = load_stage_data(
        project_root,
        dataset,
        str(preflight_dataset[0]["dataset_manifest_sha256"]),
    )
    seeds = {
        namespace: stream_seed(unit, replicate, namespace)
        for namespace in NAMESPACES
    }
    entity_numpy = random_unit_array(
        data["entity_count"], 384, seeds["random_entity_initialisation"]
    )
    relation_numpy = random_unit_array(
        data["relation_count"], 384, seeds["relation_initialisation"]
    )
    entity_initialisation_sha = hashlib.sha256(
        entity_numpy.tobytes(order="C")
    ).hexdigest()
    relation_initialisation_sha = hashlib.sha256(
        relation_numpy.tobytes(order="C")
    ).hexdigest()
    entity_statistics = initialisation_statistics(entity_numpy)
    relation_statistics = initialisation_statistics(relation_numpy)
    entity = torch.nn.Parameter(
        torch.as_tensor(entity_numpy, dtype=torch.float32, device="cuda")
    )
    relation = torch.nn.Parameter(
        torch.as_tensor(relation_numpy, dtype=torch.float32, device="cuda")
    )
    globally_unseen = data["globally_train_unseen_entities"]
    globally_unseen_device = torch.as_tensor(
        globally_unseen, dtype=torch.long, device="cuda"
    )
    unseen_initial = (
        entity[globally_unseen_device].detach().clone()
        if len(globally_unseen)
        else None
    )
    optimiser = torch.optim.Adam(
        [entity, relation],
        lr=float(parameters["learning_rate"]),
        betas=(
            float(stage3_design["fixed_training"]["adam_beta1"]),
            float(stage3_design["fixed_training"]["adam_beta2"]),
        ),
        eps=float(stage3_design["fixed_training"]["adam_epsilon"]),
        weight_decay=float(stage3_design["fixed_training"]["weight_decay"]),
    )
    batch_size = int(parameters["batch_size"])
    negatives_per_positive = int(parameters["negatives_per_positive"])
    regularisation_coefficient = float(parameters["regularisation_coefficient"])
    transe_distance_norm = int(parameters.get("transe_distance_norm", 1))
    batch_rng = np.random.default_rng(seeds["batch_order"])
    negative_rng = np.random.default_rng(seeds["negative_sampling"])
    sequence_digest = hashlib.sha256()
    batch_losses: List[float] = []
    ranking_losses: List[float] = []
    regularisation_values: List[float] = []
    epoch_history = []
    validation_history = []
    validation_rank_tables: List[Tuple[int, bytes]] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    trajectory_started = time.perf_counter()
    print(
        "HPO_STAGE3_START "
        + json.dumps(
            {
                "array_index": array_index,
                "dataset": dataset,
                "model": model,
                "candidate_id": candidate_id,
                "replicate": replicate,
                "parameters": parameters,
                "maximum_epochs": MAXIMUM_EPOCHS,
                "validation_epochs": list(VALIDATION_EPOCHS),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epoch_started = time.perf_counter()
        epoch_first_batch = len(batch_losses)
        order = batch_rng.permutation(len(data["train"]))
        sequence_digest.update(epoch.to_bytes(8, "big"))
        sequence_digest.update(order.astype("<i8", copy=False).tobytes())
        for start in range(0, len(order), batch_size):
            positives_numpy = data["train"][order[start : start + batch_size]].copy()
            negatives_numpy = sample_negative_triples_vectorized(
                positives_numpy,
                negatives_per_positive=negatives_per_positive,
                replacement_entities=data["train_seen_entities"],
                sorted_true_training_keys=data["sorted_train_keys"],
                entity_count=data["entity_count"],
                relation_count=data["relation_count"],
                rng=negative_rng,
                maximum_attempts=1000,
            )
            sequence_digest.update(
                positives_numpy.astype("<i8", copy=False).tobytes()
            )
            sequence_digest.update(
                negatives_numpy.astype("<i8", copy=False).tobytes()
            )
            positives = torch.as_tensor(
                positives_numpy, dtype=torch.long, device="cuda"
            )
            negatives = torch.as_tensor(
                negatives_numpy, dtype=torch.long, device="cuda"
            )
            ranking_loss, regularisation, loss = batch_training_loss(
                model,
                entity,
                relation,
                positives,
                negatives,
                regularisation_coefficient=regularisation_coefficient,
                transe_distance_norm=transe_distance_norm,
            )
            if not torch.isfinite(loss):
                raise HPOStage3Error("Stage 3 training loss is non-finite")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            referenced = torch.cat((positives, negatives.reshape(-1, 3)), dim=0)
            referenced_entities = torch.unique(referenced[:, [0, 2]])
            referenced_relations = torch.unique(referenced[:, 1])
            if not torch.isfinite(entity[referenced_entities]).all() or not torch.isfinite(
                relation[referenced_relations]
            ).all():
                raise HPOStage3Error("Stage 3 training produced a non-finite embedding")
            batch_losses.append(float(loss.detach().cpu()))
            ranking_losses.append(float(ranking_loss.detach().cpu()))
            regularisation_values.append(float(regularisation.detach().cpu()))
        if not torch.isfinite(entity).all() or not torch.isfinite(relation).all():
            raise HPOStage3Error("A Stage 3 epoch ended with a non-finite embedding")
        epoch_values = np.asarray(
            batch_losses[epoch_first_batch:], dtype=np.float64
        )
        epoch_record = {
            "epoch": epoch,
            "batches": int(len(epoch_values)),
            "mean_total_loss": float(epoch_values.mean()),
            "minimum_total_loss": float(epoch_values.min()),
            "maximum_total_loss": float(epoch_values.max()),
            "runtime_seconds": float(time.perf_counter() - epoch_started),
        }
        epoch_history.append(epoch_record)
        print(
            "HPO_STAGE3_EPOCH " + json.dumps(epoch_record, sort_keys=True),
            flush=True,
        )
        if epoch in VALIDATION_EPOCHS:
            print("HPO_STAGE3_VALIDATION_START epoch={}".format(epoch), flush=True)
            evaluation_started = time.perf_counter()
            validation, rank_table = evaluate_validation(
                model,
                entity,
                relation,
                data["train"],
                data["valid"],
                data["train_seen_entities"],
                transe_distance_norm=transe_distance_norm,
            )
            torch.cuda.synchronize()
            evaluation_runtime = time.perf_counter() - evaluation_started
            validation_history.append(
                {
                    "epoch": epoch,
                    "validation": validation,
                    "runtime_seconds": float(evaluation_runtime),
                }
            )
            validation_rank_tables.append((epoch, rank_table))
            print(
                "HPO_STAGE3_VALIDATION_DONE "
                + json.dumps(
                    {
                        "epoch": epoch,
                        "combined_mrr": validation["combined"]["mrr"],
                        "runtime_seconds": float(evaluation_runtime),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize()
    trajectory_runtime = time.perf_counter() - trajectory_started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    unseen_change = 0.0
    if len(globally_unseen):
        unseen_change = float(
            torch.max(
                torch.abs(entity[globally_unseen_device] - unseen_initial)
            ).detach().cpu()
        )
        if unseen_change != 0.0:
            raise HPOStage3Error("A globally training-unseen entity changed")
    array_job_id = str(
        os.environ.get("SLURM_ARRAY_JOB_ID", execution["scheduler_job_id"])
    )
    attempt_name = "attempt-{}-{}".format(array_job_id, array_index)
    run_directory = (
        artifact_root
        / str(unit["unit_id"])
        / candidate_id
        / ("replicate-{}".format(replicate))
        / attempt_name
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    job_id = str(execution["scheduler_job_id"])
    shared_initialisation_directory = (
        artifact_root
        / "_shared_initialisations"
        / str(unit["unit_id"])
        / ("replicate-{}".format(replicate))
    )
    entity_initialisation_artifact = save_numpy_artifact(
        entity_numpy,
        shared_initialisation_directory / "entity_initialisation.npy",
        scratch=scratch,
        job_id=job_id,
        label="entity_initialisation",
    )
    relation_initialisation_artifact = save_numpy_artifact(
        relation_numpy,
        shared_initialisation_directory / "relation_initialisation.npy",
        scratch=scratch,
        job_id=job_id,
        label="relation_initialisation",
    )
    losses_matrix = np.column_stack(
        (
            np.asarray(batch_losses, dtype=np.float64),
            np.asarray(ranking_losses, dtype=np.float64),
            np.asarray(regularisation_values, dtype=np.float64),
        )
    )
    loss_artifact = save_numpy_artifact(
        losses_matrix,
        run_directory / "batch_loss_history.npy",
        scratch=scratch,
        job_id=job_id,
        label="batch_loss_history",
    )
    rank_artifacts = []
    for epoch, rank_table in validation_rank_tables:
        path = run_directory / "validation_epoch_{:03d}_ranks.tsv".format(epoch)
        write_bytes_new_or_identical(path, rank_table, job_id)
        rank_artifacts.append(
            {
                "epoch": epoch,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": int(2 * len(data["valid"])),
            }
        )
    checkpoint = save_checkpoint(
        {
            "schema_version": 1,
            "stage": STAGE,
            "replicate": replicate,
            "dataset": dataset,
            "model": model,
            "candidate_id": candidate_id,
            "parameters": parameters,
            "epoch": MAXIMUM_EPOCHS,
            "validation_mrr_history": [
                {
                    "epoch": item["epoch"],
                    "combined_mrr": item["validation"]["combined"]["mrr"],
                }
                for item in validation_history
            ],
            "entity_embeddings": entity.detach().cpu(),
            "relation_embeddings": relation.detach().cpu(),
        },
        run_directory / "checkpoint_epoch_200.pt",
        scratch=scratch,
        job_id=job_id,
    )
    report = {
        "schema_version": 1,
        "status": "hpo_stage_3_run_completed",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": STAGE,
        "replicate": replicate,
        "array_index": array_index,
        "array_job_id": array_job_id,
        "dataset": dataset,
        "model": model,
        "unit_id": unit["unit_id"],
        "candidate_id": candidate_id,
        "candidate_configuration_sha256": candidate[
            "candidate_configuration_sha256"
        ],
        "parameters": parameters,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "validation_epochs": list(VALIDATION_EPOCHS),
        "stage_1_checkpoint_loaded": False,
        "stage_2_checkpoint_loaded": False,
        "prior_stage_provenance": {
            "stage_1_rank": candidate["stage_1_rank"],
            "stage_2_rank": candidate["stage_2_rank"],
            "stage_2_mean_validation_mrr": candidate[
                "stage_2_mean_validation_mrr"
            ],
            "stage_2_replicate_validation_mrr": candidate[
                "stage_2_replicate_validation_mrr"
            ],
            "stage_2_report_sha256_by_replicate": candidate[
                "stage_2_report_sha256_by_replicate"
            ],
        },
        "seeds": seeds,
        "data": {
            "dataset_manifest_sha256": sha256_file(data["manifest_path"]),
            "entities": data["entity_count"],
            "relations": data["relation_count"],
            "training_triples": int(len(data["train"])),
            "validation_triples": int(len(data["valid"])),
            "training_seen_entities": int(len(data["train_seen_entities"])),
            "globally_training_unseen_entities": int(len(globally_unseen)),
        },
        "initialisation": {
            "dimension": 384,
            "source": "regenerated_from_frozen_tuning_stream",
            "entity_matrix_raw_sha256": entity_initialisation_sha,
            "relation_matrix_raw_sha256": relation_initialisation_sha,
            "entity_statistics": entity_statistics,
            "relation_statistics": relation_statistics,
            "entity_artifact": entity_initialisation_artifact,
            "relation_artifact": relation_initialisation_artifact,
        },
        "training": {
            "batches": len(batch_losses),
            "training_sequence_sha256": sequence_digest.hexdigest(),
            "epoch_history": epoch_history,
            "runtime_seconds_including_validation": float(trajectory_runtime),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "globally_training_unseen_maximum_absolute_change": unseen_change,
            "batch_loss_columns": [
                "total_loss",
                "ranking_loss",
                "unweighted_batch_local_l2",
            ],
            "rng_algorithm": "NumPy Generator(PCG64)",
            "data_loader_workers": 0,
        },
        "validation_history": validation_history,
        "artifacts": {
            "checkpoint_epoch_200": checkpoint,
            "batch_loss_history": loss_artifact,
            "validation_ranks_by_epoch": rank_artifacts,
        },
        "stage_3_design_sha256": sha256_file(stage3_design_path),
        "source_stage_2_design_sha256": sha256_file(stage2_design_path),
        "source_hpo_design_sha256": sha256_file(source_design_path),
        "hardware_preflight_report_sha256": sha256_file(preflight_path),
        "protocol_sha256": sha256_file(protocol_path),
        "execution": {
            **execution,
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "training_environment": environment,
        "git": source_git,
        "run_directory": str(run_directory),
    }
    del entity, relation, optimiser
    torch.cuda.empty_cache()
    gc.collect()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-design", type=Path, required=True)
    parser.add_argument("--stage2-design", type=Path, required=True)
    parser.add_argument("--source-design", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    try:
        report = run_stage3(
            args.stage3_design.resolve(),
            args.stage2_design.resolve(),
            args.source_design.resolve(),
            args.preflight_report.resolve(),
            args.protocol.resolve(),
            array_index=args.array_index,
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
            artifact_root=args.artifact_root.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "hpo_stage_3_run_failed",
            "official_result": False,
            "test_data_accessed": False,
            "text_conditions_accessed": False,
            "stage": STAGE,
            "array_index": args.array_index,
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
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
    print("HPO Stage 3 run recorded in {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
