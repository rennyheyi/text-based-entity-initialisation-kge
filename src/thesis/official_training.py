"""Train one frozen official KGE run without accessing test outcomes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .kg_core import batch_training_loss, sample_negative_triples_vectorized
from .official_design import validate_design_invariants
from .official_shuffle import OFFICIAL_DESIGN_SHA256, verify_permutation
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object
from .training_smoke import load_encoded_hashes


class OfficialTrainingError(RuntimeError):
    """Raised when an official training invariant is violated."""


HPO_DESIGN_SHA256 = "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93"
EXPECTED_DEVICE = "NVIDIA H200 NVL"
EXPECTED_RUNS = 90
MAXIMUM_NEGATIVE_SAMPLING_ATTEMPTS = 1000


def select_official_run(design: Mapping[str, Any], array_index: int) -> Mapping[str, Any]:
    validate_design_invariants(design)
    if not 0 <= array_index < EXPECTED_RUNS:
        raise OfficialTrainingError("Official array index is outside 0..89")
    row = design["runs"][array_index]
    if row["array_index"] != array_index:
        raise OfficialTrainingError("Official array mapping is not contiguous")
    return row


def configure_numerics(torch: Any) -> Dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise OfficialTrainingError("CUBLAS_WORKSPACE_CONFIG is not frozen")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "parameter_dtype": "float32",
        "automatic_mixed_precision": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _load_text_matrix(
    run: Mapping[str, Any], project_root: Path, entity_count: int
) -> Tuple[Any, Mapping[str, Any], Sequence[str], Sequence[str]]:
    import numpy as np

    dataset = str(run["dataset"])
    initialisation = run["entity_initialisation"]
    manifest_key = (
        "text_embedding_manifest"
        if run["condition"] == "correct_text"
        else "source_text_embedding_manifest"
    )
    manifest_hash_key = manifest_key + "_sha256"
    manifest_path = project_root / str(initialisation[manifest_key])
    if sha256_file(manifest_path) != initialisation[manifest_hash_key]:
        raise OfficialTrainingError("Text embedding manifest hash differs")
    manifest = read_json_object(manifest_path)
    artifact = manifest["artifacts"]["normalised_embeddings"]
    expected_embedding_hash = (
        initialisation["normalised_embeddings_sha256"]
        if run["condition"] == "correct_text"
        else initialisation["source_normalised_embeddings_sha256"]
    )
    if artifact["sha256"] != expected_embedding_hash:
        raise OfficialTrainingError("Text embedding design/manifest hash differs")
    embedding_path = project_root / "artifacts" / "embeddings" / dataset / artifact["filename"]
    if sha256_file(embedding_path) != expected_embedding_hash:
        raise OfficialTrainingError("Normalised text embedding artifact hash differs")
    matrix = np.load(embedding_path, allow_pickle=False)
    if matrix.shape != (entity_count, 384) or matrix.dtype != np.float32:
        raise OfficialTrainingError("Normalised text embedding shape/dtype differs")
    evidence_table = project_root / "artifacts" / "descriptions" / dataset / "evidence_encoded.tsv"
    encoded_artifact = manifest["artifacts"]["encoded_evidence_table"]
    if sha256_file(evidence_table) != encoded_artifact["sha256"]:
        raise OfficialTrainingError("Encoded evidence table hash differs")
    evidence_hashes, vector_hashes = load_encoded_hashes(evidence_table, entity_count)
    return matrix, {
        "manifest_path": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "embedding_path": str(embedding_path.relative_to(project_root)),
        "embedding_sha256": sha256_file(embedding_path),
        "encoded_evidence_table_sha256": sha256_file(evidence_table),
    }, evidence_hashes, vector_hashes


def load_entity_initialisation(
    run: Mapping[str, Any], project_root: Path, entity_count: int
) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np
    from .hpo_stage1 import random_unit_array

    condition = str(run["condition"])
    if condition == "random":
        seed = int(run["rng_streams"]["random_entity_initialisation"]["seed"])
        matrix = random_unit_array(entity_count, 384, seed)
        return matrix, {
            "kind": "random_unit_norm_float32",
            "seed": seed,
            "matrix_raw_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
        }
    matrix, source, evidence_hashes, vector_hashes = _load_text_matrix(
        run, project_root, entity_count
    )
    if condition == "correct_text":
        return matrix.copy(), {
            "kind": "frozen_l2_normalised_correct_text",
            "source": source,
            "matrix_raw_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
        }
    if condition != "shuffled_text":
        raise OfficialTrainingError("Unknown official initialisation condition")
    specifications = [
        row
        for row in read_json_object(project_root / "configs/official_experiment_design.json")[
            "shuffle_artifacts"
        ]
        if row["shuffle_id"] == run["shuffle_id"]
    ]
    if len(specifications) != 1:
        raise OfficialTrainingError("Official shuffle specification is not unique")
    specification = specifications[0]
    manifest_path = project_root / specification["permutation_manifest"]
    manifest = read_json_object(manifest_path)
    if (
        manifest.get("shuffle_id") != run["shuffle_id"]
        or manifest.get("official_design_sha256") != OFFICIAL_DESIGN_SHA256
        or manifest.get("status") != "official_shuffle_materialised_pending_training"
    ):
        raise OfficialTrainingError("Official shuffle manifest differs")
    artifact_path = project_root / specification["permutation_artifact"]
    artifact = manifest["permutation_artifact"]
    if sha256_file(artifact_path) != artifact["sha256"]:
        raise OfficialTrainingError("Official shuffle artifact hash differs")
    permutation = np.load(artifact_path, allow_pickle=False)
    observed = verify_permutation(permutation, evidence_hashes, vector_hashes)
    if observed != manifest["invariants"]:
        raise OfficialTrainingError("Official shuffle invariants differ")
    shuffled = matrix[np.asarray(permutation, dtype=np.int64)].copy()
    if sorted(vector_hashes[int(index)] for index in permutation) != sorted(vector_hashes):
        raise OfficialTrainingError("Shuffled and correct vector multisets differ")
    return shuffled, {
        "kind": "frozen_l2_normalised_shuffled_text",
        "source": source,
        "shuffle_id": run["shuffle_id"],
        "shuffle_manifest": specification["permutation_manifest"],
        "shuffle_manifest_sha256": sha256_file(manifest_path),
        "permutation_artifact": specification["permutation_artifact"],
        "permutation_artifact_sha256": sha256_file(artifact_path),
        "permutation_raw_sha256": observed["permutation_raw_sha256"],
        "matrix_raw_sha256": hashlib.sha256(shuffled.tobytes(order="C")).hexdigest(),
    }


def run_official_training(
    design_path: Path,
    hpo_design_path: Path,
    preflight_path: Path,
    *,
    array_index: int,
    project_root: Path,
    scratch: Path,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from .hpo_stage1 import (
        initialisation_statistics,
        load_stage_data,
        random_unit_array,
        save_checkpoint,
        save_numpy_artifact,
        training_environment,
    )

    execution = require_scheduled_execution(scratch.resolve())
    if sha256_file(design_path) != OFFICIAL_DESIGN_SHA256:
        raise OfficialTrainingError("Official design hash differs")
    if sha256_file(hpo_design_path) != HPO_DESIGN_SHA256:
        raise OfficialTrainingError("Frozen HPO design hash differs")
    design = read_json_object(design_path)
    hpo_design = read_json_object(hpo_design_path)
    run = select_official_run(design, array_index)
    if sha256_file(preflight_path) != hpo_design["hardware_preflight_report_sha256"]:
        raise OfficialTrainingError("Hardware preflight report hash differs")
    preflight = read_json_object(preflight_path)
    git = git_state(project_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise OfficialTrainingError("Exactly one scheduler-visible CUDA device is required")
    numerical_policy = configure_numerics(torch)
    environment = training_environment(preflight)
    if environment["device_name"] != EXPECTED_DEVICE:
        raise OfficialTrainingError("Official training did not run on the frozen H200 device")

    dataset = str(run["dataset"])
    model = str(run["model"])
    condition = str(run["condition"])
    preflight_dataset = [row for row in preflight["datasets"] if row["dataset"] == dataset]
    if len(preflight_dataset) != 1:
        raise OfficialTrainingError("Dataset is absent from hardware preflight")
    data = load_stage_data(
        project_root, dataset, str(preflight_dataset[0]["dataset_manifest_sha256"])
    )
    entity_numpy, entity_provenance = load_entity_initialisation(
        run, project_root, data["entity_count"]
    )
    relation_seed = int(run["rng_streams"]["relation_initialisation"]["seed"])
    relation_numpy = random_unit_array(data["relation_count"], 384, relation_seed)
    initial_entity_sha = hashlib.sha256(entity_numpy.tobytes(order="C")).hexdigest()
    initial_relation_sha = hashlib.sha256(relation_numpy.tobytes(order="C")).hexdigest()
    entity_statistics = initialisation_statistics(entity_numpy)
    relation_statistics = initialisation_statistics(relation_numpy)
    entity = torch.nn.Parameter(torch.as_tensor(entity_numpy, dtype=torch.float32, device="cuda"))
    relation = torch.nn.Parameter(torch.as_tensor(relation_numpy, dtype=torch.float32, device="cuda"))
    globally_unseen = data["globally_train_unseen_entities"]
    globally_unseen_device = torch.as_tensor(globally_unseen, dtype=torch.long, device="cuda")
    unseen_initial = (
        entity[globally_unseen_device].detach().clone() if len(globally_unseen) else None
    )
    selected = run["selected_configuration"]
    parameters = selected["parameters"]
    optimiser = torch.optim.Adam(
        [entity, relation],
        lr=float(parameters["learning_rate"]),
        betas=(
            float(design["fixed_training"]["adam_beta1"]),
            float(design["fixed_training"]["adam_beta2"]),
        ),
        eps=float(design["fixed_training"]["adam_epsilon"]),
        weight_decay=float(design["fixed_training"]["weight_decay"]),
    )
    batch_size = int(parameters["batch_size"])
    negatives_per_positive = int(parameters["negatives_per_positive"])
    regularisation_coefficient = float(parameters["regularisation_coefficient"])
    transe_distance_norm = int(parameters.get("transe_distance_norm", 1))
    epochs = int(selected["training_epochs"])
    batch_rng = np.random.default_rng(int(run["rng_streams"]["batch_order"]["seed"]))
    negative_rng = np.random.default_rng(int(run["rng_streams"]["negative_sampling"]["seed"]))
    sequence_digest = hashlib.sha256()
    loss_rows = []
    epoch_history = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    print(
        "OFFICIAL_TRAINING_START "
        + json.dumps(
            {
                "array_index": array_index,
                "run_id": run["run_id"],
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "official_replicate": run["official_replicate"],
                "epochs": epochs,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        first = len(loss_rows)
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
                maximum_attempts=MAXIMUM_NEGATIVE_SAMPLING_ATTEMPTS,
            )
            sequence_digest.update(positives_numpy.astype("<i8", copy=False).tobytes())
            sequence_digest.update(negatives_numpy.astype("<i8", copy=False).tobytes())
            positives = torch.as_tensor(positives_numpy, dtype=torch.long, device="cuda")
            negatives = torch.as_tensor(negatives_numpy, dtype=torch.long, device="cuda")
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
                raise OfficialTrainingError("Official training loss is non-finite")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            loss_rows.append(
                (
                    float(loss.detach().cpu()),
                    float(ranking_loss.detach().cpu()),
                    float(regularisation.detach().cpu()),
                )
            )
        if not torch.isfinite(entity).all() or not torch.isfinite(relation).all():
            raise OfficialTrainingError("Official training produced a non-finite embedding")
        epoch_values = np.asarray(loss_rows[first:], dtype=np.float64)[:, 0]
        record = {
            "epoch": epoch,
            "batches": int(len(epoch_values)),
            "mean_total_loss": float(epoch_values.mean()),
            "minimum_total_loss": float(epoch_values.min()),
            "maximum_total_loss": float(epoch_values.max()),
            "runtime_seconds": float(time.perf_counter() - epoch_started),
        }
        epoch_history.append(record)
        print("OFFICIAL_TRAINING_EPOCH " + json.dumps(record, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    unseen_change = 0.0
    if len(globally_unseen):
        unseen_change = float(
            torch.max(torch.abs(entity[globally_unseen_device] - unseen_initial))
            .detach()
            .cpu()
        )
        if unseen_change != 0.0:
            raise OfficialTrainingError("A globally training-unseen entity changed")

    report_path = project_root / str(run["expected_run_report"])
    checkpoint_path = project_root / str(run["expected_checkpoint"])
    run_directory = checkpoint_path.parent
    run_directory.mkdir(parents=True, exist_ok=True)
    job_id = str(execution["scheduler_job_id"])
    loss_artifact = save_numpy_artifact(
        np.asarray(loss_rows, dtype=np.float64),
        run_directory / "batch_loss_history.npy",
        scratch=scratch,
        job_id=job_id,
        label="official_batch_loss_history",
    )
    checkpoint = save_checkpoint(
        {
            "schema_version": 1,
            "status": "official_checkpoint_frozen_pending_calibration_and_test_evaluation",
            "run_id": run["run_id"],
            "array_index": array_index,
            "dataset": dataset,
            "model": model,
            "condition": condition,
            "official_replicate": run["official_replicate"],
            "official_seed": run["official_seed"],
            "run_configuration_sha256": run["run_configuration_sha256"],
            "selected_configuration": selected,
            "training_sequence_sha256": sequence_digest.hexdigest(),
            "entity_embeddings": entity.detach().cpu(),
            "relation_embeddings": relation.detach().cpu(),
        },
        checkpoint_path,
        scratch=scratch,
        job_id=job_id,
    )
    report = {
        "schema_version": 1,
        "status": "official_training_completed_pending_calibration_and_test_evaluation",
        "official_result": False,
        "test_data_accessed": False,
        "validation_metrics_computed": False,
        "test_metrics_computed": False,
        "array_index": array_index,
        "run_id": run["run_id"],
        "paired_block_id": run["paired_block_id"],
        "dataset": dataset,
        "model": model,
        "condition": condition,
        "official_replicate": run["official_replicate"],
        "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "selected_configuration": selected,
        "rng_streams": run["rng_streams"],
        "data": {
            "dataset_manifest_sha256": sha256_file(data["manifest_path"]),
            "entities": data["entity_count"],
            "relations": data["relation_count"],
            "training_triples": int(len(data["train"])),
            "training_seen_entities": int(len(data["train_seen_entities"])),
            "globally_training_unseen_entities": int(len(globally_unseen)),
        },
        "initialisation": {
            "dimension": 384,
            "entity": entity_provenance,
            "entity_matrix_raw_sha256": initial_entity_sha,
            "relation_seed": relation_seed,
            "relation_matrix_raw_sha256": initial_relation_sha,
            "entity_statistics": entity_statistics,
            "relation_statistics": relation_statistics,
        },
        "training": {
            "epochs": epochs,
            "batches": len(loss_rows),
            "training_sequence_sha256": sequence_digest.hexdigest(),
            "epoch_history": epoch_history,
            "runtime_seconds": float(runtime),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "globally_training_unseen_maximum_absolute_change": unseen_change,
            "batch_loss_columns": [
                "total_loss",
                "ranking_loss",
                "unweighted_batch_local_l2",
            ],
            "maximum_negative_sampling_attempts": MAXIMUM_NEGATIVE_SAMPLING_ATTEMPTS,
            "rng_algorithm": "NumPy Generator(PCG64)",
            "data_loader_workers": 0,
        },
        "artifacts": {"checkpoint": checkpoint, "batch_loss_history": loss_artifact},
        "official_design_sha256": sha256_file(design_path),
        "hpo_design_sha256": sha256_file(hpo_design_path),
        "hardware_preflight_report_sha256": sha256_file(preflight_path),
        "execution": {
            **execution,
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "numerical_policy": numerical_policy,
        "training_environment": environment,
        "git": git,
    }
    write_bytes_new_or_identical(report_path, canonical_json_bytes(report), job_id)
    del entity, relation, optimiser
    torch.cuda.empty_cache()
    gc.collect()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--hpo-design", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_official_training(
            args.design.resolve(),
            args.hpo_design.resolve(),
            args.preflight_report.resolve(),
            array_index=args.array_index,
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "official_training_failed",
            "official_result": False,
            "test_data_accessed": False,
            "array_index": args.array_index,
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failure_path = (
            args.project_root.resolve()
            / "reports"
            / "official_failures"
            / "official_training_{}_{}.json".format(
                os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "unknown")),
                args.array_index,
            )
        )
        write_bytes_new_or_identical(
            failure_path,
            canonical_json_bytes(failure),
            str(os.environ.get("SLURM_JOB_ID", "unknown")),
        )
        raise
    print(
        "Official training completed for {} pending calibration and test evaluation".format(
            report["run_id"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
