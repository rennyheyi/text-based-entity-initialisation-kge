"""Fit one frozen official run temperature using validation queries only."""

from __future__ import annotations

import argparse
import gc
import os
import platform
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .official_design import (
    EXPECTED_OFFICIAL_DESIGN_SHA256,
    canonical_json_bytes,
    git_state,
    read_json_object,
    require_scheduled_execution,
    sha256_file,
    validate_design_invariants,
    write_new_or_identical,
)
from .official_evaluation_core import (
    CANDIDATE_CHUNK_SIZE,
    QUERY_BATCH_SIZE,
    fit_validation_temperature,
    validation_score_matrix,
)


OFFICIAL_DESIGN_SHA256 = EXPECTED_OFFICIAL_DESIGN_SHA256
EXPECTED_DEVICE = "NVIDIA H200 NVL"


class OfficialCalibrationError(RuntimeError):
    """Raised when validation-only official calibration cannot be completed."""


OFFICIAL_CHECKPOINT_LOCK_SHA256 = (
    "220e4dd3d74a7a3699a32c64fd763f99b6af97dbcf9d1fedbd2ddebbe27f7a86"
)
OFFICIAL_EVALUATION_LOCK_SHA256 = (
    "ffc14e3084a63d336e6da4c8b6dbe950e8f41cb6558b7e244809be7d9cccf4fa"
)
EXPECTED_RUNS = 90


def select_run_and_checkpoint(
    design: Mapping[str, Any],
    checkpoint_lock: Mapping[str, Any],
    array_index: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    validate_design_invariants(design)
    if not 0 <= array_index < EXPECTED_RUNS:
        raise OfficialCalibrationError("Calibration array index is outside 0..89")
    run = design["runs"][array_index]
    if run["array_index"] != array_index:
        raise OfficialCalibrationError("Official calibration array mapping differs")
    if (
        checkpoint_lock.get("status")
        != "official_checkpoints_frozen_pending_calibration_and_test_evaluation"
        or checkpoint_lock.get("official_result") is not False
        or checkpoint_lock.get("test_data_accessed") is not False
        or checkpoint_lock.get("validation_metrics_computed") is not False
        or checkpoint_lock.get("test_metrics_computed") is not False
        or checkpoint_lock.get("counts", {}).get("official_runs") != EXPECTED_RUNS
    ):
        raise OfficialCalibrationError("Official checkpoint lock boundary differs")
    matches = [
        row for row in checkpoint_lock.get("runs", ()) if row.get("run_id") == run["run_id"]
    ]
    if len(matches) != 1:
        raise OfficialCalibrationError("Checkpoint lock run identity is not unique")
    locked = matches[0]
    exact = {
        "array_index": run["array_index"],
        "dataset": run["dataset"],
        "model": run["model"],
        "condition": run["condition"],
        "official_replicate": run["official_replicate"],
        "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "selected_configuration": run["selected_configuration"],
    }
    for key, expected in exact.items():
        if locked.get(key) != expected:
            raise OfficialCalibrationError("Checkpoint lock field differs: " + key)
    return run, locked


def validate_evaluation_lock(lock: Mapping[str, Any]) -> None:
    if (
        lock.get("schema_version") != 1
        or lock.get("status")
        != "official_evaluation_procedure_frozen_pending_validation_calibration"
        or lock.get("official_result") is not False
        or lock.get("test_data_accessed") is not False
    ):
        raise OfficialCalibrationError("Official evaluation lock boundary differs")
    expected_inputs = {
        "official_experiment_design_sha256": OFFICIAL_DESIGN_SHA256,
        "official_checkpoint_lock_sha256": OFFICIAL_CHECKPOINT_LOCK_SHA256,
    }
    for key, expected in expected_inputs.items():
        if lock.get("inputs", {}).get(key) != expected:
            raise OfficialCalibrationError("Official evaluation input differs: " + key)
    temperature = lock.get("temperature_scaling", {})
    expected_temperature = {
        "temperature_lower_bound": 0.001,
        "temperature_upper_bound": 1000.0,
        "optimiser": "scipy.optimize.minimize_scalar_bounded",
        "xatol_log_temperature": 1e-06,
        "maximum_iterations": 100,
        "objective_reduction_dtype": "float64",
        "test_queries_used_for_fitting": False,
    }
    for key, expected in expected_temperature.items():
        if temperature.get(key) != expected:
            raise OfficialCalibrationError("Temperature procedure differs: " + key)
    scoring = lock.get("scoring", {})
    if (
        scoring.get("candidate_score_dtype") != "float32"
        or scoring.get("query_batch_size") != QUERY_BATCH_SIZE
        or scoring.get("candidate_chunk_size") != CANDIDATE_CHUNK_SIZE
    ):
        raise OfficialCalibrationError("Frozen scoring procedure differs")


def _load_checkpoint(
    project_root: Path,
    run: Mapping[str, Any],
    locked: Mapping[str, Any],
) -> Mapping[str, Any]:
    import torch

    checkpoint_record = locked.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise OfficialCalibrationError("Checkpoint record is missing from lock")
    checkpoint_path = project_root / str(checkpoint_record["path"])
    if checkpoint_path.resolve() != (project_root / run["expected_checkpoint"]).resolve():
        raise OfficialCalibrationError("Checkpoint path differs from official design")
    if sha256_file(checkpoint_path) != checkpoint_record.get("sha256"):
        raise OfficialCalibrationError("Checkpoint hash differs from checkpoint lock")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise OfficialCalibrationError("Checkpoint payload is invalid")
    exact = {
        "run_id": run["run_id"],
        "array_index": run["array_index"],
        "dataset": run["dataset"],
        "model": run["model"],
        "condition": run["condition"],
        "official_replicate": run["official_replicate"],
        "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "selected_configuration": run["selected_configuration"],
        "training_sequence_sha256": locked["training_sequence_sha256"],
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise OfficialCalibrationError("Checkpoint payload field differs: " + key)
    for key, shape_key in (
        ("entity_embeddings", "entity_shape"),
        ("relation_embeddings", "relation_shape"),
    ):
        tensor = payload.get(key)
        if tensor is None or list(tensor.shape) != checkpoint_record[shape_key]:
            raise OfficialCalibrationError("Checkpoint tensor shape differs: " + key)
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise OfficialCalibrationError("Checkpoint tensor is not finite float32: " + key)
    return payload


def run_official_calibration(
    *,
    design_path: Path,
    checkpoint_lock_path: Path,
    evaluation_lock_path: Path,
    array_index: int,
    project_root: Path,
    scratch: Path,
) -> Dict[str, Any]:
    import numpy as np
    import scipy
    import torch
    from .hpo_stage1 import load_stage_data
    from .official_training import configure_numerics

    execution = require_scheduled_execution(scratch.resolve())
    if sha256_file(design_path) != OFFICIAL_DESIGN_SHA256:
        raise OfficialCalibrationError("Official design hash differs")
    if sha256_file(checkpoint_lock_path) != OFFICIAL_CHECKPOINT_LOCK_SHA256:
        raise OfficialCalibrationError("Official checkpoint lock hash differs")
    if sha256_file(evaluation_lock_path) != OFFICIAL_EVALUATION_LOCK_SHA256:
        raise OfficialCalibrationError("Official evaluation lock hash differs")
    design = read_json_object(design_path)
    checkpoint_lock = read_json_object(checkpoint_lock_path)
    evaluation_lock = read_json_object(evaluation_lock_path)
    validate_evaluation_lock(evaluation_lock)
    run, locked = select_run_and_checkpoint(design, checkpoint_lock, array_index)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise OfficialCalibrationError("Exactly one scheduler-visible CUDA device is required")
    numerical_policy = configure_numerics(torch)
    if torch.cuda.get_device_name(0) != EXPECTED_DEVICE:
        raise OfficialCalibrationError("Calibration did not use the frozen H200 device")

    dataset = str(run["dataset"])
    data = load_stage_data(
        project_root,
        dataset,
        design["dataset_resources"][dataset]["dataset_manifest_sha256"],
    )
    payload = _load_checkpoint(project_root, run, locked)
    entity = payload["entity_embeddings"].to(device="cuda", dtype=torch.float32)
    relation = payload["relation_embeddings"].to(device="cuda", dtype=torch.float32)
    del payload
    parameters = run["selected_configuration"]["parameters"]
    transe_distance_norm = int(parameters.get("transe_distance_norm", 1))
    torch.cuda.reset_peak_memory_stats()
    scores, target_scores, candidate_counts = validation_score_matrix(
        str(run["model"]),
        entity,
        relation,
        data["train"],
        data["valid"],
        transe_distance_norm=transe_distance_norm,
    )
    temperature = fit_validation_temperature(scores, target_scores)
    if temperature["fitted_temperature"] < 0.001 or temperature["fitted_temperature"] > 1000.0:
        raise OfficialCalibrationError("Fitted temperature is outside frozen bounds")

    report = {
        "schema_version": 1,
        "status": "official_validation_calibration_completed_pending_temperature_freeze",
        "official_result": False,
        "validation_data_accessed": True,
        "test_data_accessed": False,
        "test_metrics_computed": False,
        "array_index": array_index,
        "run_id": run["run_id"],
        "paired_block_id": run["paired_block_id"],
        "dataset": dataset,
        "model": run["model"],
        "condition": run["condition"],
        "official_replicate": run["official_replicate"],
        "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "checkpoint": locked["checkpoint"],
        "validation": {
            "triples": int(len(data["valid"])),
            "head_queries": int(len(data["valid"])),
            "tail_queries": int(len(data["valid"])),
            "combined_queries": int(2 * len(data["valid"])),
            "candidate_vocabulary": int(data["entity_count"]),
            "minimum_filtered_candidate_count": int(candidate_counts.min().cpu()),
            "maximum_filtered_candidate_count": int(candidate_counts.max().cpu()),
            "filtering_splits": ["train", "validation"],
            "test_split_loaded": False,
        },
        "temperature": temperature,
        "official_design_sha256": OFFICIAL_DESIGN_SHA256,
        "official_checkpoint_lock_sha256": OFFICIAL_CHECKPOINT_LOCK_SHA256,
        "official_evaluation_lock_sha256": OFFICIAL_EVALUATION_LOCK_SHA256,
        "execution": {
            **execution,
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "numerical_policy": numerical_policy,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "git": git_state(project_root),
    }
    report_path = (
        project_root
        / "reports"
        / "official_calibration"
        / dataset
        / str(run["model"])
        / "replicate_{:02d}".format(int(run["official_replicate"]))
        / (str(run["condition"]) + ".json")
    )
    write_new_or_identical(report_path, canonical_json_bytes(report))
    del scores, target_scores, candidate_counts, entity, relation
    torch.cuda.empty_cache()
    gc.collect()
    return {**report, "report_path": str(report_path.relative_to(project_root))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--checkpoint-lock", type=Path, required=True)
    parser.add_argument("--evaluation-lock", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_official_calibration(
        design_path=args.design.resolve(),
        checkpoint_lock_path=args.checkpoint_lock.resolve(),
        evaluation_lock_path=args.evaluation_lock.resolve(),
        array_index=args.array_index,
        project_root=args.project_root.resolve(),
        scratch=args.scratch.resolve(),
    )
    print(
        "Official validation calibration completed for {}: T={}".format(
            report["run_id"], report["temperature"]["fitted_temperature"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
