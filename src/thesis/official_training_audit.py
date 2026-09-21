"""Audit the 90 official training outputs and freeze their checkpoint identities."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .official_design import (
    EXPECTED_OFFICIAL_DESIGN_SHA256,
    HPO_DESIGN_SHA256,
    canonical_json_bytes,
    git_state,
    read_json_object,
    require_scheduled_execution,
    sha256_bytes,
    sha256_file,
    validate_design_invariants,
    write_new_or_identical,
)


class OfficialTrainingAuditError(RuntimeError):
    """Raised when a frozen official training artifact fails audit."""


EXPECTED_RUNS = 90
EXPECTED_BLOCKS = 30
EXPECTED_SHUFFLES = 15
EXPECTED_CONDITIONS = {"random", "correct_text", "shuffled_text"}
EXPECTED_TRAINING_STATUS = (
    "official_training_completed_pending_calibration_and_test_evaluation"
)
EXPECTED_CHECKPOINT_STATUS = (
    "official_checkpoint_frozen_pending_calibration_and_test_evaluation"
)
EXPECTED_SHUFFLE_STATUS = "official_shuffle_materialised_pending_training"
EXPECTED_DEVICE = "NVIDIA H200 NVL"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
COMMON_STREAMS = (
    "relation_initialisation",
    "batch_order",
    "negative_sampling",
    "data_loader_workers",
)
EXPECTED_NUMERICAL_POLICY = {
    "parameter_dtype": "float32",
    "automatic_mixed_precision": False,
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms_enabled": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
}
EXPECTED_BATCH_LOSS_COLUMNS = [
    "total_loss",
    "ranking_loss",
    "unweighted_batch_local_l2",
]


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise OfficialTrainingAuditError(label + " differs from the frozen design")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise OfficialTrainingAuditError(label + " is not a SHA-256 digest")
    return value


def _recorded_path(
    project_root: Path,
    recorded: Any,
    expected_relative: str,
    label: str,
) -> Path:
    if not isinstance(recorded, str) or not recorded:
        raise OfficialTrainingAuditError(label + " path is missing")
    expected = (project_root / expected_relative).resolve()
    candidate = Path(recorded)
    observed = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    if observed != expected:
        raise OfficialTrainingAuditError(label + " path differs from the frozen design")
    try:
        observed.relative_to(project_root.resolve())
    except ValueError as error:
        raise OfficialTrainingAuditError(label + " escapes the project root") from error
    return observed


def _file_record(path: Path, project_root: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise OfficialTrainingAuditError("Required artifact is missing: " + str(path))
    return {
        "path": str(path.relative_to(project_root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _reject_outcome_fields(value: Any, location: str = "report") -> None:
    allowed = {
        "test_data_accessed",
        "validation_metrics_computed",
        "test_metrics_computed",
    }
    forbidden_fragments = (
        "mrr",
        "hits_at",
        "mean_rank",
        "temperature",
        "calibration",
        "expected_calibration_error",
        "negative_log_likelihood",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if key not in allowed and (
                lowered.startswith("test_")
                or lowered.startswith("validation_")
                or any(fragment in lowered for fragment in forbidden_fragments)
            ):
                raise OfficialTrainingAuditError(
                    "Outcome field appeared before checkpoint freeze: {}.{}".format(
                        location, key
                    )
                )
            _reject_outcome_fields(child, location + "." + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_outcome_fields(child, "{}[{}]".format(location, index))


def validate_report_metadata(
    report: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    preflight_sha256: str,
    dataset_manifest_sha256: str,
) -> None:
    _reject_outcome_fields(report)
    exact = {
        "schema_version": 1,
        "status": EXPECTED_TRAINING_STATUS,
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
        "official_design_sha256": EXPECTED_OFFICIAL_DESIGN_SHA256,
        "hpo_design_sha256": HPO_DESIGN_SHA256,
        "hardware_preflight_report_sha256": preflight_sha256,
        "numerical_policy": EXPECTED_NUMERICAL_POLICY,
    }
    for key, expected in exact.items():
        _require_equal(report.get(key), expected, "Official report field " + key)

    training = report.get("training")
    if not isinstance(training, dict):
        raise OfficialTrainingAuditError("Official training summary is missing")
    _require_equal(
        training.get("epochs"),
        run["selected_configuration"]["training_epochs"],
        "Official training epoch count",
    )
    if not isinstance(training.get("batches"), int) or training["batches"] <= 0:
        raise OfficialTrainingAuditError("Official training batch count is invalid")
    _require_sha256(training.get("training_sequence_sha256"), "Training sequence")
    _require_equal(
        training.get("batch_loss_columns"),
        EXPECTED_BATCH_LOSS_COLUMNS,
        "Batch loss columns",
    )
    _require_equal(
        training.get("maximum_negative_sampling_attempts"),
        1000,
        "Maximum negative-sampling attempts",
    )
    _require_equal(training.get("rng_algorithm"), "NumPy Generator(PCG64)", "RNG algorithm")
    _require_equal(training.get("data_loader_workers"), 0, "Data-loader worker count")
    _require_equal(
        training.get("globally_training_unseen_maximum_absolute_change"),
        0.0,
        "Training-unseen entity change",
    )
    epochs = training.get("epoch_history")
    if not isinstance(epochs, list) or [row.get("epoch") for row in epochs] != list(
        range(1, training["epochs"] + 1)
    ):
        raise OfficialTrainingAuditError("Epoch history is incomplete or non-contiguous")
    if sum(row.get("batches", -1) for row in epochs) != training["batches"]:
        raise OfficialTrainingAuditError("Epoch and aggregate batch counts differ")

    initialisation = report.get("initialisation")
    if not isinstance(initialisation, dict):
        raise OfficialTrainingAuditError("Initialisation record is missing")
    _require_equal(initialisation.get("dimension"), 384, "Embedding dimension")
    _require_equal(
        initialisation.get("relation_seed"),
        run["rng_streams"]["relation_initialisation"]["seed"],
        "Relation initialisation seed",
    )
    _require_sha256(initialisation.get("entity_matrix_raw_sha256"), "Entity initialisation")
    _require_sha256(initialisation.get("relation_matrix_raw_sha256"), "Relation initialisation")

    data = report.get("data")
    if not isinstance(data, dict):
        raise OfficialTrainingAuditError("Official training data summary is missing")
    _require_equal(
        data.get("dataset_manifest_sha256"),
        dataset_manifest_sha256,
        "Dataset manifest hash",
    )
    for key in ("entities", "relations", "training_triples", "training_seen_entities"):
        if not isinstance(data.get(key), int) or data[key] <= 0:
            raise OfficialTrainingAuditError("Official training data count is invalid: " + key)
    if not isinstance(data.get("globally_training_unseen_entities"), int) or data[
        "globally_training_unseen_entities"
    ] < 0:
        raise OfficialTrainingAuditError("Training-unseen entity count is invalid")

    environment = report.get("training_environment")
    if not isinstance(environment, dict) or environment.get("device_name") != EXPECTED_DEVICE:
        raise OfficialTrainingAuditError("Official run did not use the frozen H200 device")
    git = report.get("git")
    if not isinstance(git, dict) or git.get("tracked_worktree_clean") is not True:
        raise OfficialTrainingAuditError("Official run did not use a clean tracked worktree")
    if not COMMIT_RE.fullmatch(str(git.get("commit", ""))):
        raise OfficialTrainingAuditError("Official run Git commit is invalid")


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    run: Mapping[str, Any],
    report: Mapping[str, Any],
) -> Tuple[List[int], List[int]]:
    exact = {
        "schema_version": 1,
        "status": EXPECTED_CHECKPOINT_STATUS,
        "run_id": run["run_id"],
        "array_index": run["array_index"],
        "dataset": run["dataset"],
        "model": run["model"],
        "condition": run["condition"],
        "official_replicate": run["official_replicate"],
        "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "selected_configuration": run["selected_configuration"],
        "training_sequence_sha256": report["training"]["training_sequence_sha256"],
    }
    for key, expected in exact.items():
        _require_equal(payload.get(key), expected, "Checkpoint field " + key)
    entity = payload.get("entity_embeddings")
    relation = payload.get("relation_embeddings")
    if entity is None or relation is None:
        raise OfficialTrainingAuditError("Checkpoint embedding tensors are missing")
    entity_shape = list(entity.shape)
    relation_shape = list(relation.shape)
    expected_entity_shape = [report["data"]["entities"], 384]
    expected_relation_shape = [report["data"]["relations"], 384]
    if entity_shape != expected_entity_shape or relation_shape != expected_relation_shape:
        raise OfficialTrainingAuditError("Checkpoint embedding shape differs")
    if str(entity.dtype) != "torch.float32" or str(relation.dtype) != "torch.float32":
        raise OfficialTrainingAuditError("Checkpoint embedding dtype differs")
    return entity_shape, relation_shape


def paired_block_record(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(rows) != 3 or {row["condition"] for row in rows} != EXPECTED_CONDITIONS:
        raise OfficialTrainingAuditError("A paired block lacks the three frozen conditions")
    first = rows[0]
    for row in rows[1:]:
        _require_equal(
            row["selected_configuration"],
            first["selected_configuration"],
            "Paired selected configuration",
        )
        _require_equal(
            row["training_sequence_sha256"],
            first["training_sequence_sha256"],
            "Paired training sequence",
        )
        _require_equal(
            row["relation_matrix_raw_sha256"],
            first["relation_matrix_raw_sha256"],
            "Paired relation initialisation",
        )
        for namespace in COMMON_STREAMS:
            _require_equal(
                row["rng_streams"][namespace],
                first["rng_streams"][namespace],
                "Paired RNG stream " + namespace,
            )
    ordered = sorted(rows, key=lambda row: ("random", "correct_text", "shuffled_text").index(row["condition"]))
    return {
        "paired_block_id": first["paired_block_id"],
        "dataset": first["dataset"],
        "model": first["model"],
        "official_replicate": first["official_replicate"],
        "run_ids": [row["run_id"] for row in ordered],
        "training_sequence_sha256": first["training_sequence_sha256"],
        "relation_matrix_raw_sha256": first["relation_matrix_raw_sha256"],
        "common_rng_streams": {
            namespace: first["rng_streams"][namespace] for namespace in COMMON_STREAMS
        },
    }


def _audit_shuffle_artifacts(
    design: Mapping[str, Any],
    shuffle_report: Mapping[str, Any],
    project_root: Path,
) -> List[Dict[str, Any]]:
    import numpy as np
    from .official_shuffle import load_dataset_evidence, verify_permutation

    exact = {
        "status": "official_shuffles_materialised_pending_training",
        "official_result": False,
        "test_data_accessed": False,
        "official_design_sha256": EXPECTED_OFFICIAL_DESIGN_SHA256,
        "shuffle_count": EXPECTED_SHUFFLES,
    }
    for key, expected in exact.items():
        _require_equal(shuffle_report.get(key), expected, "Gate F report field " + key)
    report_rows = shuffle_report.get("shuffles")
    if not isinstance(report_rows, list) or len(report_rows) != EXPECTED_SHUFFLES:
        raise OfficialTrainingAuditError("Gate F report does not contain 15 shuffles")
    report_by_id = {row.get("shuffle_id"): row for row in report_rows}
    if len(report_by_id) != EXPECTED_SHUFFLES:
        raise OfficialTrainingAuditError("Gate F report contains duplicate shuffle IDs")
    _require_equal(
        shuffle_report.get("shuffle_manifest_sha256"),
        sha256_bytes(canonical_json_bytes(report_rows)),
        "Gate F shuffle record hash",
    )
    shuffle_git = shuffle_report.get("git")
    if (
        not isinstance(shuffle_git, dict)
        or shuffle_git.get("tracked_worktree_clean") is not True
        or not COMMIT_RE.fullmatch(str(shuffle_git.get("commit", "")))
    ):
        raise OfficialTrainingAuditError("Gate F did not use a clean committed worktree")

    records: List[Dict[str, Any]] = []
    for dataset in design["dataset_order"]:
        evidence_hashes, vector_hashes, _ = load_dataset_evidence(
            project_root, dataset, design["dataset_resources"][dataset]
        )
        for specification in [
            row for row in design["shuffle_artifacts"] if row["dataset"] == dataset
        ]:
            reported = report_by_id.get(specification["shuffle_id"])
            if not isinstance(reported, dict):
                raise OfficialTrainingAuditError("A frozen shuffle is absent from Gate F")
            manifest_path = project_root / specification["permutation_manifest"]
            artifact_path = project_root / specification["permutation_artifact"]
            manifest_record = _file_record(manifest_path, project_root)
            artifact_record = _file_record(artifact_path, project_root)
            manifest = read_json_object(manifest_path)
            manifest_exact = {
                "status": EXPECTED_SHUFFLE_STATUS,
                "official_result": False,
                "shuffle_id": specification["shuffle_id"],
                "dataset": dataset,
                "official_replicate": specification["official_replicate"],
                "seed_derivation": specification["derivation"],
                "official_design_sha256": EXPECTED_OFFICIAL_DESIGN_SHA256,
            }
            for key, expected in manifest_exact.items():
                _require_equal(manifest.get(key), expected, "Shuffle manifest field " + key)
            artifact_meta = manifest.get("permutation_artifact")
            if not isinstance(artifact_meta, dict):
                raise OfficialTrainingAuditError("Shuffle artifact metadata is missing")
            _require_equal(artifact_meta.get("path"), specification["permutation_artifact"], "Shuffle artifact path")
            _require_equal(artifact_meta.get("sha256"), artifact_record["sha256"], "Shuffle artifact hash")
            _require_equal(artifact_meta.get("size_bytes"), artifact_record["size_bytes"], "Shuffle artifact size")
            permutation = np.load(artifact_path, allow_pickle=False)
            if permutation.dtype != np.int64:
                raise OfficialTrainingAuditError("Shuffle permutation dtype differs")
            invariants = verify_permutation(permutation, evidence_hashes, vector_hashes)
            _require_equal(manifest.get("invariants"), invariants, "Shuffle invariants")
            _require_equal(reported.get("permutation_manifest_sha256"), manifest_record["sha256"], "Gate F manifest hash")
            _require_equal(reported.get("permutation_artifact_sha256"), artifact_record["sha256"], "Gate F artifact hash")
            records.append(
                {
                    "shuffle_id": specification["shuffle_id"],
                    "dataset": dataset,
                    "official_replicate": specification["official_replicate"],
                    "manifest": manifest_record,
                    "permutation_artifact": artifact_record,
                    "accepted_attempt": manifest["accepted_attempt"],
                    "invariants": invariants,
                }
            )
    if len(records) != EXPECTED_SHUFFLES:
        raise OfficialTrainingAuditError("Audited shuffle count differs from 15")
    return records


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise OfficialTrainingAuditError("Checkpoint root is not a dictionary")
    return payload


def audit_official_training(
    *,
    design_path: Path,
    hpo_design_path: Path,
    preflight_path: Path,
    shuffle_report_path: Path,
    project_root: Path,
    scratch: Path,
    output_lock: Path,
    output_report: Path,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    execution = require_scheduled_execution(scratch)
    if sha256_file(design_path) != EXPECTED_OFFICIAL_DESIGN_SHA256:
        raise OfficialTrainingAuditError("Official experiment design hash differs")
    if sha256_file(hpo_design_path) != HPO_DESIGN_SHA256:
        raise OfficialTrainingAuditError("Frozen HPO design hash differs")
    preflight_sha256 = sha256_file(preflight_path)
    _require_sha256(preflight_sha256, "Hardware preflight report")
    design = read_json_object(design_path)
    validate_design_invariants(design)
    shuffle_report = read_json_object(shuffle_report_path)
    shuffle_records = _audit_shuffle_artifacts(design, shuffle_report, project_root)

    run_records: List[Dict[str, Any]] = []
    paired_inputs: Dict[str, List[Dict[str, Any]]] = {}
    training_commits = set()
    for run in design["runs"]:
        report_path = project_root / run["expected_run_report"]
        report_file = _file_record(report_path, project_root)
        report = read_json_object(report_path)
        validate_report_metadata(
            report,
            run,
            preflight_sha256=preflight_sha256,
            dataset_manifest_sha256=design["dataset_resources"][run["dataset"]][
                "dataset_manifest_sha256"
            ],
        )
        training_commits.add(report["git"]["commit"])

        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict):
            raise OfficialTrainingAuditError("Official artifact record is missing")
        checkpoint_meta = artifacts.get("checkpoint")
        loss_meta = artifacts.get("batch_loss_history")
        if not isinstance(checkpoint_meta, dict) or not isinstance(loss_meta, dict):
            raise OfficialTrainingAuditError("Official checkpoint or loss artifact is missing")
        checkpoint_path = _recorded_path(
            project_root,
            checkpoint_meta.get("path"),
            run["expected_checkpoint"],
            "Checkpoint",
        )
        loss_relative = str(Path(run["expected_checkpoint"]).parent / "batch_loss_history.npy")
        loss_path = _recorded_path(
            project_root,
            loss_meta.get("path"),
            loss_relative,
            "Batch loss history",
        )
        checkpoint_file = _file_record(checkpoint_path, project_root)
        loss_file = _file_record(loss_path, project_root)
        for metadata, observed, label in (
            (checkpoint_meta, checkpoint_file, "Checkpoint"),
            (loss_meta, loss_file, "Batch loss history"),
        ):
            _require_equal(metadata.get("sha256"), observed["sha256"], label + " hash")
            _require_equal(metadata.get("size_bytes"), observed["size_bytes"], label + " size")

        loss_values = np.load(loss_path, allow_pickle=False)
        if loss_values.dtype != np.float64 or list(loss_values.shape) != [
            report["training"]["batches"],
            3,
        ]:
            raise OfficialTrainingAuditError("Batch loss history shape or dtype differs")
        if not np.isfinite(loss_values).all():
            raise OfficialTrainingAuditError("Batch loss history contains non-finite values")
        _require_equal(loss_meta.get("dtype"), "float64", "Batch loss metadata dtype")
        _require_equal(loss_meta.get("shape"), list(loss_values.shape), "Batch loss metadata shape")

        payload = _load_checkpoint(checkpoint_path)
        entity_shape, relation_shape = validate_checkpoint_payload(payload, run, report)
        if not torch.isfinite(payload["entity_embeddings"]).all() or not torch.isfinite(
            payload["relation_embeddings"]
        ).all():
            raise OfficialTrainingAuditError("Checkpoint contains non-finite embeddings")
        del payload, loss_values
        gc.collect()

        summary = {
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
            "training_sequence_sha256": report["training"]["training_sequence_sha256"],
            "entity_matrix_raw_sha256": report["initialisation"]["entity_matrix_raw_sha256"],
            "relation_matrix_raw_sha256": report["initialisation"]["relation_matrix_raw_sha256"],
            "report": report_file,
            "checkpoint": {
                **checkpoint_file,
                "entity_shape": entity_shape,
                "relation_shape": relation_shape,
                "dtype": "float32",
            },
            "batch_loss_history": {
                **loss_file,
                "dtype": "float64",
                "shape": [report["training"]["batches"], 3],
            },
        }
        run_records.append(summary)
        paired_inputs.setdefault(run["paired_block_id"], []).append(summary)

    if len(run_records) != EXPECTED_RUNS or len({row["run_id"] for row in run_records}) != EXPECTED_RUNS:
        raise OfficialTrainingAuditError("Official audited run count differs from 90")
    if len(training_commits) != 1:
        raise OfficialTrainingAuditError("Official runs do not share one training-code commit")
    training_commit = next(iter(training_commits))
    shuffle_git = shuffle_report.get("git")
    if not isinstance(shuffle_git, dict) or shuffle_git.get("commit") != training_commit:
        raise OfficialTrainingAuditError("Gate F and official training commits differ")
    paired_records = [
        paired_block_record(paired_inputs[block["paired_block_id"]])
        for block in design["paired_blocks"]
    ]
    if len(paired_records) != EXPECTED_BLOCKS:
        raise OfficialTrainingAuditError("Official paired-block count differs from 30")

    lock = {
        "schema_version": 1,
        "status": "official_checkpoints_frozen_pending_calibration_and_test_evaluation",
        "official_result": False,
        "test_data_accessed": False,
        "validation_metrics_computed": False,
        "test_metrics_computed": False,
        "official_design": {
            "path": str(design_path.relative_to(project_root)),
            "sha256": EXPECTED_OFFICIAL_DESIGN_SHA256,
        },
        "inputs": {
            "hpo_design_sha256": HPO_DESIGN_SHA256,
            "hardware_preflight_report_sha256": preflight_sha256,
            "gate_f_shuffle_report_sha256": sha256_file(shuffle_report_path),
        },
        "training_git_commit": training_commit,
        "counts": {
            "official_runs": len(run_records),
            "paired_blocks": len(paired_records),
            "shuffle_artifacts": len(shuffle_records),
        },
        "runs_sha256": sha256_bytes(canonical_json_bytes(run_records)),
        "paired_blocks_sha256": sha256_bytes(canonical_json_bytes(paired_records)),
        "shuffles_sha256": sha256_bytes(canonical_json_bytes(shuffle_records)),
        "runs": run_records,
        "paired_blocks": paired_records,
        "shuffles": shuffle_records,
    }
    lock_bytes = canonical_json_bytes(lock)
    write_new_or_identical(output_lock, lock_bytes)
    audit_git = git_state(project_root)
    report = {
        "schema_version": 1,
        "status": "official_training_audited_checkpoints_frozen_pending_calibration_and_test_evaluation",
        "official_result": False,
        "test_data_accessed": False,
        "validation_metrics_computed": False,
        "test_metrics_computed": False,
        "counts": lock["counts"],
        "official_checkpoint_lock": {
            "path": str(output_lock.relative_to(project_root)),
            "sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "size_bytes": len(lock_bytes),
        },
        "training_git_commit": training_commit,
        "execution": execution,
        "git": audit_git,
    }
    write_new_or_identical(output_report, canonical_json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--hpo-design", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--shuffle-report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_official_training(
        design_path=args.design.resolve(),
        hpo_design_path=args.hpo_design.resolve(),
        preflight_path=args.preflight_report.resolve(),
        shuffle_report_path=args.shuffle_report.resolve(),
        project_root=args.project_root.resolve(),
        scratch=args.scratch.resolve(),
        output_lock=args.output_lock.resolve(),
        output_report=args.output_report.resolve(),
    )
    print(
        "Official training audit recorded in {}".format(
            args.output_report.resolve()
        )
    )
    print(
        "Official checkpoint lock written to {}".format(
            args.output_lock.resolve()
        )
    )
    print(report["official_checkpoint_lock"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
