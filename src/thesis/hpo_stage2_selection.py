"""Audit HPO Stage 2 trajectories and freeze the Stage 3 entrants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .hpo_stage1 import HPO_DESIGN_SHA256, validate_design
from .hpo_stage2 import (
    EPOCHS,
    EXPECTED_ARRAY_TASKS,
    EXPECTED_CANDIDATES_PER_UNIT,
    EXPECTED_REPLICATES,
    EXPECTED_UNITS,
    NAMESPACES,
    PROTOCOL_SHA256,
    STAGE2_DESIGN_SHA256,
    select_array_task,
    stream_seed,
    validate_stage2_design,
)
from .kg_core import ranking_metrics
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object


class HPOStage2SelectionError(RuntimeError):
    """Raised when Stage 2 cannot be promoted mechanically."""


REPORT_NAME_RE = re.compile(r"^hpo_stage2_([0-9]+)_([0-9]+)\.json$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ATTEMPT_LOCK_SHA256 = "cce80551e4997c4bc573eb5f82d551fecba8f3450094170458962558dd09f157"
COMPLETED_STATUS = "hpo_stage_2_run_completed"
FAILED_STATUS = "hpo_stage_2_run_failed"
STAGE3_EPOCHS = tuple(range(10, 201, 10))


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HPOStage2SelectionError(label + " is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise HPOStage2SelectionError(label + " is non-finite")
    return number


def _finite_text_number(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise HPOStage2SelectionError(label + " is not numeric") from error
    if not math.isfinite(number):
        raise HPOStage2SelectionError(label + " is non-finite")
    return number


def _full_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise HPOStage2SelectionError(label + " is not a full SHA-256 digest")
    return value


def _resolve_under(project_root: Path, allowed_root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise HPOStage2SelectionError(label + " path is invalid")
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(allowed_root.resolve())
    except ValueError as error:
        raise HPOStage2SelectionError(label + " escapes its allowed root") from error
    if not resolved.is_file():
        raise HPOStage2SelectionError(label + " is missing")
    return resolved


def _project_relative(path: Path, project_root: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError as error:
        raise HPOStage2SelectionError(label + " escapes the project root") from error


def validate_artifact(
    artifact: Any,
    *,
    project_root: Path,
    artifact_root: Path,
    label: str,
) -> Tuple[Path, str]:
    if not isinstance(artifact, dict):
        raise HPOStage2SelectionError(label + " metadata is invalid")
    path = _resolve_under(project_root, artifact_root, artifact.get("path"), label)
    expected_size = artifact.get("size_bytes")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
        or path.stat().st_size != expected_size
    ):
        raise HPOStage2SelectionError(label + " size differs")
    expected_digest = _full_sha256(artifact.get("sha256"), label + " hash")
    if sha256_file(path) != expected_digest:
        raise HPOStage2SelectionError(label + " hash differs")
    return path, expected_digest


def _metric_block(block: Any, expected_queries: int, label: str) -> None:
    if not isinstance(block, dict) or block.get("queries") != expected_queries:
        raise HPOStage2SelectionError(label + " query count differs")
    for key in ("mrr", "hits_at_1", "hits_at_3", "hits_at_10"):
        value = _finite_number(block.get(key), label + " " + key)
        if not 0.0 <= value <= 1.0:
            raise HPOStage2SelectionError(label + " metric is outside [0,1]")


def _metrics_match(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key in ("queries", "mrr", "hits_at_1", "hits_at_3", "hits_at_10"):
        if observed.get(key) != expected.get(key):
            raise HPOStage2SelectionError(label + " differs for " + key)


def audit_rank_table(
    path: Path,
    *,
    validation_triples: int,
    validation: Mapping[str, Any],
) -> None:
    expected_header = [
        "validation_row",
        "direction",
        "head",
        "relation",
        "tail",
        "target_entity",
        "rank",
    ]
    head_ranks: List[float] = []
    tail_ranks: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        try:
            header = next(reader)
        except StopIteration as error:
            raise HPOStage2SelectionError("Validation rank table is empty") from error
        if header != expected_header:
            raise HPOStage2SelectionError("Validation rank-table header differs")
        rows = list(reader)
    if len(rows) != 2 * validation_triples:
        raise HPOStage2SelectionError("Validation rank-table row count differs")
    for validation_row in range(validation_triples):
        pair = rows[2 * validation_row : 2 * validation_row + 2]
        if len(pair) != 2 or any(len(row) != len(expected_header) for row in pair):
            raise HPOStage2SelectionError("Validation rank-table shape differs")
        if [row[1] for row in pair] != ["head", "tail"]:
            raise HPOStage2SelectionError("Validation rank-table direction order differs")
        if any(int(row[0]) != validation_row for row in pair):
            raise HPOStage2SelectionError("Validation rank-table index differs")
        triples = [tuple(map(int, row[2:5])) for row in pair]
        if triples[0] != triples[1]:
            raise HPOStage2SelectionError("Head/tail rows refer to different triples")
        head, _, tail = triples[0]
        if int(pair[0][5]) != head or int(pair[1][5]) != tail:
            raise HPOStage2SelectionError("Validation target entity differs")
        ranks = [_finite_text_number(row[6], "validation rank") for row in pair]
        if any(rank < 1.0 for rank in ranks):
            raise HPOStage2SelectionError("Validation rank is below one")
        head_ranks.append(ranks[0])
        tail_ranks.append(ranks[1])
    _metrics_match(validation["head"], ranking_metrics(head_ranks), "head metrics")
    _metrics_match(validation["tail"], ranking_metrics(tail_ranks), "tail metrics")
    _metrics_match(
        validation["combined"],
        ranking_metrics(head_ranks + tail_ranks),
        "combined metrics",
    )


def candidate_mean(records: Sequence[Mapping[str, Any]]) -> float:
    replicates = [int(record["replicate"]) for record in records]
    if sorted(replicates) != list(EXPECTED_REPLICATES):
        raise HPOStage2SelectionError("A candidate does not contain replicates 1 and 2")
    identifiers = {str(record["candidate_id"]) for record in records}
    if len(identifiers) != 1:
        raise HPOStage2SelectionError("Candidate replicate identifiers differ")
    values = [
        _finite_number(record["validation_mrr"], "Stage 2 validation MRR")
        for record in records
    ]
    return math.fsum(values) / len(values)


def selection_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Protocol ordering: mean MRR desc, K asc, batch asc, ID lexicographic."""

    parameters = record["parameters"]
    return (
        -float(record["mean_validation_mrr"]),
        int(parameters["negatives_per_positive"]),
        int(parameters["batch_size"]),
        str(record["candidate_id"]),
    )


def select_advancing(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if len(records) < 2:
        raise HPOStage2SelectionError("A tuning unit has fewer than two valid candidates")
    ordered = sorted(records, key=selection_key)
    return [
        {**dict(record), "stage_2_rank": rank}
        for rank, record in enumerate(ordered[:2], 1)
    ]


def discover_reports(reports_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in reports_dir.iterdir()
            if path.is_file() and REPORT_NAME_RE.fullmatch(path.name)
        ],
        key=lambda path: path.name,
    )


def _attempt_range(
    item: Any,
    *,
    expected_tasks: int,
    require_reason: bool,
) -> Tuple[str, int, int]:
    if not isinstance(item, dict):
        raise HPOStage2SelectionError("A Stage 2 attempt range is invalid")
    job_id = item.get("array_job_id")
    first = item.get("first_array_index")
    last = item.get("last_array_index")
    reason = item.get("reason")
    if (
        not isinstance(job_id, str)
        or not job_id.isdigit()
        or not isinstance(first, int)
        or isinstance(first, bool)
        or not isinstance(last, int)
        or isinstance(last, bool)
        or not 0 <= first <= last < expected_tasks
        or (require_reason and (not isinstance(reason, str) or not reason))
    ):
        raise HPOStage2SelectionError("A Stage 2 attempt range is invalid")
    return job_id, first, last


def validate_attempt_lock(
    attempt_lock: Mapping[str, Any],
) -> Tuple[Dict[int, str], Dict[str, Mapping[str, Any]]]:
    exact = {
        "schema_version": 1,
        "status": "hpo_stage_2_attempts_frozen_pending_selection",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_hpo_design_sha256": HPO_DESIGN_SHA256,
        "stage_2_design_sha256": STAGE2_DESIGN_SHA256,
    }
    for key, value in exact.items():
        if attempt_lock.get(key) != value:
            raise HPOStage2SelectionError("Stage 2 attempt lock differs in " + key)
    ranges = attempt_lock.get("canonical_attempt_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise HPOStage2SelectionError("Canonical Stage 2 ranges are missing")
    canonical: Dict[int, str] = {}
    for item in ranges:
        job_id, first, last = _attempt_range(
            item, expected_tasks=EXPECTED_ARRAY_TASKS, require_reason=False
        )
        for index in range(first, last + 1):
            if index in canonical:
                raise HPOStage2SelectionError("Canonical Stage 2 ranges overlap")
            canonical[index] = "hpo_stage2_{}_{}.json".format(job_id, index)
    if set(canonical) != set(range(EXPECTED_ARRAY_TASKS)):
        raise HPOStage2SelectionError("Canonical Stage 2 ranges do not cover 0..47")
    excluded_ranges = attempt_lock.get("excluded_completed_attempt_ranges")
    if not isinstance(excluded_ranges, list):
        raise HPOStage2SelectionError("Excluded Stage 2 ranges are invalid")
    excluded: Dict[str, Mapping[str, Any]] = {}
    for item in excluded_ranges:
        job_id, first, last = _attempt_range(
            item, expected_tasks=EXPECTED_ARRAY_TASKS, require_reason=True
        )
        for index in range(first, last + 1):
            name = "hpo_stage2_{}_{}.json".format(job_id, index)
            if name in excluded or name in canonical.values():
                raise HPOStage2SelectionError("Stage 2 attempt identities overlap")
            excluded[name] = {**item, "array_index": index}
    cancelled = attempt_lock.get("cancelled_duplicate_submission")
    if not isinstance(cancelled, dict):
        raise HPOStage2SelectionError("Cancelled duplicate submission is missing")
    if cancelled.get("array_job_id") != "440859":
        raise HPOStage2SelectionError("Cancelled duplicate job identity differs")
    expected_excluded_names = {
        "hpo_stage2_440859_{}.json".format(index) for index in range(1, 8)
    }
    if set(excluded) != expected_excluded_names:
        raise HPOStage2SelectionError("Completed duplicate report identities differ")
    if (
        cancelled.get("requested_first_array_index") != 1
        or cancelled.get("requested_last_array_index") != 47
        or cancelled.get("maximum_concurrent_tasks") != 2
    ):
        raise HPOStage2SelectionError("Cancelled duplicate request shape differs")
    if cancelled.get("completed_report_indices") != list(range(1, 8)):
        raise HPOStage2SelectionError("Completed duplicate indices differ")
    if cancelled.get("cancelled_running_indices_without_reports") != [8, 9]:
        raise HPOStage2SelectionError("Cancelled running duplicate indices differ")
    if cancelled.get("cancelled_pending_ranges_without_reports") != [
        {"first_array_index": 10, "last_array_index": 47}
    ]:
        raise HPOStage2SelectionError("Cancelled pending duplicate indices differ")
    if not isinstance(cancelled.get("reason"), str) or not cancelled["reason"]:
        raise HPOStage2SelectionError("Cancelled duplicate reason is missing")
    if not isinstance(attempt_lock.get("selection_rule"), str):
        raise HPOStage2SelectionError("Stage 2 selection rule is missing")
    return canonical, excluded


def _validate_completed_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    stage2_design: Mapping[str, Any],
    source_design: Mapping[str, Any],
    project_root: Path,
    artifact_root: Path,
) -> Dict[str, Any]:
    array_index = int(report["array_index"])
    unit, candidate, replicate = select_array_task(
        stage2_design, source_design, array_index
    )
    exact = {
        "schema_version": 1,
        "status": COMPLETED_STATUS,
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": 2,
        "replicate": replicate,
        "dataset": unit["dataset"],
        "model": unit["model"],
        "unit_id": unit["unit_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_configuration_sha256": candidate[
            "candidate_configuration_sha256"
        ],
        "parameters": candidate["parameters"],
        "epochs": EPOCHS,
        "stage_1_checkpoint_loaded": False,
        "stage_2_design_sha256": STAGE2_DESIGN_SHA256,
        "source_hpo_design_sha256": HPO_DESIGN_SHA256,
        "hardware_preflight_report_sha256": source_design[
            "hardware_preflight_report_sha256"
        ],
        "protocol_sha256": PROTOCOL_SHA256,
    }
    for key, value in exact.items():
        if report.get(key) != value:
            raise HPOStage2SelectionError(
                "{} differs in {}".format(report_path.name, key)
            )
    expected_provenance = {
        "rank": candidate["stage_1_rank"],
        "validation_mrr": candidate["stage_1_validation_mrr"],
        "report_sha256": candidate["stage_1_report_sha256"],
    }
    if report.get("stage_1_provenance") != expected_provenance:
        raise HPOStage2SelectionError(report_path.name + " Stage 1 provenance differs")
    expected_seeds = {
        namespace: stream_seed(unit, replicate, namespace)
        for namespace in NAMESPACES
    }
    if report.get("seeds") != expected_seeds:
        raise HPOStage2SelectionError(report_path.name + " has different tuning seeds")
    data = report.get("data")
    if not isinstance(data, dict):
        raise HPOStage2SelectionError(report_path.name + " data audit is missing")
    validation_triples = data.get("validation_triples")
    training_triples = data.get("training_triples")
    if (
        not isinstance(validation_triples, int)
        or validation_triples <= 0
        or not isinstance(training_triples, int)
        or training_triples <= 0
    ):
        raise HPOStage2SelectionError(report_path.name + " split counts are invalid")
    training = report.get("training")
    if not isinstance(training, dict):
        raise HPOStage2SelectionError(report_path.name + " training audit is missing")
    batch_size = int(candidate["parameters"]["batch_size"])
    batches_per_epoch = (training_triples + batch_size - 1) // batch_size
    expected_batches = EPOCHS * batches_per_epoch
    if training.get("batches") != expected_batches:
        raise HPOStage2SelectionError(report_path.name + " batch count differs")
    history = training.get("epoch_history")
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise HPOStage2SelectionError(report_path.name + " epoch history differs")
    for epoch, item in enumerate(history, 1):
        if item.get("epoch") != epoch or item.get("batches") != batches_per_epoch:
            raise HPOStage2SelectionError(report_path.name + " epoch audit differs")
        minimum = _finite_number(item.get("minimum_total_loss"), "minimum loss")
        mean = _finite_number(item.get("mean_total_loss"), "mean loss")
        maximum = _finite_number(item.get("maximum_total_loss"), "maximum loss")
        if not minimum <= mean <= maximum:
            raise HPOStage2SelectionError(report_path.name + " loss ordering differs")
        if _finite_number(item.get("runtime_seconds"), "epoch runtime") <= 0:
            raise HPOStage2SelectionError(report_path.name + " epoch runtime is invalid")
    if training.get("globally_training_unseen_maximum_absolute_change") != 0.0:
        raise HPOStage2SelectionError(report_path.name + " changed an unseen entity")
    _full_sha256(training.get("training_sequence_sha256"), "training sequence hash")
    validation = report.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("filter_scope") != "train_plus_validation"
    ):
        raise HPOStage2SelectionError(report_path.name + " validation scope differs")
    _metric_block(validation.get("head"), validation_triples, "head validation")
    _metric_block(validation.get("tail"), validation_triples, "tail validation")
    _metric_block(
        validation.get("combined"), 2 * validation_triples, "combined validation"
    )
    initialisation = report.get("initialisation")
    if (
        not isinstance(initialisation, dict)
        or initialisation.get("dimension") != 384
        or initialisation.get("source") != "regenerated_from_frozen_tuning_stream"
    ):
        raise HPOStage2SelectionError(report_path.name + " initialisation differs")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HPOStage2SelectionError(report_path.name + " artifacts are missing")
    artifact_hashes = {}
    for key in ("checkpoint", "batch_loss_history", "validation_ranks"):
        path, digest = validate_artifact(
            artifacts.get(key),
            project_root=project_root,
            artifact_root=artifact_root,
            label=report_path.name + " " + key,
        )
        artifact_hashes[key] = digest
        if key == "validation_ranks":
            if artifacts[key].get("rows") != 2 * validation_triples:
                raise HPOStage2SelectionError(
                    report_path.name + " rank row count differs"
                )
            audit_rank_table(
                path,
                validation_triples=validation_triples,
                validation=validation,
            )
        elif key == "batch_loss_history":
            if (
                artifacts[key].get("shape") != [expected_batches, 3]
                or artifacts[key].get("dtype") != "float64"
            ):
                raise HPOStage2SelectionError(
                    report_path.name + " loss artifact shape differs"
                )
    for key in ("entity_artifact", "relation_artifact"):
        _, digest = validate_artifact(
            initialisation.get(key),
            project_root=project_root,
            artifact_root=artifact_root,
            label=report_path.name + " " + key,
        )
        artifact_hashes[key] = digest
    if (
        initialisation["entity_artifact"].get("shape") != [data.get("entities"), 384]
        or initialisation["relation_artifact"].get("shape")
        != [data.get("relations"), 384]
        or initialisation["entity_artifact"].get("dtype") != "float32"
        or initialisation["relation_artifact"].get("dtype") != "float32"
    ):
        raise HPOStage2SelectionError(
            report_path.name + " initialisation artifact shape differs"
        )
    entity_raw_sha = _full_sha256(
        initialisation.get("entity_matrix_raw_sha256"),
        "entity initialisation raw hash",
    )
    relation_raw_sha = _full_sha256(
        initialisation.get("relation_matrix_raw_sha256"),
        "relation initialisation raw hash",
    )
    execution = report.get("execution")
    if not isinstance(execution, dict) or execution.get("execution_context") != "slurm":
        raise HPOStage2SelectionError(report_path.name + " was not scheduler-executed")
    if str(execution.get("slurm_array_task_id")) != str(array_index):
        raise HPOStage2SelectionError(report_path.name + " execution task differs")
    environment = report.get("training_environment")
    if not isinstance(environment, dict) or environment.get("device_name") != "NVIDIA H200 NVL":
        raise HPOStage2SelectionError(report_path.name + " environment differs")
    source_git = report.get("git")
    if not isinstance(source_git, dict) or source_git.get("tracked_worktree_clean") is not True:
        raise HPOStage2SelectionError(report_path.name + " source worktree was not clean")
    source_commit = source_git.get("commit")
    if not isinstance(source_commit, str) or not COMMIT_RE.fullmatch(source_commit):
        raise HPOStage2SelectionError(report_path.name + " source commit is invalid")
    return {
        "array_index": array_index,
        "replicate": replicate,
        "dataset": unit["dataset"],
        "model": unit["model"],
        "unit_id": unit["unit_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_configuration_sha256": candidate[
            "candidate_configuration_sha256"
        ],
        "parameters": candidate["parameters"],
        "validation_mrr": validation["combined"]["mrr"],
        "report_path": _project_relative(report_path, project_root, "Stage 2 report"),
        "report_sha256": sha256_file(report_path),
        "artifact_sha256": artifact_hashes,
        "training_environment": environment,
        "source_git_commit": source_commit,
        "initialisation_identity": {
            "entity_matrix_raw_sha256": entity_raw_sha,
            "relation_matrix_raw_sha256": relation_raw_sha,
            "entity_artifact_sha256": artifact_hashes["entity_artifact"],
            "relation_artifact_sha256": artifact_hashes["relation_artifact"],
        },
    }


def audit_and_select(
    stage2_design_path: Path,
    source_design_path: Path,
    protocol_path: Path,
    attempt_lock_path: Path,
    reports_dir: Path,
    *,
    project_root: Path,
    artifact_root: Path,
    scratch: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    execution = require_scheduled_execution(scratch.resolve())
    if sha256_file(stage2_design_path) != STAGE2_DESIGN_SHA256:
        raise HPOStage2SelectionError("Frozen Stage 2 design hash differs")
    if sha256_file(source_design_path) != HPO_DESIGN_SHA256:
        raise HPOStage2SelectionError("Frozen source HPO design hash differs")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise HPOStage2SelectionError("Frozen protocol hash differs")
    if sha256_file(attempt_lock_path) != ATTEMPT_LOCK_SHA256:
        raise HPOStage2SelectionError("Frozen Stage 2 attempt-lock hash differs")
    stage2_design = read_json_object(stage2_design_path)
    source_design = read_json_object(source_design_path)
    validate_design(source_design)
    validate_stage2_design(stage2_design, source_design)
    attempt_lock = read_json_object(attempt_lock_path)
    canonical_names, excluded_specs = validate_attempt_lock(attempt_lock)
    _project_relative(reports_dir, project_root, "Stage 2 reports directory")
    _project_relative(artifact_root, project_root, "Stage 2 artifact root")
    paths = discover_reports(reports_dir)
    expected_names = set(canonical_names.values()) | set(excluded_specs)
    observed_names = {path.name for path in paths}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise HPOStage2SelectionError(
            "Stage 2 attempt set differs; missing={!r}, unexpected={!r}".format(
                missing, unexpected
            )
        )
    by_index: Dict[int, Path] = {}
    for index, filename in canonical_names.items():
        path = reports_dir / filename
        report = read_json_object(path)
        report_index = report.get("array_index")
        match = REPORT_NAME_RE.fullmatch(path.name)
        if report_index != index or match is None or int(match.group(2)) != index:
            raise HPOStage2SelectionError(path.name + " has an invalid array index")
        job_id_field = (
            "array_job_id"
            if report.get("status") == COMPLETED_STATUS
            else "slurm_array_job_id"
        )
        if str(report.get(job_id_field)) != match.group(1):
            raise HPOStage2SelectionError(path.name + " disagrees with its array job ID")
        by_index[index] = path
    if set(by_index) != set(range(EXPECTED_ARRAY_TASKS)):
        raise HPOStage2SelectionError("Stage 2 array-index coverage differs")
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for index in range(EXPECTED_ARRAY_TASKS):
        path = by_index[index]
        report = read_json_object(path)
        unit, candidate, replicate = select_array_task(
            stage2_design, source_design, index
        )
        if report.get("status") == COMPLETED_STATUS:
            valid.append(
                _validate_completed_report(
                    report,
                    report_path=path,
                    stage2_design=stage2_design,
                    source_design=source_design,
                    project_root=project_root,
                    artifact_root=artifact_root,
                )
            )
        elif report.get("status") == FAILED_STATUS:
            invalid.append(
                {
                    "array_index": index,
                    "replicate": replicate,
                    "unit_id": unit["unit_id"],
                    "candidate_id": candidate["candidate_id"],
                    "error_type": report.get("error_type"),
                    "error": report.get("error"),
                    "report_path": _project_relative(path, project_root, "failed report"),
                    "report_sha256": sha256_file(path),
                }
            )
        else:
            raise HPOStage2SelectionError(path.name + " has an unknown status")
    excluded_attempts = []
    for filename, specification in sorted(excluded_specs.items()):
        path = reports_dir / filename
        report = read_json_object(path)
        index = int(specification["array_index"])
        match = REPORT_NAME_RE.fullmatch(filename)
        if (
            report.get("array_index") != index
            or match is None
            or str(report.get("array_job_id")) != match.group(1)
        ):
            raise HPOStage2SelectionError(filename + " identity differs")
        audited = _validate_completed_report(
            report,
            report_path=path,
            stage2_design=stage2_design,
            source_design=source_design,
            project_root=project_root,
            artifact_root=artifact_root,
        )
        excluded_attempts.append(
            {
                "array_job_id": specification["array_job_id"],
                "array_index": index,
                "dataset": audited["dataset"],
                "model": audited["model"],
                "candidate_id": audited["candidate_id"],
                "replicate": audited["replicate"],
                "status": COMPLETED_STATUS,
                "reason": specification["reason"],
                "report_path": audited["report_path"],
                "report_sha256": audited["report_sha256"],
                "used_for_selection": False,
            }
        )
    if not valid:
        raise HPOStage2SelectionError("Stage 2 contains no valid canonical runs")
    environments = {canonical_json_bytes(item["training_environment"]) for item in valid}
    commits = {item["source_git_commit"] for item in valid}
    if len(environments) != 1 or len(commits) != 1:
        raise HPOStage2SelectionError("Stage 2 environment/source commit is inconsistent")
    identities_by_stream: Dict[Tuple[str, int], set] = defaultdict(set)
    for item in valid:
        identities_by_stream[(item["unit_id"], item["replicate"])].add(
            canonical_json_bytes(item["initialisation_identity"])
        )
    if any(len(values) != 1 for values in identities_by_stream.values()):
        raise HPOStage2SelectionError("Stage 2 paired initialisation differs")
    valid_by_candidate: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    invalid_by_candidate: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in valid:
        valid_by_candidate[(item["unit_id"], item["candidate_id"])].append(item)
    for item in invalid:
        invalid_by_candidate[(item["unit_id"], item["candidate_id"])].append(item)
    units = []
    stage3_units = []
    for design_unit in stage2_design["units"]:
        unit_id = str(design_unit["unit_id"])
        valid_candidates = []
        invalid_candidates = []
        for candidate in design_unit["advancing_candidates"]:
            candidate_id = str(candidate["candidate_id"])
            key = (unit_id, candidate_id)
            runs = sorted(valid_by_candidate[key], key=lambda item: item["replicate"])
            failed = sorted(
                invalid_by_candidate[key], key=lambda item: item["replicate"]
            )
            if failed or [run["replicate"] for run in runs] != list(EXPECTED_REPLICATES):
                invalid_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "valid_replicates": [run["replicate"] for run in runs],
                        "failed_runs": failed,
                        "reason": "A required Stage 2 seed trajectory is invalid or missing.",
                    }
                )
                continue
            mean_mrr = candidate_mean(runs)
            valid_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_configuration_sha256": candidate[
                        "candidate_configuration_sha256"
                    ],
                    "parameters": candidate["parameters"],
                    "stage_1_rank": candidate["stage_1_rank"],
                    "replicate_runs": [
                        {
                            "replicate": run["replicate"],
                            "array_index": run["array_index"],
                            "validation_mrr": run["validation_mrr"],
                            "report_path": run["report_path"],
                            "report_sha256": run["report_sha256"],
                            "artifact_sha256": run["artifact_sha256"],
                        }
                        for run in runs
                    ],
                    "mean_validation_mrr": mean_mrr,
                }
            )
        ordered = sorted(valid_candidates, key=selection_key)
        ranking = [
            {**item, "stage_2_rank": rank}
            for rank, item in enumerate(ordered, 1)
        ]
        advancing = select_advancing(valid_candidates)
        advancing_compact = [
            {
                "candidate_id": item["candidate_id"],
                "candidate_configuration_sha256": item[
                    "candidate_configuration_sha256"
                ],
                "parameters": item["parameters"],
                "stage_1_rank": item["stage_1_rank"],
                "stage_2_rank": item["stage_2_rank"],
                "stage_2_mean_validation_mrr": item["mean_validation_mrr"],
                "stage_2_replicate_validation_mrr": {
                    str(run["replicate"]): run["validation_mrr"]
                    for run in item["replicate_runs"]
                },
                "stage_2_report_sha256_by_replicate": {
                    str(run["replicate"]): run["report_sha256"]
                    for run in item["replicate_runs"]
                },
            }
            for item in advancing
        ]
        units.append(
            {
                "dataset": design_unit["dataset"],
                "model": design_unit["model"],
                "unit_id": unit_id,
                "valid_candidates": len(valid_candidates),
                "invalid_candidates": invalid_candidates,
                "ranking": ranking,
                "advancing_candidate_ids": [
                    item["candidate_id"] for item in advancing
                ],
            }
        )
        stage3_units.append(
            {
                "dataset": design_unit["dataset"],
                "model": design_unit["model"],
                "unit_id": unit_id,
                "tuning_stream_derivations": design_unit[
                    "tuning_stream_derivations"
                ],
                "advancing_candidates": advancing_compact,
            }
        )
    if len(units) != EXPECTED_UNITS or any(
        unit["valid_candidates"] + len(unit["invalid_candidates"])
        != EXPECTED_CANDIDATES_PER_UNIT
        for unit in units
    ):
        raise HPOStage2SelectionError("Stage 2 unit coverage differs")
    report_manifest = [
        {
            "array_index": index,
            "path": _project_relative(by_index[index], project_root, "Stage 2 report"),
            "sha256": sha256_file(by_index[index]),
        }
        for index in range(EXPECTED_ARRAY_TASKS)
    ]
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(report_manifest)).hexdigest()
    source_commit = next(iter(commits))
    stage3_design = {
        "schema_version": 1,
        "status": "hpo_stage_3_design_frozen_pending_runs",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": 3,
        "maximum_epochs": 200,
        "tuning_replicates": [1, 2, 3],
        "validation_epochs": list(STAGE3_EPOCHS),
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
        "protocol_sha256": PROTOCOL_SHA256,
        "source_hpo_design_sha256": HPO_DESIGN_SHA256,
        "source_stage_2_design_sha256": STAGE2_DESIGN_SHA256,
        "source_stage_2_attempt_lock_sha256": ATTEMPT_LOCK_SHA256,
        "source_stage_2_report_manifest_sha256": manifest_sha256,
        "source_stage_2_runner_git_commit": source_commit,
        "fixed_training": stage2_design["fixed_training"],
        "units": stage3_units,
    }
    audit_report = {
        "schema_version": 1,
        "status": "hpo_stage_2_audited_and_selected_pending_stage_3",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "execution": execution,
        "git": git_state(project_root),
        "protocol_sha256": PROTOCOL_SHA256,
        "source_hpo_design_sha256": HPO_DESIGN_SHA256,
        "stage_2_design_sha256": STAGE2_DESIGN_SHA256,
        "stage_2_attempt_lock_sha256": ATTEMPT_LOCK_SHA256,
        "source_stage_2_report_manifest": report_manifest,
        "source_stage_2_report_manifest_sha256": manifest_sha256,
        "valid_runs": len(valid),
        "invalid_runs": invalid,
        "excluded_completed_attempts": excluded_attempts,
        "cancelled_duplicate_submission": attempt_lock[
            "cancelled_duplicate_submission"
        ],
        "training_environment": valid[0]["training_environment"],
        "source_stage_2_runner_git_commit": source_commit,
        "units": units,
    }
    return audit_report, stage3_design


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-design", type=Path, required=True)
    parser.add_argument("--source-design", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--attempt-lock", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--stage3-design-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    audit, stage3_design = audit_and_select(
        args.stage2_design.resolve(),
        args.source_design.resolve(),
        args.protocol.resolve(),
        args.attempt_lock.resolve(),
        args.reports_dir.resolve(),
        project_root=args.project_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        scratch=args.scratch.resolve(),
    )
    job_id = str(os.environ.get("SLURM_JOB_ID", "unknown"))
    stage3_output = args.stage3_design_output.resolve()
    write_bytes_new_or_identical(
        stage3_output, canonical_json_bytes(stage3_design), job_id
    )
    audit["stage_3_design"] = str(stage3_output)
    audit["stage_3_design_sha256"] = sha256_file(stage3_output)
    write_bytes_new_or_identical(
        args.output.resolve(), canonical_json_bytes(audit), job_id
    )
    print("HPO Stage 2 selection recorded in {}".format(args.output.resolve()))
    print("Frozen Stage 3 design written to {}".format(stage3_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
