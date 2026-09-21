"""Audit all HPO Stage 1 trajectories and freeze the Stage 2 entrants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .hpo_stage1 import (
    EXPECTED_ARRAY_TASKS,
    EXPECTED_CANDIDATES_PER_UNIT,
    EXPECTED_UNITS,
HPO_DESIGN_SHA256,
    select_array_task,
    stream_seed,
    validate_design,
)
from .kg_core import ranking_metrics
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object


class HPOStage1SelectionError(RuntimeError):
    """Raised when Stage 1 cannot be promoted mechanically."""


REPORT_NAME_RE = re.compile(r"^hpo_stage1_([0-9]+)_([0-9]+)\.json$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROTOCOL_SHA256 = "98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc"
ATTEMPT_LOCK_SHA256 = "dee93cf7da08cff01b1d32dc295bd0bb64770380921b5e2e880f33f564bd5f20"
COMPLETED_STATUS = "hpo_stage_1_run_completed"
FAILED_STATUS = "hpo_stage_1_run_failed"
NAMESPACES = (
    "random_entity_initialisation",
    "relation_initialisation",
    "batch_order",
    "negative_sampling",
    "data_loader_workers",
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HPOStage1SelectionError(label + " is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise HPOStage1SelectionError(label + " is non-finite")
    return number


def _finite_text_number(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise HPOStage1SelectionError(label + " is not numeric") from error
    if not math.isfinite(number):
        raise HPOStage1SelectionError(label + " is non-finite")
    return number


def _full_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise HPOStage1SelectionError(label + " is not a full SHA-256 digest")
    return value


def _resolve_under(project_root: Path, allowed_root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise HPOStage1SelectionError(label + " path is invalid")
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(allowed_root.resolve())
    except ValueError as error:
        raise HPOStage1SelectionError(label + " escapes its allowed root") from error
    if not resolved.is_file():
        raise HPOStage1SelectionError(label + " is missing")
    return resolved


def _project_relative(path: Path, project_root: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError as error:
        raise HPOStage1SelectionError(label + " escapes the project root") from error


def validate_artifact(
    artifact: Any,
    *,
    project_root: Path,
    artifact_root: Path,
    label: str,
) -> Tuple[Path, str]:
    if not isinstance(artifact, dict):
        raise HPOStage1SelectionError(label + " metadata is invalid")
    path = _resolve_under(project_root, artifact_root, artifact.get("path"), label)
    expected_size = artifact.get("size_bytes")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
        or path.stat().st_size != expected_size
    ):
        raise HPOStage1SelectionError(label + " size differs")
    expected_digest = _full_sha256(artifact.get("sha256"), label + " hash")
    if sha256_file(path) != expected_digest:
        raise HPOStage1SelectionError(label + " hash differs")
    return path, expected_digest


def _metric_block(block: Any, expected_queries: int, label: str) -> None:
    if not isinstance(block, dict) or block.get("queries") != expected_queries:
        raise HPOStage1SelectionError(label + " query count differs")
    for key in ("mrr", "hits_at_1", "hits_at_3", "hits_at_10"):
        value = _finite_number(block.get(key), label + " " + key)
        if not 0.0 <= value <= 1.0:
            raise HPOStage1SelectionError(label + " metric is outside [0,1]")


def _metrics_match(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key in ("queries", "mrr", "hits_at_1", "hits_at_3", "hits_at_10"):
        if observed.get(key) != expected.get(key):
            raise HPOStage1SelectionError(label + " differs for " + key)


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
            raise HPOStage1SelectionError("Validation rank table is empty") from error
        if header != expected_header:
            raise HPOStage1SelectionError("Validation rank-table header differs")
        rows = list(reader)
    if len(rows) != 2 * validation_triples:
        raise HPOStage1SelectionError("Validation rank-table row count differs")
    for validation_row in range(validation_triples):
        pair = rows[2 * validation_row : 2 * validation_row + 2]
        if len(pair) != 2 or any(len(row) != len(expected_header) for row in pair):
            raise HPOStage1SelectionError("Validation rank-table shape differs")
        if [row[1] for row in pair] != ["head", "tail"]:
            raise HPOStage1SelectionError("Validation rank-table direction order differs")
        if any(int(row[0]) != validation_row for row in pair):
            raise HPOStage1SelectionError("Validation rank-table index differs")
        triples = [tuple(map(int, row[2:5])) for row in pair]
        if triples[0] != triples[1]:
            raise HPOStage1SelectionError("Head/tail rank rows refer to different triples")
        head, _, tail = triples[0]
        if int(pair[0][5]) != head or int(pair[1][5]) != tail:
            raise HPOStage1SelectionError("Validation target entity differs")
        ranks = [_finite_text_number(row[6], "validation rank") for row in pair]
        if any(rank < 1.0 for rank in ranks):
            raise HPOStage1SelectionError("Validation rank is below one")
        head_ranks.append(ranks[0])
        tail_ranks.append(ranks[1])
    _metrics_match(validation["head"], ranking_metrics(head_ranks), "head metrics")
    _metrics_match(validation["tail"], ranking_metrics(tail_ranks), "tail metrics")
    _metrics_match(
        validation["combined"],
        ranking_metrics(head_ranks + tail_ranks),
        "combined metrics",
    )


def selection_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Protocol ordering: MRR desc, K asc, batch asc, ID lexicographic."""

    parameters = record["parameters"]
    return (
        -float(record["validation_mrr"]),
        int(parameters["negatives_per_positive"]),
        int(parameters["batch_size"]),
        str(record["candidate_id"]),
    )


def select_advancing(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if len(records) < 4:
        raise HPOStage1SelectionError("A tuning unit has fewer than four valid runs")
    ordered = sorted(records, key=selection_key)
    selected = []
    for rank, record in enumerate(ordered[:4], 1):
        selected.append({**dict(record), "stage_1_rank": rank})
    return selected


def discover_reports(reports_dir: Path) -> List[Path]:
    paths = []
    for path in reports_dir.iterdir():
        if path.is_file() and REPORT_NAME_RE.fullmatch(path.name):
            paths.append(path)
    return sorted(paths, key=lambda path: path.name)


def validate_attempt_lock(
    attempt_lock: Mapping[str, Any]
) -> Tuple[Dict[int, str], Dict[str, Mapping[str, Any]]]:
    exact = {
        "schema_version": 1,
        "status": "hpo_stage_1_attempts_frozen_pending_selection",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "hpo_design_sha256": HPO_DESIGN_SHA256,
    }
    for key, value in exact.items():
        if attempt_lock.get(key) != value:
            raise HPOStage1SelectionError("Stage 1 attempt lock differs in " + key)
    ranges = attempt_lock.get("canonical_attempt_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise HPOStage1SelectionError("Canonical Stage 1 attempt ranges are missing")
    canonical = {}
    for item in ranges:
        if not isinstance(item, dict):
            raise HPOStage1SelectionError("A canonical attempt range is invalid")
        job_id = item.get("array_job_id")
        first = item.get("first_array_index")
        last = item.get("last_array_index")
        if (
            not isinstance(job_id, str)
            or not job_id.isdigit()
            or not isinstance(first, int)
            or isinstance(first, bool)
            or not isinstance(last, int)
            or isinstance(last, bool)
            or not 0 <= first <= last < EXPECTED_ARRAY_TASKS
        ):
            raise HPOStage1SelectionError("A canonical attempt range is invalid")
        for index in range(first, last + 1):
            if index in canonical:
                raise HPOStage1SelectionError("Canonical attempt ranges overlap")
            canonical[index] = "hpo_stage1_{}_{}.json".format(job_id, index)
    if set(canonical) != set(range(EXPECTED_ARRAY_TASKS)):
        raise HPOStage1SelectionError("Canonical attempt ranges do not cover 0..71")
    exclusions = attempt_lock.get("excluded_attempts")
    if not isinstance(exclusions, list):
        raise HPOStage1SelectionError("Excluded Stage 1 attempts are invalid")
    excluded_by_name = {}
    for item in exclusions:
        if not isinstance(item, dict):
            raise HPOStage1SelectionError("An excluded Stage 1 attempt is invalid")
        job_id = item.get("array_job_id")
        index = item.get("array_index")
        reason = item.get("reason")
        if (
            not isinstance(job_id, str)
            or not job_id.isdigit()
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < EXPECTED_ARRAY_TASKS
            or not isinstance(reason, str)
            or not reason
        ):
            raise HPOStage1SelectionError("An excluded Stage 1 attempt is invalid")
        name = "hpo_stage1_{}_{}.json".format(job_id, index)
        if name in excluded_by_name or name in canonical.values():
            raise HPOStage1SelectionError("Stage 1 attempt identities overlap")
        excluded_by_name[name] = item
    if not isinstance(attempt_lock.get("selection_rule"), str):
        raise HPOStage1SelectionError("Stage 1 canonical selection rule is missing")
    return canonical, excluded_by_name


def _validate_completed_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    design: Mapping[str, Any],
    project_root: Path,
    artifact_root: Path,
    protocol_sha256: str,
) -> Dict[str, Any]:
    array_index = int(report["array_index"])
    unit, candidate = select_array_task(design, array_index)
    exact = {
        "schema_version": 1,
        "status": COMPLETED_STATUS,
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": 1,
        "replicate": 1,
        "dataset": unit["dataset"],
        "model": unit["model"],
        "unit_id": unit["unit_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_configuration_sha256": candidate["configuration_sha256"],
        "parameters": candidate["parameters"],
        "epochs": 25,
        "hpo_design_sha256": HPO_DESIGN_SHA256,
        "hardware_preflight_report_sha256": design["hardware_preflight_report_sha256"],
        "protocol_sha256": protocol_sha256,
    }
    for key, value in exact.items():
        if report.get(key) != value:
            raise HPOStage1SelectionError(
                "{} differs in {}".format(report_path.name, key)
            )
    expected_seeds = {namespace: stream_seed(unit, namespace) for namespace in NAMESPACES}
    if report.get("seeds") != expected_seeds:
        raise HPOStage1SelectionError(report_path.name + " has different tuning seeds")
    data = report.get("data")
    if not isinstance(data, dict):
        raise HPOStage1SelectionError(report_path.name + " data audit is missing")
    validation_triples = data.get("validation_triples")
    training_triples = data.get("training_triples")
    if (
        not isinstance(validation_triples, int)
        or validation_triples <= 0
        or not isinstance(training_triples, int)
        or training_triples <= 0
    ):
        raise HPOStage1SelectionError(report_path.name + " split counts are invalid")
    training = report.get("training")
    if not isinstance(training, dict):
        raise HPOStage1SelectionError(report_path.name + " training audit is missing")
    batch_size = int(candidate["parameters"]["batch_size"])
    batches_per_epoch = (training_triples + batch_size - 1) // batch_size
    expected_batches = 25 * batches_per_epoch
    if training.get("batches") != expected_batches:
        raise HPOStage1SelectionError(report_path.name + " batch count differs")
    history = training.get("epoch_history")
    if not isinstance(history, list) or len(history) != 25:
        raise HPOStage1SelectionError(report_path.name + " epoch history differs")
    for epoch, item in enumerate(history, 1):
        if item.get("epoch") != epoch or item.get("batches") != batches_per_epoch:
            raise HPOStage1SelectionError(report_path.name + " epoch audit differs")
        minimum = _finite_number(item.get("minimum_total_loss"), "minimum loss")
        mean = _finite_number(item.get("mean_total_loss"), "mean loss")
        maximum = _finite_number(item.get("maximum_total_loss"), "maximum loss")
        if not minimum <= mean <= maximum:
            raise HPOStage1SelectionError(report_path.name + " loss ordering differs")
        if _finite_number(item.get("runtime_seconds"), "epoch runtime") <= 0:
            raise HPOStage1SelectionError(report_path.name + " epoch runtime is invalid")
    if training.get("globally_training_unseen_maximum_absolute_change") != 0.0:
        raise HPOStage1SelectionError(report_path.name + " changed an unseen entity")
    _full_sha256(training.get("training_sequence_sha256"), "training sequence hash")
    validation = report.get("validation")
    if not isinstance(validation, dict) or validation.get("filter_scope") != "train_plus_validation":
        raise HPOStage1SelectionError(report_path.name + " validation scope differs")
    _metric_block(validation.get("head"), validation_triples, "head validation")
    _metric_block(validation.get("tail"), validation_triples, "tail validation")
    _metric_block(validation.get("combined"), 2 * validation_triples, "combined validation")
    initialisation = report.get("initialisation")
    if not isinstance(initialisation, dict) or initialisation.get("dimension") != 384:
        raise HPOStage1SelectionError(report_path.name + " initialisation differs")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HPOStage1SelectionError(report_path.name + " artifacts are missing")
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
                raise HPOStage1SelectionError(report_path.name + " rank row count differs")
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
                raise HPOStage1SelectionError(report_path.name + " loss artifact shape differs")
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
        raise HPOStage1SelectionError(report_path.name + " initialisation artifact shape differs")
    entity_raw_sha = _full_sha256(
        initialisation.get("entity_matrix_raw_sha256"), "entity initialisation raw hash"
    )
    relation_raw_sha = _full_sha256(
        initialisation.get("relation_matrix_raw_sha256"), "relation initialisation raw hash"
    )
    execution = report.get("execution")
    if not isinstance(execution, dict) or execution.get("execution_context") != "slurm":
        raise HPOStage1SelectionError(report_path.name + " was not scheduler-executed")
    environment = report.get("training_environment")
    if not isinstance(environment, dict) or environment.get("device_name") != "NVIDIA H200 NVL":
        raise HPOStage1SelectionError(report_path.name + " environment differs")
    source_git = report.get("git")
    if not isinstance(source_git, dict) or source_git.get("tracked_worktree_clean") is not True:
        raise HPOStage1SelectionError(report_path.name + " source worktree was not clean")
    source_commit = source_git.get("commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise HPOStage1SelectionError(report_path.name + " source commit is invalid")
    return {
        "array_index": array_index,
        "dataset": unit["dataset"],
        "model": unit["model"],
        "unit_id": unit["unit_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_configuration_sha256": candidate["configuration_sha256"],
        "parameters": candidate["parameters"],
        "validation_mrr": validation["combined"]["mrr"],
        "report_path": _project_relative(report_path, project_root, "Stage 1 report"),
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
    design_path: Path,
    protocol_path: Path,
    attempt_lock_path: Path,
    reports_dir: Path,
    *,
    project_root: Path,
    artifact_root: Path,
    scratch: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    execution = require_scheduled_execution(scratch.resolve())
    if sha256_file(design_path) != HPO_DESIGN_SHA256:
        raise HPOStage1SelectionError("Frozen HPO design hash differs")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise HPOStage1SelectionError("Frozen protocol hash differs")
    if sha256_file(attempt_lock_path) != ATTEMPT_LOCK_SHA256:
        raise HPOStage1SelectionError("Frozen Stage 1 attempt-lock hash differs")
    design = read_json_object(design_path)
    validate_design(design)
    attempt_lock = read_json_object(attempt_lock_path)
    canonical_names, excluded_specs = validate_attempt_lock(attempt_lock)
    _project_relative(reports_dir, project_root, "Stage 1 reports directory")
    _project_relative(artifact_root, project_root, "Stage 1 artifact root")
    paths = discover_reports(reports_dir)
    expected_names = set(canonical_names.values()) | set(excluded_specs)
    observed_names = {path.name for path in paths}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise HPOStage1SelectionError(
            "Stage 1 attempt set differs; missing={!r}, unexpected={!r}".format(
                missing, unexpected
            )
        )
    by_index: Dict[int, Path] = {}
    for index, filename in canonical_names.items():
        path = reports_dir / filename
        report = read_json_object(path)
        report_index = report.get("array_index")
        if report_index != index:
            raise HPOStage1SelectionError(path.name + " has an invalid array index")
        filename_match = REPORT_NAME_RE.fullmatch(path.name)
        if filename_match is None or int(filename_match.group(2)) != report_index:
            raise HPOStage1SelectionError(path.name + " disagrees with its array index")
        job_id_field = (
            "array_job_id"
            if report.get("status") == COMPLETED_STATUS
            else "slurm_array_job_id"
        )
        if str(report.get(job_id_field)) != filename_match.group(1):
            raise HPOStage1SelectionError(path.name + " disagrees with its array job ID")
        by_index[index] = path
    if set(by_index) != set(range(72)):
        raise HPOStage1SelectionError("Stage 1 array-index coverage differs")
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for index in range(72):
        path = by_index[index]
        report = read_json_object(path)
        status = report.get("status")
        if status == COMPLETED_STATUS:
            valid.append(
                _validate_completed_report(
                    report,
                    report_path=path,
                    design=design,
                    project_root=project_root,
                    artifact_root=artifact_root,
                    protocol_sha256=PROTOCOL_SHA256,
                )
            )
        elif status == FAILED_STATUS:
            unit, candidate = select_array_task(design, index)
            invalid.append(
                {
                    "array_index": index,
                    "unit_id": unit["unit_id"],
                    "candidate_id": candidate["candidate_id"],
                    "error_type": report.get("error_type"),
                    "error": report.get("error"),
                    "report_path": _project_relative(path, project_root, "failed report"),
                    "report_sha256": sha256_file(path),
                }
            )
        else:
            raise HPOStage1SelectionError(path.name + " has an unknown status")
    excluded_attempts = []
    for filename, specification in sorted(excluded_specs.items()):
        path = reports_dir / filename
        report = read_json_object(path)
        index = int(specification["array_index"])
        if report.get("array_index") != index:
            raise HPOStage1SelectionError(filename + " has an invalid excluded index")
        unit, candidate = select_array_task(design, index)
        status = report.get("status")
        if status == COMPLETED_STATUS:
            audited = _validate_completed_report(
                report,
                report_path=path,
                design=design,
                project_root=project_root,
                artifact_root=artifact_root,
                protocol_sha256=PROTOCOL_SHA256,
            )
            candidate_id = audited["candidate_id"]
        elif status == FAILED_STATUS:
            candidate_id = candidate["candidate_id"]
        else:
            raise HPOStage1SelectionError(filename + " has an unknown excluded status")
        excluded_attempts.append(
            {
                "array_job_id": specification["array_job_id"],
                "array_index": index,
                "dataset": unit["dataset"],
                "model": unit["model"],
                "candidate_id": candidate_id,
                "status": status,
                "reason": specification["reason"],
                "report_path": _project_relative(path, project_root, "excluded report"),
                "report_sha256": sha256_file(path),
                "used_for_selection": False,
            }
        )
    environments = {canonical_json_bytes(item["training_environment"]) for item in valid}
    commits = {item["source_git_commit"] for item in valid}
    if len(environments) != 1 or len(commits) != 1:
        raise HPOStage1SelectionError("Stage 1 environment/source commit is inconsistent")
    by_unit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in valid:
        by_unit[item["unit_id"]].append(item)
    invalid_by_unit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in invalid:
        invalid_by_unit[item["unit_id"]].append(item)
    units = []
    selection_units = []
    for design_unit in design["units"]:
        unit_id = str(design_unit["unit_id"])
        unit_valid = by_unit[unit_id]
        identities = {
            canonical_json_bytes(item["initialisation_identity"])
            for item in unit_valid
        }
        if len(identities) != 1:
            raise HPOStage1SelectionError(unit_id + " did not share paired initialisation")
        ordered = sorted(unit_valid, key=selection_key)
        ranking = [
            {
                **{key: value for key, value in item.items() if key not in {
                    "training_environment", "source_git_commit", "initialisation_identity"
                }},
                "stage_1_rank": rank,
            }
            for rank, item in enumerate(ordered, 1)
        ]
        advancing = select_advancing(unit_valid)
        advancing_compact = [
            {
                "candidate_id": item["candidate_id"],
                "candidate_configuration_sha256": item["candidate_configuration_sha256"],
                "parameters": item["parameters"],
                "stage_1_rank": item["stage_1_rank"],
                "stage_1_validation_mrr": item["validation_mrr"],
                "stage_1_report_sha256": item["report_sha256"],
            }
            for item in advancing
        ]
        units.append(
            {
                "dataset": design_unit["dataset"],
                "model": design_unit["model"],
                "unit_id": unit_id,
                "valid_runs": len(unit_valid),
                "invalid_runs": invalid_by_unit[unit_id],
                "ranking": ranking,
                "advancing_candidate_ids": [item["candidate_id"] for item in advancing],
            }
        )
        selection_units.append(
            {
                "dataset": design_unit["dataset"],
                "model": design_unit["model"],
                "unit_id": unit_id,
                "tuning_stream_derivations": design_unit["tuning_stream_derivations"],
                "advancing_candidates": advancing_compact,
            }
        )
    if len(units) != EXPECTED_UNITS or any(
        len(unit["ranking"]) + len(unit["invalid_runs"])
        != EXPECTED_CANDIDATES_PER_UNIT
        for unit in units
    ):
        raise HPOStage1SelectionError("Stage 1 unit coverage differs")
    report_manifest = [
        {
            "array_index": index,
            "path": _project_relative(by_index[index], project_root, "Stage 1 report"),
            "sha256": sha256_file(by_index[index]),
        }
        for index in range(72)
    ]
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(report_manifest)).hexdigest()
    stage2_design = {
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
        "protocol_sha256": PROTOCOL_SHA256,
        "source_hpo_design_sha256": HPO_DESIGN_SHA256,
        "source_stage_1_attempt_lock_sha256": ATTEMPT_LOCK_SHA256,
        "source_stage_1_report_manifest_sha256": manifest_sha256,
        "source_stage_1_runner_git_commit": next(iter(commits)),
        "fixed_training": design["fixed_training"],
        "units": selection_units,
    }
    audit_report = {
        "schema_version": 1,
        "status": "hpo_stage_1_audited_and_selected_pending_stage_2",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "execution": execution,
        "git": git_state(project_root),
        "protocol_sha256": PROTOCOL_SHA256,
        "hpo_design_sha256": HPO_DESIGN_SHA256,
        "stage_1_attempt_lock_sha256": ATTEMPT_LOCK_SHA256,
        "source_stage_1_report_manifest": report_manifest,
        "source_stage_1_report_manifest_sha256": manifest_sha256,
        "valid_runs": len(valid),
        "invalid_runs": invalid,
        "excluded_attempts": excluded_attempts,
        "training_environment": valid[0]["training_environment"],
        "source_stage_1_runner_git_commit": next(iter(commits)),
        "units": units,
    }
    return audit_report, stage2_design


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--attempt-lock", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--stage2-design-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    audit, stage2_design = audit_and_select(
        args.design.resolve(),
        args.protocol.resolve(),
        args.attempt_lock.resolve(),
        args.reports_dir.resolve(),
        project_root=args.project_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        scratch=args.scratch.resolve(),
    )
    job_id = str(os.environ.get("SLURM_JOB_ID", "unknown"))
    stage2_output = args.stage2_design_output.resolve()
    write_bytes_new_or_identical(
        stage2_output, canonical_json_bytes(stage2_design), job_id
    )
    audit["stage_2_design"] = str(stage2_output)
    audit["stage_2_design_sha256"] = sha256_file(stage2_output)
    write_bytes_new_or_identical(args.output.resolve(), canonical_json_bytes(audit), job_id)
    print("HPO Stage 1 selection recorded in {}".format(args.output.resolve()))
    print("Frozen Stage 2 design written to {}".format(stage2_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
