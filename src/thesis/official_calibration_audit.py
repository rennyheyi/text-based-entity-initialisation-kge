"""Audit 90 validation calibrations and freeze their temperatures before test access."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .official_calibration import (
    EXPECTED_RUNS,
    OFFICIAL_CHECKPOINT_LOCK_SHA256,
    OFFICIAL_EVALUATION_LOCK_SHA256,
    select_run_and_checkpoint,
    validate_evaluation_lock,
)
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
    LOG_TEMPERATURE_XATOL,
    TEMPERATURE_LOWER,
    TEMPERATURE_MAXITER,
    TEMPERATURE_UPPER,
)


OFFICIAL_DESIGN_SHA256 = EXPECTED_OFFICIAL_DESIGN_SHA256


class OfficialCalibrationAuditError(RuntimeError):
    """Raised when an official validation-calibration report fails audit."""


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def calibration_report_path(project_root: Path, run: Mapping[str, Any]) -> Path:
    return (
        project_root
        / "reports"
        / "official_calibration"
        / str(run["dataset"])
        / str(run["model"])
        / "replicate_{:02d}".format(int(run["official_replicate"]))
        / (str(run["condition"]) + ".json")
    )


def validate_calibration_report(
    report: Mapping[str, Any],
    run: Mapping[str, Any],
    locked: Mapping[str, Any],
) -> None:
    exact = {
        "schema_version": 1,
        "status": "official_validation_calibration_completed_pending_temperature_freeze",
        "official_result": False,
        "validation_data_accessed": True,
        "test_data_accessed": False,
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
        "checkpoint": locked["checkpoint"],
        "official_design_sha256": OFFICIAL_DESIGN_SHA256,
        "official_checkpoint_lock_sha256": OFFICIAL_CHECKPOINT_LOCK_SHA256,
        "official_evaluation_lock_sha256": OFFICIAL_EVALUATION_LOCK_SHA256,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise OfficialCalibrationAuditError("Calibration report field differs: " + key)
    validation = report.get("validation")
    if not isinstance(validation, dict):
        raise OfficialCalibrationAuditError("Calibration validation summary is missing")
    triples = validation.get("triples")
    if (
        not isinstance(triples, int)
        or triples <= 0
        or validation.get("head_queries") != triples
        or validation.get("tail_queries") != triples
        or validation.get("combined_queries") != 2 * triples
        or validation.get("test_split_loaded") is not False
        or validation.get("filtering_splits") != ["train", "validation"]
    ):
        raise OfficialCalibrationAuditError("Calibration query boundary differs")
    if (
        not isinstance(validation.get("minimum_filtered_candidate_count"), int)
        or validation["minimum_filtered_candidate_count"] <= 0
        or validation["maximum_filtered_candidate_count"]
        > validation["candidate_vocabulary"]
    ):
        raise OfficialCalibrationAuditError("Calibration candidate counts are invalid")

    temperature = report.get("temperature")
    if not isinstance(temperature, dict):
        raise OfficialCalibrationAuditError("Calibration temperature record is missing")
    expected_temperature = {
        "method": "scipy.optimize.minimize_scalar_bounded",
        "parameterisation": "natural_log_temperature",
        "temperature_bounds": [TEMPERATURE_LOWER, TEMPERATURE_UPPER],
        "xatol_log_temperature": LOG_TEMPERATURE_XATOL,
        "maximum_iterations": TEMPERATURE_MAXITER,
        "success": True,
        "scipy_success": True,
    }
    for key, expected in expected_temperature.items():
        if temperature.get(key) != expected:
            raise OfficialCalibrationAuditError("Temperature field differs: " + key)
    fitted = temperature.get("fitted_temperature")
    log_fitted = temperature.get("fitted_log_temperature")
    objectives = (
        temperature.get("validation_mean_multiclass_nll"),
        temperature.get("raw_temperature_one_validation_mean_multiclass_nll"),
        temperature.get("boundary_objectives", {}).get("lower"),
        temperature.get("boundary_objectives", {}).get("upper"),
    )
    if (
        not isinstance(fitted, (int, float))
        or not TEMPERATURE_LOWER <= fitted <= TEMPERATURE_UPPER
        or not math.isfinite(float(fitted))
        or not isinstance(log_fitted, (int, float))
        or not math.isclose(math.log(float(fitted)), float(log_fitted), rel_tol=0.0, abs_tol=1e-12)
        or temperature.get("boundary_status") not in {"lower", "interior", "upper"}
        or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in objectives)
    ):
        raise OfficialCalibrationAuditError("Fitted temperature or objective is invalid")
    if temperature["validation_mean_multiclass_nll"] > min(
        temperature["boundary_objectives"].values()
    ):
        raise OfficialCalibrationAuditError("Fitted objective is worse than a frozen boundary")
    git = report.get("git")
    if (
        not isinstance(git, dict)
        or git.get("tracked_worktree_clean") is not True
        or not COMMIT_RE.fullmatch(str(git.get("commit", "")))
    ):
        raise OfficialCalibrationAuditError("Calibration Git identity is invalid")


def audit_official_calibrations(
    *,
    design_path: Path,
    checkpoint_lock_path: Path,
    evaluation_lock_path: Path,
    project_root: Path,
    scratch: Path,
    output_lock: Path,
    output_report: Path,
) -> Dict[str, Any]:
    execution = require_scheduled_execution(scratch.resolve())
    if sha256_file(design_path) != OFFICIAL_DESIGN_SHA256:
        raise OfficialCalibrationAuditError("Official design hash differs")
    if sha256_file(checkpoint_lock_path) != OFFICIAL_CHECKPOINT_LOCK_SHA256:
        raise OfficialCalibrationAuditError("Official checkpoint lock hash differs")
    if sha256_file(evaluation_lock_path) != OFFICIAL_EVALUATION_LOCK_SHA256:
        raise OfficialCalibrationAuditError("Official evaluation lock hash differs")
    design = read_json_object(design_path)
    validate_design_invariants(design)
    checkpoint_lock = read_json_object(checkpoint_lock_path)
    evaluation_lock = read_json_object(evaluation_lock_path)
    validate_evaluation_lock(evaluation_lock)

    records: List[Dict[str, Any]] = []
    commits = set()
    for run in design["runs"]:
        selected, locked = select_run_and_checkpoint(
            design, checkpoint_lock, int(run["array_index"])
        )
        if selected["run_id"] != run["run_id"]:
            raise OfficialCalibrationAuditError("Calibration design selection differs")
        path = calibration_report_path(project_root, run)
        if not path.is_file():
            raise OfficialCalibrationAuditError("Calibration report is missing: " + str(path))
        report_sha256 = sha256_file(path)
        report = read_json_object(path)
        validate_calibration_report(report, run, locked)
        commits.add(report["git"]["commit"])
        records.append(
            {
                "array_index": run["array_index"],
                "run_id": run["run_id"],
                "paired_block_id": run["paired_block_id"],
                "dataset": run["dataset"],
                "model": run["model"],
                "condition": run["condition"],
                "official_replicate": run["official_replicate"],
                "official_seed": run["official_seed"],
                "run_configuration_sha256": run["run_configuration_sha256"],
                "checkpoint_sha256": locked["checkpoint"]["sha256"],
                "calibration_report": {
                    "path": str(path.relative_to(project_root)),
                    "sha256": report_sha256,
                    "size_bytes": path.stat().st_size,
                },
                "temperature": report["temperature"],
            }
        )
    if len(records) != EXPECTED_RUNS or len({row["run_id"] for row in records}) != EXPECTED_RUNS:
        raise OfficialCalibrationAuditError("Calibration record count differs from 90")
    if len(commits) != 1:
        raise OfficialCalibrationAuditError("Calibration runs do not share one code commit")
    calibration_commit = next(iter(commits))
    lock = {
        "schema_version": 1,
        "status": "official_temperatures_frozen_pending_test_evaluation",
        "official_result": False,
        "validation_data_accessed": True,
        "test_data_accessed": False,
        "test_metrics_computed": False,
        "official_design_sha256": OFFICIAL_DESIGN_SHA256,
        "official_checkpoint_lock_sha256": OFFICIAL_CHECKPOINT_LOCK_SHA256,
        "official_evaluation_lock_sha256": OFFICIAL_EVALUATION_LOCK_SHA256,
        "calibration_git_commit": calibration_commit,
        "calibration_runs": len(records),
        "temperatures_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
        "temperatures": records,
    }
    lock_bytes = canonical_json_bytes(lock)
    write_new_or_identical(output_lock, lock_bytes)
    report = {
        "schema_version": 1,
        "status": "official_calibrations_audited_temperatures_frozen_pending_test_evaluation",
        "official_result": False,
        "validation_data_accessed": True,
        "test_data_accessed": False,
        "test_metrics_computed": False,
        "calibration_runs": len(records),
        "official_calibration_lock": {
            "path": str(output_lock.relative_to(project_root)),
            "sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "size_bytes": len(lock_bytes),
        },
        "calibration_git_commit": calibration_commit,
        "execution": execution,
        "git": git_state(project_root),
    }
    write_new_or_identical(output_report, canonical_json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--checkpoint-lock", type=Path, required=True)
    parser.add_argument("--evaluation-lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_official_calibrations(
        design_path=args.design.resolve(),
        checkpoint_lock_path=args.checkpoint_lock.resolve(),
        evaluation_lock_path=args.evaluation_lock.resolve(),
        project_root=args.project_root.resolve(),
        scratch=args.scratch.resolve(),
        output_lock=args.output_lock.resolve(),
        output_report=args.output_report.resolve(),
    )
    print("Official calibration audit recorded in {}".format(args.output_report.resolve()))
    print("Official temperature lock written to {}".format(args.output_lock.resolve()))
    print(report["official_calibration_lock"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
