"""Audit 90 official test evaluations and execute the frozen paired analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .official_calibration import (
    EXPECTED_RUNS,
    OFFICIAL_CHECKPOINT_LOCK_SHA256,
    OFFICIAL_DESIGN_SHA256,
    OFFICIAL_EVALUATION_LOCK_SHA256,
    select_run_and_checkpoint,
    validate_evaluation_lock,
)
from .official_design import (
    canonical_json_bytes,
    git_state,
    read_json_object,
    require_scheduled_execution,
    sha256_file,
    validate_design_invariants,
    write_new_or_identical,
)
from .official_test_evaluation import (
    ECE_BIN_COUNT,
    OFFICIAL_CALIBRATION_LOCK_SHA256,
    OFFICIAL_CLAIMS_LOCK_SHA256,
    PER_QUERY_COLUMNS,
    aggregate_query_metrics,
    artifact_record,
    select_temperature,
    strict_tsv_bytes,
)


class OfficialTestEvaluationAuditError(RuntimeError):
    """Raised when official test artifacts or paired statistics fail audit."""


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CONTRASTS = (
    ("correct_text_vs_random", "correct_text", "random", "primary"),
    ("correct_text_vs_shuffled_text", "correct_text", "shuffled_text", "primary"),
    ("shuffled_text_vs_random", "shuffled_text", "random", "secondary"),
)
HIGHER_BETTER = {
    "combined_filtered_test_mrr", "test_hits_at_1", "test_hits_at_3", "test_hits_at_10",
    "top_label_accuracy", "raw_test_auroc", "calibrated_test_auroc",
}
LOWER_BETTER = {
    "raw_test_multiclass_nll", "calibrated_test_multiclass_nll",
    "raw_test_ece", "calibrated_test_ece", "raw_test_top_label_brier",
    "calibrated_test_top_label_brier", "raw_test_aurc", "calibrated_test_aurc",
    "raw_test_eaurc", "calibrated_test_eaurc",
}
DESCRIPTIVE = {
    "raw_test_normalised_entropy", "calibrated_test_normalised_entropy", "fitted_temperature",
}
ANALYSIS_METRICS = tuple(sorted(HIGHER_BETTER | LOWER_BETTER | DESCRIPTIVE))
PRIMARY_METRICS = (
    "combined_filtered_test_mrr",
    "calibrated_test_multiclass_nll",
    "calibrated_test_eaurc",
)
FAILED_OFFICIAL_TEST_ATTEMPTS = (
    {
        "array_job_id": "441554",
        "array_index": 0,
        "status": "failed_before_test_metric_generation",
        "error_type": "KeyError",
        "reason": (
            "The pilot used the HPO-safe train/validation loader, which intentionally "
            "does not expose the test split; no official test report or metric artifact "
            "was produced. The implementation was corrected before resubmission."
        ),
        "used_for_results": False,
    },
)


def official_test_report_path(project_root: Path, run: Mapping[str, Any]) -> Path:
    return (
        project_root / "reports" / "official_test" / str(run["dataset"])
        / str(run["model"]) / "replicate_{:02d}".format(int(run["official_replicate"]))
        / (str(run["condition"]) + ".json")
    )


def parse_per_query_table(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PER_QUERY_COLUMNS:
            raise OfficialTestEvaluationAuditError("Per-query columns differ")
        source = list(reader)
    integer_fields = {
        "official_replicate", "head_entity_index", "relation_index", "tail_entity_index",
        "target_entity_index", "filtered_candidate_count", "predicted_entity_index",
        "top_score_tie_count",
    }
    float_fields = {
        "target_score", "realistic_target_rank", "maximum_score", "raw_target_probability",
        "calibrated_target_probability", "raw_confidence", "calibrated_confidence", "raw_nll",
        "calibrated_nll", "raw_normalised_entropy", "calibrated_normalised_entropy",
        "fitted_temperature",
    }
    rows = []
    for row in source:
        converted: Dict[str, Any] = dict(row)
        for key in integer_fields:
            converted[key] = int(converted[key])
        for key in float_fields:
            converted[key] = float(converted[key])
            if not math.isfinite(converted[key]):
                raise OfficialTestEvaluationAuditError("Per-query table contains non-finite value")
        if converted["top_label_correct"] not in {"true", "false"}:
            raise OfficialTestEvaluationAuditError("Per-query correctness is invalid")
        converted["top_label_correct"] = converted["top_label_correct"] == "true"
        rows.append(converted)
    return rows


def validate_artifact(record: Mapping[str, Any], project_root: Path) -> Path:
    path_value = record.get("path")
    if not isinstance(path_value, str):
        raise OfficialTestEvaluationAuditError("Artifact path is missing")
    path = project_root / path_value
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise OfficialTestEvaluationAuditError("Artifact escapes project root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise OfficialTestEvaluationAuditError("Artifact identity differs: " + path_value)
    return path


def validate_test_report(
    report: Mapping[str, Any], run: Mapping[str, Any], locked: Mapping[str, Any],
    temperature_row: Mapping[str, Any], project_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Mapping[str, Any]]]:
    exact = {
        "schema_version": 1,
        "status": "official_test_evaluation_completed_pending_audit",
        "official_result": True,
        "validation_data_accessed": True,
        "test_data_accessed": True,
        "test_metrics_computed": True,
        "array_index": run["array_index"], "run_id": run["run_id"],
        "paired_block_id": run["paired_block_id"], "dataset": run["dataset"],
        "model": run["model"], "condition": run["condition"],
        "official_replicate": run["official_replicate"], "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "checkpoint_sha256": locked["checkpoint"]["sha256"],
        "fitted_temperature": temperature_row["temperature"]["fitted_temperature"],
        "official_design_sha256": OFFICIAL_DESIGN_SHA256,
        "official_checkpoint_lock_sha256": OFFICIAL_CHECKPOINT_LOCK_SHA256,
        "official_evaluation_lock_sha256": OFFICIAL_EVALUATION_LOCK_SHA256,
        "official_calibration_lock_sha256": OFFICIAL_CALIBRATION_LOCK_SHA256,
        "official_claims_lock_sha256": OFFICIAL_CLAIMS_LOCK_SHA256,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise OfficialTestEvaluationAuditError("Official test report field differs: " + key)
    test = report.get("test", {})
    triples = test.get("triples")
    if (
        not isinstance(triples, int) or triples <= 0
        or test.get("head_queries") != triples or test.get("tail_queries") != triples
        or test.get("combined_queries") != 2 * triples
        or test.get("filtering_splits") != ["train", "validation", "test"]
        or not isinstance(test.get("candidate_vocabulary"), int)
        or test.get("minimum_filtered_candidate_count", 0) <= 0
        or test.get("maximum_filtered_candidate_count", 0) > test["candidate_vocabulary"]
    ):
        raise OfficialTestEvaluationAuditError("Official test query summary differs")
    artifacts = report.get("artifacts", {})
    if set(artifacts) != {"per_query_table", "aggregate_metrics", "ece_bins", "risk_coverage"}:
        raise OfficialTestEvaluationAuditError("Official test artifact set differs")
    paths = {key: validate_artifact(value, project_root) for key, value in artifacts.items()}
    rows = parse_per_query_table(paths["per_query_table"])
    if len(rows) != 2 * triples or len({row["query_id"] for row in rows}) != len(rows):
        raise OfficialTestEvaluationAuditError("Per-query row count or identity differs")
    for row in rows:
        row_exact = {
            "dataset": run["dataset"], "model": run["model"], "condition": run["condition"],
            "official_replicate": run["official_replicate"], "split": "test",
            "fitted_temperature": report["fitted_temperature"],
        }
        if any(row.get(key) != value for key, value in row_exact.items()):
            raise OfficialTestEvaluationAuditError("Per-query frozen identity differs")
        if (
            row["direction"] not in {"head", "tail"}
            or row["target_seen_status"] not in {"seen", "unseen"}
            or row["triple_endpoint_seen_status"] not in {"both_seen", "at_least_one_unseen"}
            or row["filtered_candidate_count"] <= 0
            or row["filtered_candidate_count"] > test["candidate_vocabulary"]
            or not 1.0 <= row["realistic_target_rank"] <= row["filtered_candidate_count"]
            or not 0 <= row["predicted_entity_index"] < test["candidate_vocabulary"]
            or row["top_score_tie_count"] <= 0
            or any(not 0.0 <= row[key] <= 1.0 for key in (
                "raw_target_probability", "calibrated_target_probability",
                "raw_confidence", "calibrated_confidence",
                "raw_normalised_entropy", "calibrated_normalised_entropy",
            ))
        ):
            raise OfficialTestEvaluationAuditError("Per-query value is outside frozen bounds")
    recomputed, ece_rows, risk_rows = aggregate_query_metrics(rows, float(report["fitted_temperature"]))
    if report.get("metrics") != recomputed:
        raise OfficialTestEvaluationAuditError("Aggregate metrics do not reproduce from per-query table")
    metric_payload = read_json_object(paths["aggregate_metrics"])
    if metric_payload != {"schema_version": 1, "run_id": run["run_id"], **recomputed}:
        raise OfficialTestEvaluationAuditError("Aggregate metrics artifact differs")
    expected_ece = strict_tsv_bytes(tuple(ece_rows[0].keys()), ece_rows)
    expected_risk = strict_tsv_bytes(tuple(risk_rows[0].keys()), risk_rows)
    if paths["ece_bins"].read_bytes() != expected_ece or paths["risk_coverage"].read_bytes() != expected_risk:
        raise OfficialTestEvaluationAuditError("ECE or risk-coverage table is not reproducible")
    git = report.get("git", {})
    if git.get("tracked_worktree_clean") is not True or not COMMIT_RE.fullmatch(str(git.get("commit", ""))):
        raise OfficialTestEvaluationAuditError("Official test Git identity is invalid")
    return recomputed["metric_vector"], artifacts


def paired_t_summary(differences: Sequence[float]) -> Dict[str, Any]:
    from scipy.stats import t as student_t

    if len(differences) != 5 or not all(math.isfinite(value) for value in differences):
        raise OfficialTestEvaluationAuditError("A paired summary requires five finite differences")
    mean = math.fsum(differences) / 5
    variance = math.fsum((value - mean) ** 2 for value in differences) / 4
    standard_deviation = math.sqrt(max(0.0, variance))
    critical = float(student_t.ppf(0.975, 4))
    half_width = critical * standard_deviation / math.sqrt(5)
    if standard_deviation == 0.0:
        raw_p = 1.0 if mean == 0.0 else None
        t_statistic = 0.0 if mean == 0.0 else None
    else:
        t_statistic = mean / (standard_deviation / math.sqrt(5))
        raw_p = float(2.0 * student_t.sf(abs(t_statistic), 4))
    return {
        "paired_differences": list(differences), "mean_difference": mean,
        "median_difference": median(differences), "sample_standard_deviation": standard_deviation,
        "minimum_difference": min(differences), "maximum_difference": max(differences),
        "positive_differences": sum(value > 0.0 for value in differences),
        "zero_differences": sum(value == 0.0 for value in differences),
        "negative_differences": sum(value < 0.0 for value in differences),
        "unadjusted_95_percent_ci_lower": mean - half_width,
        "unadjusted_95_percent_ci_upper": mean + half_width,
        "degrees_of_freedom": 4, "t_statistic": t_statistic, "raw_two_sided_p_value": raw_p,
    }


def holm_adjust(rows: Sequence[Dict[str, Any]], alpha: float = 0.05) -> None:
    available = [row for row in rows if row["raw_two_sided_p_value"] is not None]
    ordered = sorted(available, key=lambda row: (row["raw_two_sided_p_value"], row["dataset"], row["model"]))
    running = 0.0
    m = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = max(running, min(1.0, (m - index) * row["raw_two_sided_p_value"]))
        running = adjusted
        row["holm_adjusted_p_value"] = adjusted
        row["holm_reject_familywise_0_05"] = adjusted <= alpha
        row["holm_order"] = index + 1
    for row in rows:
        if row["raw_two_sided_p_value"] is None:
            row["holm_adjusted_p_value"] = None
            row["holm_reject_familywise_0_05"] = None
            row["holm_order"] = None


def paired_analysis(run_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lookup = {(row["dataset"], row["model"], row["official_replicate"], row["condition"]): row for row in run_rows}
    datasets = ("FB15k-237", "WN18RR", "CoDEx-M")
    models = ("TransE", "DistMult")
    summaries: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for model in models:
            for contrast_id, condition, control, role in CONTRASTS:
                for metric in ANALYSIS_METRICS:
                    condition_values = []
                    control_values = []
                    differences = []
                    for replicate in range(1, 6):
                        condition_value = lookup[(dataset, model, replicate, condition)]["metrics"].get(metric)
                        control_value = lookup[(dataset, model, replicate, control)]["metrics"].get(metric)
                        if condition_value is None or control_value is None:
                            differences = []
                            break
                        condition_value = float(condition_value)
                        control_value = float(control_value)
                        if metric in HIGHER_BETTER:
                            difference = condition_value - control_value
                            orientation = "condition_minus_control"
                        elif metric in LOWER_BETTER:
                            difference = control_value - condition_value
                            orientation = "control_minus_condition"
                        else:
                            difference = condition_value - control_value
                            orientation = "descriptive_condition_minus_control"
                        condition_values.append(condition_value)
                        control_values.append(control_value)
                        differences.append(difference)
                        seed_rows.append({
                            "dataset": dataset, "model": model, "contrast_id": contrast_id,
                            "contrast_role": role, "metric": metric, "official_replicate": replicate,
                            "condition": condition, "control": control,
                            "condition_value": condition_value, "control_value": control_value,
                            "difference_orientation": orientation, "paired_difference": difference,
                        })
                    if not differences:
                        summaries.append({
                            "dataset": dataset, "model": model, "contrast_id": contrast_id,
                            "contrast_role": role, "metric": metric, "condition": condition,
                            "control": control, "status": "not_applicable_undefined_run_metric",
                        })
                        continue
                    summary = paired_t_summary(differences)
                    summaries.append({
                        "dataset": dataset, "model": model, "contrast_id": contrast_id,
                        "contrast_role": role, "metric": metric, "condition": condition,
                        "control": control, "status": "complete_five_seed_paired_summary",
                        "difference_orientation": orientation,
                        "condition_mean": math.fsum(condition_values) / 5,
                        "control_mean": math.fsum(control_values) / 5,
                        **summary,
                    })
    families: List[Dict[str, Any]] = []
    for contrast_id, _, _, role in CONTRASTS[:2]:
        if role != "primary":
            continue
        for metric in PRIMARY_METRICS:
            family_rows = [row for row in summaries if row.get("contrast_id") == contrast_id and row.get("metric") == metric and row.get("status") == "complete_five_seed_paired_summary"]
            if len(family_rows) != 6:
                raise OfficialTestEvaluationAuditError("A confirmatory family does not contain six tests")
            holm_adjust(family_rows)
            families.append({
                "family_id": contrast_id + "__" + metric, "tests": 6,
                "familywise_alpha": 0.05,
                "results": [
                    {key: row[key] for key in (
                        "dataset", "model", "raw_two_sided_p_value", "holm_adjusted_p_value",
                        "holm_reject_familywise_0_05", "holm_order",
                    )}
                    for row in family_rows
                ],
            })
    if len(families) != 6:
        raise OfficialTestEvaluationAuditError("The frozen six Holm families were not produced")
    interpretations = []
    for dataset in datasets:
        for model in models:
            for contrast_id, _, _, _ in CONTRASTS[:2]:
                selected = {row["metric"]: row for row in summaries if row.get("dataset") == dataset and row.get("model") == model and row.get("contrast_id") == contrast_id and row.get("metric") in PRIMARY_METRICS}
                nll = selected["calibrated_test_multiclass_nll"]["mean_difference"]
                eaurc = selected["calibrated_test_eaurc"]["mean_difference"]
                mrr = selected["combined_filtered_test_mrr"]["mean_difference"]
                if nll > 0 and eaurc > 0:
                    category = "directionally consistent evidence"
                elif (nll > 0) != (eaurc > 0):
                    category = "mixed evidence" if (nll < 0 or eaurc < 0) else "partial evidence"
                elif nll == 0 or eaurc == 0:
                    category = "partial evidence"
                else:
                    category = "no primary evidence"
                interpretations.append({
                    "dataset": dataset, "model": model, "contrast_id": contrast_id,
                    "permitted_primary_category": category,
                    "mean_calibrated_nll_improvement": nll,
                    "mean_calibrated_eaurc_improvement": eaurc,
                    "mean_mrr_condition_minus_control": mrr,
                    "mrr_guardrail": -0.01,
                    "accuracy_uncertainty_tradeoff": (nll > 0 or eaurc > 0) and mrr < -0.01,
                    "supported_uncertainty_improvement_permitted": False,
                })
    return {
        "analysis_configuration": {
            "analysis_level": "separate_dataset_model_unit",
            "paired_unit": "official_training_seed",
            "replicates_per_unit": 5,
            "contrasts": [
                {"contrast_id": row[0], "condition": row[1], "control": row[2], "role": row[3]}
                for row in CONTRASTS
            ],
            "higher_is_better": sorted(HIGHER_BETTER),
            "lower_is_better": sorted(LOWER_BETTER),
            "descriptive_without_automatic_improvement_direction": sorted(DESCRIPTIVE),
            "confidence_interval": "two_sided_unadjusted_95_percent_student_t_df_4",
            "confirmatory_test": "two_sided_paired_t_test",
            "multiplicity_adjustment": "Holm_within_each_of_six_families",
            "familywise_alpha": 0.05,
            "official_seed_derivations_source": "configs/official_experiment_design.json",
            "failure_and_retry_records": [
                dict(record) for record in FAILED_OFFICIAL_TEST_ATTEMPTS
            ],
        },
        "seed_level_paired_differences": seed_rows,
        "summaries": summaries,
        "holm_families": families,
        "interpretations": interpretations,
    }


def audit_official_test_evaluations(
    *, design_path: Path, checkpoint_lock_path: Path, evaluation_lock_path: Path,
    calibration_lock_path: Path, claims_lock_path: Path, project_root: Path, scratch: Path,
    output_lock: Path, output_report: Path, output_directory: Path,
) -> Dict[str, Any]:
    execution = require_scheduled_execution(scratch.resolve())
    frozen = {
        design_path: OFFICIAL_DESIGN_SHA256,
        checkpoint_lock_path: OFFICIAL_CHECKPOINT_LOCK_SHA256,
        evaluation_lock_path: OFFICIAL_EVALUATION_LOCK_SHA256,
        calibration_lock_path: OFFICIAL_CALIBRATION_LOCK_SHA256,
        claims_lock_path: OFFICIAL_CLAIMS_LOCK_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_file(path) != expected:
            raise OfficialTestEvaluationAuditError("Frozen input hash differs: " + str(path))
    design = read_json_object(design_path)
    validate_design_invariants(design)
    checkpoint_lock = read_json_object(checkpoint_lock_path)
    evaluation_lock = read_json_object(evaluation_lock_path)
    calibration_lock = read_json_object(calibration_lock_path)
    claims_lock = read_json_object(claims_lock_path)
    validate_evaluation_lock(evaluation_lock)
    if claims_lock.get("confirmatory_testing", {}).get("replicates_per_unit") != 5:
        raise OfficialTestEvaluationAuditError("Frozen claims analysis differs")
    run_rows = []
    commits = set()
    artifact_locks = []
    for run in design["runs"]:
        selected, locked = select_run_and_checkpoint(design, checkpoint_lock, int(run["array_index"]))
        temperature_row = select_temperature(calibration_lock, run, locked)
        path = official_test_report_path(project_root, run)
        if not path.is_file():
            raise OfficialTestEvaluationAuditError("Official test report is missing: " + str(path))
        report = read_json_object(path)
        metric_vector, artifacts = validate_test_report(report, selected, locked, temperature_row, project_root)
        commits.add(report["git"]["commit"])
        run_rows.append({
            "array_index": run["array_index"], "run_id": run["run_id"],
            "paired_block_id": run["paired_block_id"], "dataset": run["dataset"],
            "model": run["model"], "condition": run["condition"],
            "official_replicate": run["official_replicate"], "official_seed": run["official_seed"],
            "metrics": metric_vector,
        })
        artifact_locks.append({
            "run_id": run["run_id"], "report": {"path": str(path.relative_to(project_root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size},
            "artifacts": artifacts,
        })
    if len(run_rows) != EXPECTED_RUNS or len(commits) != 1:
        raise OfficialTestEvaluationAuditError("Official test run count or code commit differs")
    block_conditions: Dict[str, set] = defaultdict(set)
    for row in run_rows:
        block_conditions[row["paired_block_id"]].add(row["condition"])
    if len(block_conditions) != 30 or any(value != {"random", "correct_text", "shuffled_text"} for value in block_conditions.values()):
        raise OfficialTestEvaluationAuditError("Official paired blocks are incomplete")
    analysis = paired_analysis(run_rows)
    run_columns = ("array_index", "run_id", "paired_block_id", "dataset", "model", "condition", "official_replicate", "official_seed") + ANALYSIS_METRICS
    flat_runs = [{**{key: row[key] for key in run_columns[:8]}, **{key: row["metrics"].get(key) for key in ANALYSIS_METRICS}} for row in run_rows]
    seed_rows = analysis["seed_level_paired_differences"]
    seed_columns = tuple(seed_rows[0].keys())
    summary_columns = (
        "dataset", "model", "contrast_id", "contrast_role", "metric", "condition",
        "control", "status", "difference_orientation", "condition_mean", "control_mean",
        "mean_difference", "median_difference", "sample_standard_deviation",
        "minimum_difference", "maximum_difference", "positive_differences",
        "zero_differences", "negative_differences", "unadjusted_95_percent_ci_lower",
        "unadjusted_95_percent_ci_upper", "degrees_of_freedom", "t_statistic",
        "raw_two_sided_p_value", "holm_adjusted_p_value",
        "holm_reject_familywise_0_05", "holm_order",
    )
    summary_rows = [
        {column: row.get(column) for column in summary_columns}
        for row in analysis["summaries"]
    ]
    holm_rows = []
    for family in analysis["holm_families"]:
        for row in family["results"]:
            holm_rows.append(
                {
                    "family_id": family["family_id"],
                    "familywise_alpha": family["familywise_alpha"],
                    **row,
                }
            )
    holm_columns = tuple(holm_rows[0].keys())
    result_artifacts = {
        "per_run_metrics": artifact_record(output_directory / "per_run_metrics.tsv", project_root, strict_tsv_bytes(run_columns, flat_runs)),
        "paired_differences": artifact_record(output_directory / "paired_differences.tsv", project_root, strict_tsv_bytes(seed_columns, seed_rows)),
        "summary_statistics_and_confidence_intervals": artifact_record(
            output_directory / "summary_statistics_and_confidence_intervals.tsv",
            project_root,
            strict_tsv_bytes(summary_columns, summary_rows),
        ),
        "holm_adjusted_p_values": artifact_record(
            output_directory / "holm_adjusted_p_values.tsv",
            project_root,
            strict_tsv_bytes(holm_columns, holm_rows),
        ),
        "statistical_analysis": artifact_record(output_directory / "statistical_analysis.json", project_root, canonical_json_bytes({"schema_version": 1, **analysis})),
    }
    lock = {
        "schema_version": 1,
        "status": "official_test_results_audited_and_frozen",
        "official_result": True,
        "test_data_accessed": True,
        "test_metrics_computed": True,
        "official_runs": 90,
        "paired_blocks": 30,
        "official_test_git_commit": next(iter(commits)),
        "official_design_sha256": OFFICIAL_DESIGN_SHA256,
        "official_checkpoint_lock_sha256": OFFICIAL_CHECKPOINT_LOCK_SHA256,
        "official_evaluation_lock_sha256": OFFICIAL_EVALUATION_LOCK_SHA256,
        "official_calibration_lock_sha256": OFFICIAL_CALIBRATION_LOCK_SHA256,
        "official_claims_lock_sha256": OFFICIAL_CLAIMS_LOCK_SHA256,
        "run_metrics_sha256": hashlib.sha256(canonical_json_bytes(run_rows)).hexdigest(),
        "official_artifacts_sha256": hashlib.sha256(canonical_json_bytes(artifact_locks)).hexdigest(),
        "result_artifacts": result_artifacts,
        "runs": artifact_locks,
    }
    lock_bytes = canonical_json_bytes(lock)
    write_new_or_identical(output_lock, lock_bytes)
    report = {
        "schema_version": 1,
        "status": "gate_i_official_test_evaluation_and_statistics_passed",
        "official_result": True,
        "test_data_accessed": True,
        "official_runs": 90,
        "paired_blocks": 30,
        "holm_families": 6,
        "official_test_result_lock": {"path": str(output_lock.relative_to(project_root)), "sha256": hashlib.sha256(lock_bytes).hexdigest(), "size_bytes": len(lock_bytes)},
        "result_artifacts": result_artifacts,
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
    parser.add_argument("--calibration-lock", type=Path, required=True)
    parser.add_argument("--claims-lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_official_test_evaluations(
        design_path=args.design.resolve(), checkpoint_lock_path=args.checkpoint_lock.resolve(),
        evaluation_lock_path=args.evaluation_lock.resolve(), calibration_lock_path=args.calibration_lock.resolve(),
        claims_lock_path=args.claims_lock.resolve(), project_root=args.project_root.resolve(),
        scratch=args.scratch.resolve(), output_lock=args.output_lock.resolve(),
        output_report=args.output_report.resolve(), output_directory=args.output_directory.resolve(),
    )
    print("Gate I official test audit recorded in {}".format(args.output_report.resolve()))
    print(report["official_test_result_lock"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
