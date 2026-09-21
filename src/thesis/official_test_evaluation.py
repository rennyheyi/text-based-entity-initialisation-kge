"""Evaluate one frozen official checkpoint on the test split exactly once."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .official_calibration import (
    EXPECTED_DEVICE,
    EXPECTED_RUNS,
    OFFICIAL_CHECKPOINT_LOCK_SHA256,
    OFFICIAL_DESIGN_SHA256,
    OFFICIAL_EVALUATION_LOCK_SHA256,
    _load_checkpoint,
    select_run_and_checkpoint,
    validate_evaluation_lock,
)
from .official_design import (
    canonical_json_bytes,
    git_state,
    read_json_object,
    require_scheduled_execution,
    sha256_file,
    write_new_or_identical,
)
from .official_evaluation_core import query_probability_records, test_score_matrix


OFFICIAL_CALIBRATION_LOCK_SHA256 = (
    "6dac32077c03e97d10ad28168c31ef39d88e1da18f4774d00969559675c51cd1"
)
OFFICIAL_CLAIMS_LOCK_SHA256 = (
    "b9c95461918502934ae43d5b17261361c98e400909658821ee1293e4d694174e"
)
ECE_BIN_COUNT = 15


class OfficialTestEvaluationError(RuntimeError):
    """Raised when an official test evaluation invariant is violated."""


PER_QUERY_COLUMNS = (
    "dataset",
    "model",
    "condition",
    "official_replicate",
    "split",
    "query_id",
    "direction",
    "head_entity_index",
    "relation_index",
    "tail_entity_index",
    "target_entity_index",
    "target_seen_status",
    "triple_endpoint_seen_status",
    "filtered_candidate_count",
    "target_score",
    "realistic_target_rank",
    "predicted_entity_index",
    "maximum_score",
    "top_score_tie_count",
    "top_label_correct",
    "raw_target_probability",
    "calibrated_target_probability",
    "raw_confidence",
    "calibrated_confidence",
    "raw_nll",
    "calibrated_nll",
    "raw_normalised_entropy",
    "calibrated_normalised_entropy",
    "fitted_temperature",
)


def strict_tsv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = ["\t".join(columns)]
    for row in rows:
        fields = []
        for column in columns:
            value = row[column]
            if value is None:
                field = "NA"
            elif isinstance(value, bool):
                field = str(value).lower()
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise OfficialTestEvaluationError("TSV contains a non-finite float")
                field = repr(value)
            else:
                field = str(value)
            if any(character in field for character in ("\t", "\n", "\r")):
                raise OfficialTestEvaluationError("TSV contains an unsafe field")
            fields.append(field)
        lines.append("\t".join(fields))
    return ("\n".join(lines) + "\n").encode("utf-8")


def artifact_record(path: Path, project_root: Path, payload: bytes) -> Dict[str, Any]:
    write_new_or_identical(path, payload)
    return {
        "path": str(path.relative_to(project_root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def ranking_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"queries": 0, "mrr": None, "hits_at_1": None, "hits_at_3": None, "hits_at_10": None}
    ranks = [float(row["realistic_target_rank"]) for row in rows]
    return {
        "queries": len(rows),
        "mrr": math.fsum(1.0 / rank for rank in ranks) / len(ranks),
        "hits_at_1": math.fsum(rank <= 1.0 for rank in ranks) / len(ranks),
        "hits_at_3": math.fsum(rank <= 3.0 for rank in ranks) / len(ranks),
        "hits_at_10": math.fsum(rank <= 10.0 for rank in ranks) / len(ranks),
    }


def ece_table(
    rows: Sequence[Mapping[str, Any]], prefix: str, bins: int = ECE_BIN_COUNT
) -> Tuple[List[Dict[str, Any]], float]:
    if not rows or bins <= 0:
        raise OfficialTestEvaluationError("ECE requires queries and positive bin count")
    grouped: List[List[Mapping[str, Any]]] = [[] for _ in range(bins)]
    key = prefix + "_confidence"
    for row in rows:
        confidence = float(row[key])
        if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
            raise OfficialTestEvaluationError("ECE confidence is outside [0,1]")
        grouped[min(int(confidence * bins), bins - 1)].append(row)
    table = []
    contributions = []
    for index, members in enumerate(grouped):
        count = len(members)
        mean_confidence = (
            math.fsum(float(row[key]) for row in members) / count if count else None
        )
        accuracy = (
            math.fsum(bool(row["top_label_correct"]) for row in members) / count
            if count
            else None
        )
        gap = abs(accuracy - mean_confidence) if count else None
        contribution = count / len(rows) * gap if count else 0.0
        contributions.append(contribution)
        table.append(
            {
                "probability": prefix,
                "bin_index": index + 1,
                "lower_inclusive": index / bins,
                "upper_exclusive_or_final_inclusive": (index + 1) / bins,
                "final_bin_includes_one": index == bins - 1,
                "queries": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": accuracy,
                "absolute_calibration_gap": gap,
                "weighted_ece_contribution": contribution,
            }
        )
    return table, math.fsum(contributions)


def binary_auroc(confidences: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    if len(confidences) != len(labels) or not confidences:
        raise OfficialTestEvaluationError("AUROC inputs are empty or misaligned")
    positives = sum(bool(value) for value in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(range(len(confidences)), key=lambda index: (confidences[index], index))
    ranks = [0.0] * len(confidences)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and confidences[ordered[stop]] == confidences[ordered[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[ordered[position]] = average_rank
        start = stop
    positive_rank_sum = math.fsum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def risk_coverage_table(
    rows: Sequence[Mapping[str, Any]], prefix: str
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not rows:
        raise OfficialTestEvaluationError("Risk-coverage requires at least one query")
    confidence_key = prefix + "_confidence"
    ordered = sorted(
        rows,
        key=lambda row: (-float(row[confidence_key]), str(row["query_id"])),
    )
    correct_total = sum(bool(row["top_label_correct"]) for row in rows)
    errors = 0
    table = []
    risks = []
    oracle_risks = []
    for index, row in enumerate(ordered, 1):
        errors += not bool(row["top_label_correct"])
        risk = errors / index
        oracle_errors = max(0, index - correct_total)
        oracle_risk = oracle_errors / index
        risks.append(risk)
        oracle_risks.append(oracle_risk)
        table.append(
            {
                "probability": prefix,
                "retained_queries": index,
                "coverage": index / len(rows),
                "query_id_at_boundary": row["query_id"],
                "confidence_at_boundary": float(row[confidence_key]),
                "cumulative_errors": errors,
                "selective_risk": risk,
                "oracle_selective_risk": oracle_risk,
            }
        )
    aurc = math.fsum(risks) / len(risks)
    oracle_aurc = math.fsum(oracle_risks) / len(oracle_risks)
    return table, {
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "eaurc": aurc - oracle_aurc,
    }


def aggregate_query_metrics(
    rows: Sequence[Mapping[str, Any]], fitted_temperature: float
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        raise OfficialTestEvaluationError("No test query rows were supplied")
    expected = {"head", "tail"}
    if set(str(row["direction"]) for row in rows) != expected:
        raise OfficialTestEvaluationError("Both test prediction directions are required")
    head = [row for row in rows if row["direction"] == "head"]
    tail = [row for row in rows if row["direction"] == "tail"]
    raw_ece_rows, raw_ece = ece_table(rows, "raw")
    calibrated_ece_rows, calibrated_ece = ece_table(rows, "calibrated")
    raw_risk_rows, raw_risk = risk_coverage_table(rows, "raw")
    calibrated_risk_rows, calibrated_risk = risk_coverage_table(rows, "calibrated")
    labels = [bool(row["top_label_correct"]) for row in rows]
    raw_confidence = [float(row["raw_confidence"]) for row in rows]
    calibrated_confidence = [float(row["calibrated_confidence"]) for row in rows]

    ranking = {
        "head": ranking_summary(head),
        "tail": ranking_summary(tail),
        "combined": ranking_summary(rows),
        "target_seen": ranking_summary([row for row in rows if row["target_seen_status"] == "seen"]),
        "target_unseen": ranking_summary([row for row in rows if row["target_seen_status"] == "unseen"]),
        "both_endpoints_seen": ranking_summary(
            [row for row in rows if row["triple_endpoint_seen_status"] == "both_seen"]
        ),
        "at_least_one_endpoint_unseen": ranking_summary(
            [row for row in rows if row["triple_endpoint_seen_status"] == "at_least_one_unseen"]
        ),
    }
    uncertainty = {
        "top_label_accuracy": math.fsum(labels) / len(rows),
        "raw_test_multiclass_nll": math.fsum(float(row["raw_nll"]) for row in rows) / len(rows),
        "calibrated_test_multiclass_nll": math.fsum(float(row["calibrated_nll"]) for row in rows) / len(rows),
        "raw_test_ece": raw_ece,
        "calibrated_test_ece": calibrated_ece,
        "raw_test_top_label_brier": math.fsum(
            (confidence - int(label)) ** 2 for confidence, label in zip(raw_confidence, labels)
        ) / len(rows),
        "calibrated_test_top_label_brier": math.fsum(
            (confidence - int(label)) ** 2 for confidence, label in zip(calibrated_confidence, labels)
        ) / len(rows),
        "raw_test_auroc": binary_auroc(raw_confidence, labels),
        "calibrated_test_auroc": binary_auroc(calibrated_confidence, labels),
        "raw_test_aurc": raw_risk["aurc"],
        "calibrated_test_aurc": calibrated_risk["aurc"],
        "raw_test_oracle_aurc": raw_risk["oracle_aurc"],
        "calibrated_test_oracle_aurc": calibrated_risk["oracle_aurc"],
        "raw_test_eaurc": raw_risk["eaurc"],
        "calibrated_test_eaurc": calibrated_risk["eaurc"],
        "raw_test_normalised_entropy": math.fsum(
            float(row["raw_normalised_entropy"]) for row in rows
        ) / len(rows),
        "calibrated_test_normalised_entropy": math.fsum(
            float(row["calibrated_normalised_entropy"]) for row in rows
        ) / len(rows),
        "single_candidate_queries": sum(int(row["filtered_candidate_count"]) == 1 for row in rows),
        "fitted_temperature": fitted_temperature,
    }
    combined = ranking["combined"]
    metric_vector = {
        "combined_filtered_test_mrr": combined["mrr"],
        "test_hits_at_1": combined["hits_at_1"],
        "test_hits_at_3": combined["hits_at_3"],
        "test_hits_at_10": combined["hits_at_10"],
        **uncertainty,
    }
    metrics = {
        "queries": len(rows),
        "ranking": ranking,
        "uncertainty": uncertainty,
        "metric_vector": metric_vector,
    }
    return metrics, raw_ece_rows + calibrated_ece_rows, raw_risk_rows + calibrated_risk_rows


def select_temperature(
    calibration_lock: Mapping[str, Any], run: Mapping[str, Any], locked: Mapping[str, Any]
) -> Mapping[str, Any]:
    if (
        calibration_lock.get("status") != "official_temperatures_frozen_pending_test_evaluation"
        or calibration_lock.get("official_result") is not False
        or calibration_lock.get("test_data_accessed") is not False
        or calibration_lock.get("test_metrics_computed") is not False
        or calibration_lock.get("calibration_runs") != EXPECTED_RUNS
        or calibration_lock.get("official_design_sha256") != OFFICIAL_DESIGN_SHA256
        or calibration_lock.get("official_checkpoint_lock_sha256") != OFFICIAL_CHECKPOINT_LOCK_SHA256
        or calibration_lock.get("official_evaluation_lock_sha256") != OFFICIAL_EVALUATION_LOCK_SHA256
    ):
        raise OfficialTestEvaluationError("Official calibration lock boundary differs")
    matches = [row for row in calibration_lock.get("temperatures", ()) if row.get("run_id") == run["run_id"]]
    if len(matches) != 1:
        raise OfficialTestEvaluationError("Frozen temperature identity is not unique")
    row = matches[0]
    exact = {
        "array_index": run["array_index"],
        "paired_block_id": run["paired_block_id"],
        "dataset": run["dataset"],
        "model": run["model"],
        "condition": run["condition"],
        "official_replicate": run["official_replicate"],
        "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "checkpoint_sha256": locked["checkpoint"]["sha256"],
    }
    for key, expected in exact.items():
        if row.get(key) != expected:
            raise OfficialTestEvaluationError("Frozen temperature field differs: " + key)
    fitted = row.get("temperature", {}).get("fitted_temperature")
    if not isinstance(fitted, (int, float)) or not math.isfinite(float(fitted)) or not 0.001 <= fitted <= 1000.0:
        raise OfficialTestEvaluationError("Frozen fitted temperature is invalid")
    return row


def build_query_rows(
    run: Mapping[str, Any], test: Any, train: Any, counts: Any,
    diagnostics: Mapping[str, Any], temperature: float, entity_count: int,
) -> List[Dict[str, Any]]:
    query_count = 2 * len(test)
    if any(len(values) != query_count for values in diagnostics.values()):
        raise OfficialTestEvaluationError("Test query diagnostics are misaligned")
    seen = {int(value) for row in train for value in (row[0], row[2])}
    counts_list = [int(value) for value in counts.detach().cpu().tolist()]
    rows = []
    for query_index in range(query_count):
        direction = "head" if query_index < len(test) else "tail"
        triple_index = query_index if direction == "head" else query_index - len(test)
        head, relation, tail = map(int, test[triple_index])
        target = head if direction == "head" else tail
        row = {
            "dataset": run["dataset"],
            "model": run["model"],
            "condition": run["condition"],
            "official_replicate": run["official_replicate"],
            "split": "test",
            "query_id": "{}|test|{}|{:08d}".format(run["dataset"], direction, triple_index),
            "direction": direction,
            "head_entity_index": head,
            "relation_index": relation,
            "tail_entity_index": tail,
            "target_entity_index": target,
            "target_seen_status": "seen" if target in seen else "unseen",
            "triple_endpoint_seen_status": "both_seen" if head in seen and tail in seen else "at_least_one_unseen",
            "filtered_candidate_count": counts_list[query_index],
            "target_score": float(diagnostics["target_score"][query_index]),
            "realistic_target_rank": float(diagnostics["rank"][query_index]),
            "predicted_entity_index": int(diagnostics["predicted_index"][query_index]),
            "maximum_score": float(diagnostics["maximum_score"][query_index]),
            "top_score_tie_count": int(diagnostics["top_score_tie_count"][query_index]),
            "top_label_correct": bool(diagnostics["correct"][query_index]),
            "raw_target_probability": float(diagnostics["raw_target_probability"][query_index]),
            "calibrated_target_probability": float(diagnostics["calibrated_target_probability"][query_index]),
            "raw_confidence": float(diagnostics["raw_confidence"][query_index]),
            "calibrated_confidence": float(diagnostics["calibrated_confidence"][query_index]),
            "raw_nll": float(diagnostics["raw_nll"][query_index]),
            "calibrated_nll": float(diagnostics["calibrated_nll"][query_index]),
            "raw_normalised_entropy": float(diagnostics["raw_normalised_entropy"][query_index]),
            "calibrated_normalised_entropy": float(diagnostics["calibrated_normalised_entropy"][query_index]),
            "fitted_temperature": temperature,
        }
        if row["target_entity_index"] < 0 or row["target_entity_index"] >= entity_count:
            raise OfficialTestEvaluationError("Target is absent from candidate vocabulary")
        rows.append(row)
    if len({row["query_id"] for row in rows}) != query_count:
        raise OfficialTestEvaluationError("Test query identifiers are not unique")
    return rows


def load_official_test_data(
    project_root: Path,
    dataset: str,
    expected_manifest_sha256: str,
) -> Dict[str, Any]:
    """Extend the HPO-safe train/valid loader with the frozen test split."""

    import numpy as np
    from .hpo_stage1 import load_indexed_tsv, load_stage_data, triple_keys

    data = load_stage_data(project_root, dataset, expected_manifest_sha256)
    manifest = read_json_object(data["manifest_path"])
    processed_hashes = manifest.get("processed_sha256")
    if not isinstance(processed_hashes, dict):
        raise OfficialTestEvaluationError("Dataset manifest lacks processed hashes")
    test_path = project_root / "data" / "processed" / dataset / "test.tsv"
    expected_test_sha256 = processed_hashes.get("test.tsv")
    if not isinstance(expected_test_sha256, str) or sha256_file(test_path) != expected_test_sha256:
        raise OfficialTestEvaluationError("Frozen test split hash differs")
    test = load_indexed_tsv(test_path)
    if (
        test.ndim != 2
        or test.shape[1] != 3
        or len(test) <= 0
        or int(test[:, [0, 2]].min()) < 0
        or int(test[:, [0, 2]].max()) >= int(data["entity_count"])
        or int(test[:, 1].min()) < 0
        or int(test[:, 1].max()) >= int(data["relation_count"])
    ):
        raise OfficialTestEvaluationError("Test split indices are invalid")
    test_keys = np.sort(
        triple_keys(
            test,
            entity_count=int(data["entity_count"]),
            relation_count=int(data["relation_count"]),
        )
    )
    if len(np.unique(test_keys)) != len(test):
        raise OfficialTestEvaluationError("Test split contains duplicate triples")
    return {**data, "test": test, "sorted_test_keys": test_keys}


def run_official_test_evaluation(
    *, design_path: Path, checkpoint_lock_path: Path, evaluation_lock_path: Path,
    calibration_lock_path: Path, claims_lock_path: Path, array_index: int,
    project_root: Path, scratch: Path,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from .official_training import configure_numerics

    execution = require_scheduled_execution(scratch.resolve())
    expected_hashes = {
        design_path: OFFICIAL_DESIGN_SHA256,
        checkpoint_lock_path: OFFICIAL_CHECKPOINT_LOCK_SHA256,
        evaluation_lock_path: OFFICIAL_EVALUATION_LOCK_SHA256,
        calibration_lock_path: OFFICIAL_CALIBRATION_LOCK_SHA256,
        claims_lock_path: OFFICIAL_CLAIMS_LOCK_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise OfficialTestEvaluationError("Frozen input hash differs: " + str(path))
    design = read_json_object(design_path)
    checkpoint_lock = read_json_object(checkpoint_lock_path)
    evaluation_lock = read_json_object(evaluation_lock_path)
    calibration_lock = read_json_object(calibration_lock_path)
    claims_lock = read_json_object(claims_lock_path)
    validate_evaluation_lock(evaluation_lock)
    if claims_lock.get("status") != "official_claims_frozen_pending_official_design" or claims_lock.get("information_boundary", {}).get("test_data_accessed") is not False:
        raise OfficialTestEvaluationError("Official claims lock boundary differs")
    run, locked = select_run_and_checkpoint(design, checkpoint_lock, array_index)
    temperature_row = select_temperature(calibration_lock, run, locked)
    temperature = float(temperature_row["temperature"]["fitted_temperature"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise OfficialTestEvaluationError("Exactly one scheduler-visible CUDA device is required")
    numerical_policy = configure_numerics(torch)
    if torch.cuda.get_device_name(0) != EXPECTED_DEVICE:
        raise OfficialTestEvaluationError("Test evaluation did not use the frozen H200 device")

    dataset = str(run["dataset"])
    data = load_official_test_data(
        project_root,
        dataset,
        design["dataset_resources"][dataset]["dataset_manifest_sha256"],
    )
    payload = _load_checkpoint(project_root, run, locked)
    entity = payload["entity_embeddings"].to(device="cuda", dtype=torch.float32)
    relation = payload["relation_embeddings"].to(device="cuda", dtype=torch.float32)
    del payload
    parameters = run["selected_configuration"]["parameters"]
    torch.cuda.reset_peak_memory_stats()
    scores, target_scores, counts, target_indices = test_score_matrix(
        str(run["model"]), entity, relation, data["train"], data["valid"], data["test"],
        transe_distance_norm=int(parameters.get("transe_distance_norm", 1)),
    )
    diagnostics = query_probability_records(scores, target_scores, target_indices, counts, temperature)
    rows = build_query_rows(
        run, data["test"], data["train"], counts, diagnostics, temperature,
        int(data["entity_count"]),
    )
    metrics, ece_rows, risk_rows = aggregate_query_metrics(rows, temperature)

    root = project_root / "artifacts" / "official_test" / dataset / str(run["model"]) / "replicate_{:02d}".format(int(run["official_replicate"])) / str(run["condition"])
    per_query = artifact_record(root / "per_query.tsv", project_root, strict_tsv_bytes(PER_QUERY_COLUMNS, rows))
    ece_columns = tuple(ece_rows[0].keys())
    ece = artifact_record(root / "ece_bins.tsv", project_root, strict_tsv_bytes(ece_columns, ece_rows))
    risk_columns = tuple(risk_rows[0].keys())
    risk = artifact_record(root / "risk_coverage.tsv", project_root, strict_tsv_bytes(risk_columns, risk_rows))
    metric_payload = canonical_json_bytes({"schema_version": 1, "run_id": run["run_id"], **metrics})
    metric_artifact = artifact_record(root / "metrics.json", project_root, metric_payload)
    report = {
        "schema_version": 1,
        "status": "official_test_evaluation_completed_pending_audit",
        "official_result": True,
        "validation_data_accessed": True,
        "test_data_accessed": True,
        "test_metrics_computed": True,
        "array_index": array_index,
        "run_id": run["run_id"],
        "paired_block_id": run["paired_block_id"],
        "dataset": dataset,
        "model": run["model"],
        "condition": run["condition"],
        "official_replicate": run["official_replicate"],
        "official_seed": run["official_seed"],
        "run_configuration_sha256": run["run_configuration_sha256"],
        "checkpoint_sha256": locked["checkpoint"]["sha256"],
        "fitted_temperature": temperature,
        "test": {
            "triples": int(len(data["test"])), "head_queries": int(len(data["test"])),
            "tail_queries": int(len(data["test"])), "combined_queries": int(2 * len(data["test"])),
            "candidate_vocabulary": int(data["entity_count"]),
            "minimum_filtered_candidate_count": int(counts.min().cpu()),
            "maximum_filtered_candidate_count": int(counts.max().cpu()),
            "filtering_splits": ["train", "validation", "test"],
        },
        "metrics": metrics,
        "artifacts": {"per_query_table": per_query, "aggregate_metrics": metric_artifact, "ece_bins": ece, "risk_coverage": risk},
        "official_design_sha256": OFFICIAL_DESIGN_SHA256,
        "official_checkpoint_lock_sha256": OFFICIAL_CHECKPOINT_LOCK_SHA256,
        "official_evaluation_lock_sha256": OFFICIAL_EVALUATION_LOCK_SHA256,
        "official_calibration_lock_sha256": OFFICIAL_CALIBRATION_LOCK_SHA256,
        "official_claims_lock_sha256": OFFICIAL_CLAIMS_LOCK_SHA256,
        "execution": {**execution, "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")},
        "numerical_policy": numerical_policy,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device_name": torch.cuda.get_device_name(0), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
        "git": git_state(project_root),
    }
    report_path = project_root / "reports" / "official_test" / dataset / str(run["model"]) / "replicate_{:02d}".format(int(run["official_replicate"])) / (str(run["condition"]) + ".json")
    write_new_or_identical(report_path, canonical_json_bytes(report))
    del scores, target_scores, counts, target_indices, entity, relation
    torch.cuda.empty_cache()
    gc.collect()
    return {**report, "report_path": str(report_path.relative_to(project_root))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--checkpoint-lock", type=Path, required=True)
    parser.add_argument("--evaluation-lock", type=Path, required=True)
    parser.add_argument("--calibration-lock", type=Path, required=True)
    parser.add_argument("--claims-lock", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_official_test_evaluation(
        design_path=args.design.resolve(), checkpoint_lock_path=args.checkpoint_lock.resolve(),
        evaluation_lock_path=args.evaluation_lock.resolve(), calibration_lock_path=args.calibration_lock.resolve(),
        claims_lock_path=args.claims_lock.resolve(), array_index=args.array_index,
        project_root=args.project_root.resolve(), scratch=args.scratch.resolve(),
    )
    print("Official test evaluation completed for {}".format(report["run_id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
