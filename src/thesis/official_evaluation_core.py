"""Frozen filtered-scoring and validation-temperature primitives."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple


class OfficialEvaluationError(RuntimeError):
    """Raised when an official evaluation invariant is violated."""


TEMPERATURE_LOWER = 1e-3
TEMPERATURE_UPPER = 1e3
LOG_TEMPERATURE_XATOL = 1e-6
TEMPERATURE_MAXITER = 100
QUERY_BATCH_SIZE = 32
CANDIDATE_CHUNK_SIZE = 4096


def build_filter_maps(
    *splits: Any,
) -> Tuple[Mapping[Tuple[int, int], Set[int]], Mapping[Tuple[int, int], Set[int]]]:
    tail_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    head_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for split in splits:
        for row in split:
            head, relation, tail = map(int, row)
            tail_map[(head, relation)].add(tail)
            head_map[(relation, tail)].add(head)
    return tail_map, head_map


def _direction_score_matrix(
    model: str,
    entity: Any,
    relation: Any,
    triples_numpy: Any,
    filter_map: Mapping[Tuple[int, int], Set[int]],
    *,
    direction: str,
    transe_distance_norm: int,
    query_batch_size: int,
    candidate_chunk_size: int,
) -> Tuple[Any, Any, Any]:
    import numpy as np
    import torch
    from .hpo_stage1 import score_candidate_block

    if direction not in {"head", "tail"}:
        raise OfficialEvaluationError("Prediction direction must be head or tail")
    entity_count = int(entity.shape[0])
    query_count = int(len(triples_numpy))
    scores = torch.empty(
        (query_count, entity_count), dtype=torch.float32, device=entity.device
    )
    target_indices_numpy = (
        triples_numpy[:, 0] if direction == "head" else triples_numpy[:, 2]
    ).astype(np.int64, copy=False)
    with torch.inference_mode():
        for query_start in range(0, query_count, query_batch_size):
            query_stop = min(query_start + query_batch_size, query_count)
            batch_numpy = triples_numpy[query_start:query_stop]
            triples = torch.as_tensor(
                batch_numpy, dtype=torch.long, device=entity.device
            )
            for candidate_start in range(0, entity_count, candidate_chunk_size):
                candidate_stop = min(
                    candidate_start + candidate_chunk_size, entity_count
                )
                block = score_candidate_block(
                    model,
                    entity,
                    relation,
                    triples,
                    direction=direction,
                    candidate_start=candidate_start,
                    candidate_stop=candidate_stop,
                    transe_distance_norm=transe_distance_norm,
                )
                if block.dtype != torch.float32 or not torch.isfinite(block).all():
                    raise OfficialEvaluationError(
                        "Candidate scores are not finite float32 values"
                    )
                scores[
                    query_start:query_stop, candidate_start:candidate_stop
                ] = block
    if not torch.isfinite(scores).all():
        raise OfficialEvaluationError("Candidate score matrix contains a non-finite value")

    target_indices = torch.as_tensor(
        target_indices_numpy, dtype=torch.long, device=entity.device
    )
    rows = torch.arange(query_count, dtype=torch.long, device=entity.device)
    target_scores = scores[rows, target_indices].clone()
    candidate_counts = torch.full(
        (query_count,), entity_count, dtype=torch.int64, device=entity.device
    )
    for query_index, row in enumerate(triples_numpy):
        head, relation_id, tail = map(int, row)
        key = (
            (relation_id, tail) if direction == "head" else (head, relation_id)
        )
        target = int(target_indices_numpy[query_index])
        filtered = sorted(
            int(index) for index in filter_map[key] if int(index) != target
        )
        if filtered:
            indices = torch.as_tensor(
                filtered, dtype=torch.long, device=entity.device
            )
            scores[query_index, indices] = -torch.inf
            candidate_counts[query_index] -= len(filtered)
        scores[query_index, target] = target_scores[query_index]
    if torch.any(candidate_counts <= 0):
        raise OfficialEvaluationError("A filtered candidate set is empty")
    if not torch.isfinite(scores[rows, target_indices]).all():
        raise OfficialEvaluationError("A target was lost during filtering")
    return scores, target_scores, candidate_counts


def validation_score_matrix(
    model: str,
    entity: Any,
    relation: Any,
    train: Any,
    valid: Any,
    *,
    transe_distance_norm: int,
    query_batch_size: int = QUERY_BATCH_SIZE,
    candidate_chunk_size: int = CANDIDATE_CHUNK_SIZE,
) -> Tuple[Any, Any, Any]:
    import torch

    tail_map, head_map = build_filter_maps(train, valid)
    head_scores, head_targets, head_counts = _direction_score_matrix(
        model,
        entity,
        relation,
        valid,
        head_map,
        direction="head",
        transe_distance_norm=transe_distance_norm,
        query_batch_size=query_batch_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    tail_scores, tail_targets, tail_counts = _direction_score_matrix(
        model,
        entity,
        relation,
        valid,
        tail_map,
        direction="tail",
        transe_distance_norm=transe_distance_norm,
        query_batch_size=query_batch_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    scores = torch.cat((head_scores, tail_scores), dim=0)
    targets = torch.cat((head_targets, tail_targets), dim=0)
    counts = torch.cat((head_counts, tail_counts), dim=0)
    if scores.shape[0] != 2 * len(valid) or targets.shape != (2 * len(valid),):
        raise OfficialEvaluationError("Validation query matrix has the wrong shape")
    return scores, targets, counts


def test_score_matrix(
    model: str,
    entity: Any,
    relation: Any,
    train: Any,
    valid: Any,
    test: Any,
    *,
    transe_distance_norm: int,
    query_batch_size: int = QUERY_BATCH_SIZE,
    candidate_chunk_size: int = CANDIDATE_CHUNK_SIZE,
) -> Tuple[Any, Any, Any, Any]:
    """Score all head then tail test queries with train+valid+test filtering."""

    import numpy as np
    import torch

    tail_map, head_map = build_filter_maps(train, valid, test)
    head_scores, head_targets, head_counts = _direction_score_matrix(
        model,
        entity,
        relation,
        test,
        head_map,
        direction="head",
        transe_distance_norm=transe_distance_norm,
        query_batch_size=query_batch_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    tail_scores, tail_targets, tail_counts = _direction_score_matrix(
        model,
        entity,
        relation,
        test,
        tail_map,
        direction="tail",
        transe_distance_norm=transe_distance_norm,
        query_batch_size=query_batch_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    scores = torch.cat((head_scores, tail_scores), dim=0)
    targets = torch.cat((head_targets, tail_targets), dim=0)
    counts = torch.cat((head_counts, tail_counts), dim=0)
    target_indices = torch.as_tensor(
        np.concatenate((test[:, 0], test[:, 2])).astype(np.int64, copy=False),
        dtype=torch.long,
        device=entity.device,
    )
    expected = 2 * len(test)
    if (
        scores.shape[0] != expected
        or targets.shape != (expected,)
        or counts.shape != (expected,)
        or target_indices.shape != (expected,)
    ):
        raise OfficialEvaluationError("Test query matrix has the wrong shape")
    return scores, targets, counts, target_indices


def query_probability_records(
    filtered_scores: Any,
    target_scores: Any,
    target_indices: Any,
    candidate_counts: Any,
    calibrated_temperature: float,
    *,
    reduction_batch_size: int = 128,
) -> Dict[str, Any]:
    """Return deterministic query-level ranking and uncertainty arrays."""

    import numpy as np
    import torch

    query_count = int(filtered_scores.shape[0])
    if (
        filtered_scores.ndim != 2
        or target_scores.shape != (query_count,)
        or target_indices.shape != (query_count,)
        or candidate_counts.shape != (query_count,)
        or reduction_batch_size <= 0
        or not math.isfinite(calibrated_temperature)
        or calibrated_temperature <= 0.0
    ):
        raise OfficialEvaluationError("Query probability inputs are invalid")
    outputs: Dict[str, List[Any]] = {
        key: []
        for key in (
            "target_score",
            "rank",
            "predicted_index",
            "maximum_score",
            "top_score_tie_count",
            "correct",
            "raw_target_probability",
            "calibrated_target_probability",
            "raw_confidence",
            "calibrated_confidence",
            "raw_nll",
            "calibrated_nll",
            "raw_normalised_entropy",
            "calibrated_normalised_entropy",
        )
    }
    with torch.inference_mode():
        for start in range(0, query_count, reduction_batch_size):
            stop = min(start + reduction_batch_size, query_count)
            scores32 = filtered_scores[start:stop]
            scores64 = scores32.to(torch.float64)
            targets64 = target_scores[start:stop].to(torch.float64)
            indices = target_indices[start:stop]
            counts = candidate_counts[start:stop].to(torch.float64)
            maximum_scores, predicted = torch.max(scores32, dim=1)
            tie_counts = torch.sum(scores32 == maximum_scores[:, None], dim=1)
            target32 = target_scores[start:stop]
            greater = torch.sum(scores32 > target32[:, None], dim=1)
            equal_other = torch.sum(scores32 == target32[:, None], dim=1) - 1
            ranks = 1.0 + greater.to(torch.float64) + 0.5 * equal_other.to(torch.float64)
            if torch.any(equal_other < 0) or not torch.isfinite(ranks).all():
                raise OfficialEvaluationError("Realistic test rank is invalid")
            calibrated_scores = scores64 / calibrated_temperature
            calibrated_targets = targets64 / calibrated_temperature
            calibrated_greater = torch.sum(
                calibrated_scores > calibrated_targets[:, None], dim=1
            )
            calibrated_equal = torch.sum(
                calibrated_scores == calibrated_targets[:, None], dim=1
            ) - 1
            if not torch.equal(greater, calibrated_greater) or not torch.equal(
                equal_other, calibrated_equal
            ):
                raise OfficialEvaluationError("Temperature scaling changed candidate ranks")

            batch_values: Dict[str, Any] = {}
            for prefix, temperature in (
                ("raw", 1.0),
                ("calibrated", calibrated_temperature),
            ):
                logits = scores64 / temperature
                log_normaliser = torch.logsumexp(logits, dim=1)
                target_logits = targets64 / temperature
                maximum_logits = maximum_scores.to(torch.float64) / temperature
                nll = log_normaliser - target_logits
                target_probability = torch.exp(-nll)
                confidence = torch.exp(maximum_logits - log_normaliser)
                probability = torch.softmax(logits, dim=1)
                finite_logits = torch.where(torch.isfinite(logits), logits, 0.0)
                entropy = log_normaliser - torch.sum(
                    probability * finite_logits, dim=1
                )
                normalised_entropy = torch.where(
                    counts > 1.0,
                    entropy / torch.log(counts),
                    torch.zeros_like(entropy),
                )
                if torch.any(normalised_entropy < -1e-12) or torch.any(
                    normalised_entropy > 1.0 + 1e-12
                ):
                    raise OfficialEvaluationError(
                        "Normalised predictive entropy is outside numerical bounds"
                    )
                normalised_entropy = torch.clamp(normalised_entropy, 0.0, 1.0)
                for value in (nll, target_probability, confidence, normalised_entropy):
                    if not torch.isfinite(value).all():
                        raise OfficialEvaluationError(
                            "A query probability diagnostic is non-finite"
                        )
                if torch.any(target_probability < 0.0) or torch.any(
                    target_probability > 1.0
                ) or torch.any(confidence < 0.0) or torch.any(confidence > 1.0):
                    raise OfficialEvaluationError("A query probability is outside [0,1]")
                batch_values[prefix + "_target_probability"] = target_probability
                batch_values[prefix + "_confidence"] = confidence
                batch_values[prefix + "_nll"] = nll
                batch_values[prefix + "_normalised_entropy"] = normalised_entropy

            correct = predicted == indices
            values = {
                "target_score": target32,
                "rank": ranks,
                "predicted_index": predicted,
                "maximum_score": maximum_scores,
                "top_score_tie_count": tie_counts,
                "correct": correct,
                **batch_values,
            }
            for key, value in values.items():
                outputs[key].extend(value.detach().cpu().tolist())
    if any(len(values) != query_count for values in outputs.values()):
        raise OfficialEvaluationError("Query probability output count differs")
    outputs["predicted_index"] = [int(value) for value in outputs["predicted_index"]]
    outputs["top_score_tie_count"] = [
        int(value) for value in outputs["top_score_tie_count"]
    ]
    outputs["correct"] = [bool(value) for value in outputs["correct"]]
    for key, values in outputs.items():
        if key in {"predicted_index", "top_score_tie_count", "correct"}:
            continue
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise OfficialEvaluationError("A returned query diagnostic is non-finite")
    return outputs


def mean_multiclass_nll(
    filtered_scores: Any,
    target_scores: Any,
    temperature: float,
    *,
    reduction_batch_size: int = 256,
) -> float:
    import numpy as np
    import torch

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise OfficialEvaluationError("Temperature must be positive and finite")
    if filtered_scores.ndim != 2 or target_scores.shape != (filtered_scores.shape[0],):
        raise OfficialEvaluationError("NLL score inputs are misaligned")
    total = torch.zeros((), dtype=torch.float64, device=filtered_scores.device)
    with torch.inference_mode():
        for start in range(0, len(target_scores), reduction_batch_size):
            stop = min(start + reduction_batch_size, len(target_scores))
            logits = filtered_scores[start:stop].to(torch.float64) / temperature
            log_normaliser = torch.logsumexp(logits, dim=1)
            nll = log_normaliser - target_scores[start:stop].to(torch.float64) / temperature
            if not torch.isfinite(nll).all():
                raise OfficialEvaluationError("Temperature objective is non-finite")
            total += nll.sum()
    result = float((total / len(target_scores)).cpu())
    if not np.isfinite(result):
        raise OfficialEvaluationError("Mean temperature objective is non-finite")
    return result


def fit_validation_temperature(
    filtered_scores: Any,
    target_scores: Any,
    *,
    lower: float = TEMPERATURE_LOWER,
    upper: float = TEMPERATURE_UPPER,
    xatol: float = LOG_TEMPERATURE_XATOL,
    maxiter: int = TEMPERATURE_MAXITER,
) -> Dict[str, Any]:
    import numpy as np
    from scipy.optimize import minimize_scalar

    if not (0.0 < lower < upper) or xatol <= 0.0 or maxiter <= 0:
        raise OfficialEvaluationError("Temperature optimiser configuration is invalid")
    lower_log = math.log(lower)
    upper_log = math.log(upper)
    cache: Dict[float, float] = {}

    def objective(log_temperature: float) -> float:
        key = float(log_temperature)
        if key not in cache:
            cache[key] = mean_multiclass_nll(
                filtered_scores, target_scores, math.exp(key)
            )
        return cache[key]

    result = minimize_scalar(
        objective,
        bounds=(lower_log, upper_log),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x):
        raise OfficialEvaluationError("Bounded temperature optimisation failed")
    candidates = [
        (objective(lower_log), 0, lower_log, "lower"),
        (objective(float(result.x)), 1, float(result.x), "interior"),
        (objective(upper_log), 2, upper_log, "upper"),
    ]
    best_objective, _, best_log, boundary = min(candidates)
    temperature = math.exp(best_log)
    initial_objective = objective(0.0)
    if not all(np.isfinite(value) for value in (temperature, best_objective, initial_objective)):
        raise OfficialEvaluationError("Fitted temperature result is non-finite")
    return {
        "method": "scipy.optimize.minimize_scalar_bounded",
        "parameterisation": "natural_log_temperature",
        "temperature_bounds": [lower, upper],
        "log_temperature_bounds": [lower_log, upper_log],
        "xatol_log_temperature": xatol,
        "maximum_iterations": maxiter,
        "fitted_temperature": float(temperature),
        "fitted_log_temperature": float(best_log),
        "validation_mean_multiclass_nll": float(best_objective),
        "raw_temperature_one_validation_mean_multiclass_nll": float(initial_objective),
        "success": True,
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "scipy_iterations": int(result.nit),
        "scipy_function_evaluations": int(result.nfev),
        "unique_objective_evaluations": len(cache),
        "boundary_status": boundary,
        "boundary_objectives": {
            "lower": float(candidates[0][0]),
            "upper": float(candidates[2][0]),
        },
    }
