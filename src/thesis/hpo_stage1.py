"""Run one frozen HPO Stage 1 random-baseline screening trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .data_freeze import (
    copy_file_new_or_identical,
    git_state,
    write_bytes_new_or_identical,
)
from .kg_core import (
    batch_training_loss,
    ranking_metrics,
    sample_negative_triples_vectorized,
    triple_keys,
)
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object
from .training_smoke import load_entity_ids, load_indexed_tsv


class HPOStage1Error(RuntimeError):
    """Raised when a Stage 1 trajectory violates its frozen design."""


HPO_DESIGN_SHA256 = "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93"
EXPECTED_DEVICE = "NVIDIA H200 NVL"
EXPECTED_UNITS = 6
EXPECTED_CANDIDATES_PER_UNIT = 12
EXPECTED_ARRAY_TASKS = 72
STAGE = 1
REPLICATE = 1
EPOCHS = 25


def validate_design(design: Mapping[str, Any]) -> None:
    if design.get("schema_version") != 1:
        raise HPOStage1Error("HPO design schema version is invalid")
    if (
        design.get("status") != "hpo_design_frozen_pending_stage_1"
        or design.get("official_result") is not False
        or design.get("test_data_accessed") is not False
        or design.get("text_conditions_accessed") is not False
    ):
        raise HPOStage1Error("HPO design status/boundary is invalid")
    if design.get("selection_metric") != "combined_filtered_validation_mrr":
        raise HPOStage1Error("HPO selection metric differs from the protocol")
    stage = design.get("stage_plan", {}).get("stage_1")
    if stage != {
        "candidates": 12,
        "tuning_replicates": [1],
        "epochs": 25,
        "validation_epochs": [25],
        "advance": 4,
    }:
        raise HPOStage1Error("HPO Stage 1 plan differs from the protocol")
    units = design.get("units")
    if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
        raise HPOStage1Error("HPO design must contain six tuning units")
    candidate_ids = []
    for unit in units:
        candidates = unit.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != EXPECTED_CANDIDATES_PER_UNIT
        ):
            raise HPOStage1Error("An HPO unit does not contain twelve candidates")
        for candidate in candidates:
            parameters = candidate.get("parameters")
            fingerprint = hashlib.sha256(
                canonical_json_bytes(parameters)
            ).hexdigest()
            if fingerprint != candidate.get("configuration_sha256"):
                raise HPOStage1Error("An HPO candidate fingerprint is invalid")
            candidate_ids.append(candidate.get("candidate_id"))
    if len(candidate_ids) != EXPECTED_ARRAY_TASKS or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise HPOStage1Error("HPO candidate identifiers are incomplete or duplicated")


def select_array_task(
    design: Mapping[str, Any], array_index: int
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    validate_design(design)
    if not 0 <= array_index < EXPECTED_ARRAY_TASKS:
        raise HPOStage1Error("Slurm array index is outside 0..71")
    unit = design["units"][array_index // EXPECTED_CANDIDATES_PER_UNIT]
    candidate = unit["candidates"][array_index % EXPECTED_CANDIDATES_PER_UNIT]
    return unit, candidate


def stream_seed(unit: Mapping[str, Any], namespace: str) -> int:
    matches = [
        item
        for item in unit.get("tuning_stream_derivations", ())
        if item.get("replicate") == REPLICATE
        and item.get("namespace") == namespace
    ]
    if len(matches) != 1:
        raise HPOStage1Error("A required tuning RNG substream is absent")
    item = matches[0]
    digest = hashlib.sha256(str(item["input"]).encode("utf-8")).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest)[:4], "big", signed=False)
    if digest != item.get("sha256") or seed != item.get("seed"):
        raise HPOStage1Error("A tuning RNG substream derivation is invalid")
    return seed


def random_unit_array(rows: int, columns: int, seed: int) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, columns)).astype(np.float32)
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise HPOStage1Error("Random initialisation contains an invalid norm")
    values = (values.astype(np.float64) / norms[:, None]).astype(np.float32)
    if not np.isfinite(values).all():
        raise HPOStage1Error("Random initialisation contains a non-finite value")
    return values


def initialisation_statistics(values: Any) -> Dict[str, float]:
    import numpy as np

    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    statistics = {
        "minimum_l2_norm": float(norms.min()),
        "maximum_l2_norm": float(norms.max()),
        "mean_l2_norm": float(norms.mean()),
        "standard_deviation_l2_norm": float(norms.std()),
        "maximum_absolute_deviation_from_unit_norm": float(
            np.max(np.abs(norms - 1.0))
        ),
    }
    if (
        not np.isfinite(tuple(statistics.values())).all()
        or statistics["maximum_absolute_deviation_from_unit_norm"] > 1e-6
    ):
        raise HPOStage1Error("Initialisation violates the unit-norm audit")
    return statistics


def load_stage_data(
    project_root: Path,
    dataset: str,
    expected_manifest_sha256: str,
) -> Dict[str, Any]:
    import numpy as np

    manifest_path = project_root / "data" / "manifests" / (dataset + "_dataset.json")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise HPOStage1Error("Dataset manifest hash differs from the preflight")
    manifest = read_json_object(manifest_path)
    processed = project_root / "data" / "processed" / dataset
    expected = manifest.get("processed_sha256")
    if not isinstance(expected, dict):
        raise HPOStage1Error("Dataset manifest lacks processed hashes")
    required = ("train.tsv", "valid.tsv", "entities.tsv", "relations.tsv")
    for filename in required:
        if sha256_file(processed / filename) != expected.get(filename):
            raise HPOStage1Error("Processed dataset artifact hash mismatch")
    train = load_indexed_tsv(processed / "train.tsv")
    valid = load_indexed_tsv(processed / "valid.tsv")
    entity_count = len(load_entity_ids(processed / "entities.tsv"))
    relation_count = len(load_entity_ids(processed / "relations.tsv"))
    for values in (train, valid):
        if (
            int(values[:, [0, 2]].min()) < 0
            or int(values[:, [0, 2]].max()) >= entity_count
            or int(values[:, 1].min()) < 0
            or int(values[:, 1].max()) >= relation_count
        ):
            raise HPOStage1Error("A split index is outside the frozen vocabulary")
    train_keys = np.sort(
        triple_keys(
            train, entity_count=entity_count, relation_count=relation_count
        )
    )
    if len(np.unique(train_keys)) != len(train):
        raise HPOStage1Error("Training split contains duplicate triples")
    train_seen = np.unique(train[:, [0, 2]])
    return {
        "manifest_path": manifest_path,
        "train": train,
        "valid": valid,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "sorted_train_keys": train_keys,
        "train_seen_entities": train_seen,
        "globally_train_unseen_entities": np.setdiff1d(
            np.arange(entity_count, dtype=np.int64), train_seen
        ),
    }


def build_filter_maps(
    train: Any, valid: Any
) -> Tuple[Mapping[Tuple[int, int], Set[int]], Mapping[Tuple[int, int], Set[int]]]:
    tail_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    head_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for source in (train, valid):
        for row in source:
            head, relation, tail = map(int, row)
            tail_map[(head, relation)].add(tail)
            head_map[(relation, tail)].add(head)
    return tail_map, head_map


def score_candidate_block(
    model: str,
    entity: Any,
    relation: Any,
    triples: Any,
    *,
    direction: str,
    candidate_start: int,
    candidate_stop: int,
    transe_distance_norm: int,
) -> Any:
    import torch

    if direction not in {"head", "tail"}:
        raise HPOStage1Error("Prediction direction must be head or tail")
    candidates = entity[candidate_start:candidate_stop]
    heads = triples[:, 0]
    relations = triples[:, 1]
    tails = triples[:, 2]
    if model == "TransE":
        if direction == "tail":
            queries = entity[heads] + relation[relations]
        else:
            queries = entity[tails] - relation[relations]
        return -torch.linalg.vector_norm(
            queries[:, None, :] - candidates[None, :, :],
            ord=transe_distance_norm,
            dim=-1,
        )
    if model == "DistMult":
        if direction == "tail":
            queries = entity[heads] * relation[relations]
        else:
            queries = relation[relations] * entity[tails]
        return queries @ candidates.transpose(0, 1)
    raise HPOStage1Error("Unknown KGE model")


def rank_direction(
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
) -> List[float]:
    import numpy as np
    import torch

    ranks: List[float] = []
    entity_count = int(entity.shape[0])
    with torch.inference_mode():
        for query_start in range(0, len(triples_numpy), query_batch_size):
            batch_numpy = triples_numpy[
                query_start : query_start + query_batch_size
            ]
            triples = torch.as_tensor(
                batch_numpy, dtype=torch.long, device=entity.device
            )
            target_indices = (
                triples[:, 2] if direction == "tail" else triples[:, 0]
            )
            target_scores = torch.empty(
                len(triples), dtype=entity.dtype, device=entity.device
            )
            target_chunk_ids = torch.div(
                target_indices,
                candidate_chunk_size,
                rounding_mode="floor",
            )
            for target_chunk_id in torch.unique(target_chunk_ids).tolist():
                target_rows = torch.nonzero(
                    target_chunk_ids == target_chunk_id, as_tuple=False
                ).flatten()
                candidate_start = int(target_chunk_id) * candidate_chunk_size
                candidate_stop = min(
                    candidate_start + candidate_chunk_size, entity_count
                )
                target_block = score_candidate_block(
                    model,
                    entity,
                    relation,
                    triples[target_rows],
                    direction=direction,
                    candidate_start=candidate_start,
                    candidate_stop=candidate_stop,
                    transe_distance_norm=transe_distance_norm,
                )
                target_columns = target_indices[target_rows] - candidate_start
                target_scores[target_rows] = target_block[
                    torch.arange(len(target_rows), device=entity.device),
                    target_columns,
                ]
            if not torch.isfinite(target_scores).all():
                raise HPOStage1Error("A validation target score is non-finite")
            greater = torch.zeros(
                len(triples), dtype=torch.int64, device=entity.device
            )
            equal = torch.zeros_like(greater)
            filter_sets = []
            for row in batch_numpy:
                head, relation_id, tail = map(int, row)
                key = (
                    (head, relation_id)
                    if direction == "tail"
                    else (relation_id, tail)
                )
                filter_sets.append(filter_map[key])
            for candidate_start in range(
                0, entity_count, candidate_chunk_size
            ):
                candidate_stop = min(
                    candidate_start + candidate_chunk_size, entity_count
                )
                scores = score_candidate_block(
                    model,
                    entity,
                    relation,
                    triples,
                    direction=direction,
                    candidate_start=candidate_start,
                    candidate_stop=candidate_stop,
                    transe_distance_norm=transe_distance_norm,
                )
                if not torch.isfinite(scores).all():
                    raise HPOStage1Error("A validation candidate score is non-finite")
                eligible = torch.ones_like(scores, dtype=torch.bool)
                target_cpu = target_indices.detach().cpu().numpy()
                for query_index, filtered in enumerate(filter_sets):
                    local_filtered = [
                        index - candidate_start
                        for index in filtered
                        if candidate_start <= index < candidate_stop
                        and index != int(target_cpu[query_index])
                    ]
                    if local_filtered:
                        eligible[query_index, local_filtered] = False
                target_in_chunk = (target_indices >= candidate_start) & (
                    target_indices < candidate_stop
                )
                target_rows = torch.nonzero(target_in_chunk, as_tuple=False).flatten()
                if len(target_rows):
                    target_columns = target_indices[target_rows] - candidate_start
                    # The comparison reference is the value read from the target
                    # entry of its candidate block above, never an independently
                    # evaluated triple score.
                    scores[target_rows, target_columns] = target_scores[target_rows]
                    eligible[target_rows, target_columns] = True
                greater += (
                    (scores > target_scores[:, None]) & eligible
                ).sum(dim=1)
                equal += ((scores == target_scores[:, None]) & eligible).sum(dim=1)
            if torch.any(equal < 1):
                raise HPOStage1Error("A validation target disappeared from ranking")
            batch_ranks = (
                1.0 + greater.to(torch.float64) + 0.5 * (equal - 1)
            )
            values = batch_ranks.cpu().numpy()
            if not np.isfinite(values).all():
                raise HPOStage1Error("A validation rank is non-finite")
            ranks.extend(map(float, values))
    return ranks


def validation_rank_table_bytes(
    valid: Any, head_ranks: Sequence[float], tail_ranks: Sequence[float]
) -> bytes:
    if not (len(valid) == len(head_ranks) == len(tail_ranks)):
        raise HPOStage1Error("Validation rank-table inputs are misaligned")
    lines = [
        "validation_row\tdirection\thead\trelation\ttail\ttarget_entity\trank"
    ]
    for index, row in enumerate(valid):
        head, relation, tail = map(int, row)
        lines.append(
            "{}\thead\t{}\t{}\t{}\t{}\t{}".format(
                index, head, relation, tail, head, head_ranks[index]
            )
        )
        lines.append(
            "{}\ttail\t{}\t{}\t{}\t{}\t{}".format(
                index, head, relation, tail, tail, tail_ranks[index]
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def evaluate_validation(
    model: str,
    entity: Any,
    relation: Any,
    train: Any,
    valid: Any,
    train_seen_entities: Any,
    *,
    transe_distance_norm: int,
    query_batch_size: int = 32,
    candidate_chunk_size: int = 4096,
) -> Tuple[Dict[str, Any], bytes]:
    import numpy as np

    tail_map, head_map = build_filter_maps(train, valid)
    head_ranks = rank_direction(
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
    tail_ranks = rank_direction(
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
    combined = head_ranks + tail_ranks
    if len(combined) != 2 * len(valid) or not np.isfinite(combined).all():
        raise HPOStage1Error("Combined validation ranks are invalid")
    train_seen = set(map(int, train_seen_entities))
    head_seen_target = [int(row[0]) in train_seen for row in valid]
    tail_seen_target = [int(row[2]) in train_seen for row in valid]
    both_endpoints_seen = [
        int(row[0]) in train_seen and int(row[2]) in train_seen for row in valid
    ]

    def optional_metrics(ranks: Sequence[float], mask: Sequence[bool]) -> Dict[str, Any]:
        selected = [rank for rank, keep in zip(ranks, mask) if keep]
        if not selected:
            return {
                "queries": 0,
                "mrr": None,
                "hits_at_1": None,
                "hits_at_3": None,
                "hits_at_10": None,
            }
        return ranking_metrics(selected)

    def stratum(
        head_mask: Sequence[bool], tail_mask: Sequence[bool]
    ) -> Dict[str, Any]:
        return {
            "head": optional_metrics(head_ranks, head_mask),
            "tail": optional_metrics(tail_ranks, tail_mask),
            "combined": optional_metrics(
                combined, list(head_mask) + list(tail_mask)
            ),
        }

    all_mask = [True] * len(valid)
    validation_report = {
        "filter_scope": "train_plus_validation",
        "head": ranking_metrics(head_ranks),
        "tail": ranking_metrics(tail_ranks),
        "combined": ranking_metrics(combined),
        "strata": {
            "all": stratum(all_mask, all_mask),
            "seen_target": stratum(head_seen_target, tail_seen_target),
            "unseen_target": stratum(
                [not value for value in head_seen_target],
                [not value for value in tail_seen_target],
            ),
            "both_endpoints_seen": stratum(
                both_endpoints_seen, both_endpoints_seen
            ),
            "at_least_one_endpoint_unseen": stratum(
                [not value for value in both_endpoints_seen],
                [not value for value in both_endpoints_seen],
            ),
        },
    }
    return (
        validation_report,
        validation_rank_table_bytes(valid, head_ranks, tail_ranks),
    )


def save_numpy_artifact(
    values: Any,
    destination: Path,
    *,
    scratch: Path,
    job_id: str,
    label: str,
) -> Dict[str, Any]:
    import numpy as np

    temporary = scratch / (label + ".npy")
    with temporary.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    digest = sha256_file(temporary)
    size = temporary.stat().st_size
    copy_file_new_or_identical(temporary, destination, digest, size, job_id)
    return {
        "path": str(destination),
        "sha256": digest,
        "size_bytes": size,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def save_checkpoint(
    payload: Mapping[str, Any],
    destination: Path,
    *,
    scratch: Path,
    job_id: str,
) -> Dict[str, Any]:
    import torch

    temporary = scratch / "checkpoint.pt"
    with temporary.open("xb") as handle:
        torch.save(dict(payload), handle)
    digest = sha256_file(temporary)
    size = temporary.stat().st_size
    copy_file_new_or_identical(temporary, destination, digest, size, job_id)
    return {"path": str(destination), "sha256": digest, "size_bytes": size}


def training_environment(
    preflight: Mapping[str, Any]
) -> Dict[str, Any]:
    import torch

    expected = preflight.get("torch_runtime", {})
    properties = torch.cuda.get_device_properties(0)
    observed = {
        "torch_version": torch.__version__,
        "compiled_cuda_version": torch.version.cuda,
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    if observed != expected or observed["device_name"] != EXPECTED_DEVICE:
        raise HPOStage1Error("Training environment differs from the H200 preflight")
    return observed


def run_stage1(
    design_path: Path,
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
    if sha256_file(design_path) != HPO_DESIGN_SHA256:
        raise HPOStage1Error("HPO design file hash differs from the frozen design")
    design = read_json_object(design_path)
    unit, candidate = select_array_task(design, array_index)
    if sha256_file(protocol_path) != design["protocol_sha256"]:
        raise HPOStage1Error("Protocol hash differs from the HPO design")
    if sha256_file(preflight_path) != design["hardware_preflight_report_sha256"]:
        raise HPOStage1Error("Hardware-preflight report hash differs from the HPO design")
    preflight = read_json_object(preflight_path)
    git = git_state(project_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise HPOStage1Error("Exactly one scheduler-visible CUDA device is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    environment = training_environment(preflight)
    dataset = str(unit["dataset"])
    model = str(unit["model"])
    candidate_id = str(candidate["candidate_id"])
    parameters = candidate["parameters"]
    preflight_dataset = [
        item for item in preflight["datasets"] if item["dataset"] == dataset
    ]
    if len(preflight_dataset) != 1:
        raise HPOStage1Error("Dataset is absent from hardware preflight")
    data = load_stage_data(
        project_root,
        dataset,
        str(preflight_dataset[0]["dataset_manifest_sha256"]),
    )
    seeds = {
        namespace: stream_seed(unit, namespace)
        for namespace in (
            "random_entity_initialisation",
            "relation_initialisation",
            "batch_order",
            "negative_sampling",
            "data_loader_workers",
        )
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
    entity_initialisation_statistics = initialisation_statistics(entity_numpy)
    relation_initialisation_statistics = initialisation_statistics(relation_numpy)
    device = "cuda"
    entity = torch.nn.Parameter(
        torch.as_tensor(entity_numpy, dtype=torch.float32, device=device)
    )
    relation = torch.nn.Parameter(
        torch.as_tensor(relation_numpy, dtype=torch.float32, device=device)
    )
    globally_unseen = data["globally_train_unseen_entities"]
    globally_unseen_device = torch.as_tensor(
        globally_unseen, dtype=torch.long, device=device
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
            float(design["fixed_training"]["adam_beta1"]),
            float(design["fixed_training"]["adam_beta2"]),
        ),
        eps=float(design["fixed_training"]["adam_epsilon"]),
        weight_decay=float(design["fixed_training"]["weight_decay"]),
    )
    batch_size = int(parameters["batch_size"])
    negatives_per_positive = int(parameters["negatives_per_positive"])
    regularisation_coefficient = float(
        parameters["regularisation_coefficient"]
    )
    transe_distance_norm = int(parameters.get("transe_distance_norm", 1))
    batch_rng = np.random.default_rng(seeds["batch_order"])
    negative_rng = np.random.default_rng(seeds["negative_sampling"])
    sequence_digest = hashlib.sha256()
    batch_losses: List[float] = []
    ranking_losses: List[float] = []
    regularisation_values: List[float] = []
    epoch_history = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    print(
        "HPO_STAGE1_START "
        + json.dumps(
            {
                "array_index": array_index,
                "dataset": dataset,
                "model": model,
                "candidate_id": candidate_id,
                "parameters": parameters,
                "epochs": EPOCHS,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, EPOCHS + 1):
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
                positives_numpy, dtype=torch.long, device=device
            )
            negatives = torch.as_tensor(
                negatives_numpy, dtype=torch.long, device=device
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
                raise HPOStage1Error("Stage 1 training loss is non-finite")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            referenced = torch.cat((positives, negatives.reshape(-1, 3)), dim=0)
            referenced_entities = torch.unique(referenced[:, [0, 2]])
            referenced_relations = torch.unique(referenced[:, 1])
            if not torch.isfinite(entity[referenced_entities]).all() or not torch.isfinite(
                relation[referenced_relations]
            ).all():
                raise HPOStage1Error("Stage 1 training produced a non-finite embedding")
            batch_losses.append(float(loss.detach().cpu()))
            ranking_losses.append(float(ranking_loss.detach().cpu()))
            regularisation_values.append(float(regularisation.detach().cpu()))
        if not torch.isfinite(entity).all() or not torch.isfinite(relation).all():
            raise HPOStage1Error("An epoch ended with a non-finite embedding")
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
            "HPO_STAGE1_EPOCH " + json.dumps(epoch_record, sort_keys=True),
            flush=True,
        )

    torch.cuda.synchronize()
    training_runtime = time.perf_counter() - training_started
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
            raise HPOStage1Error("A globally training-unseen entity changed")

    print("HPO_STAGE1_VALIDATION_START", flush=True)
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
    print(
        "HPO_STAGE1_VALIDATION_DONE "
        + json.dumps(
            {
                "combined_mrr": validation["combined"]["mrr"],
                "runtime_seconds": float(evaluation_runtime),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    array_job_id = str(os.environ.get("SLURM_ARRAY_JOB_ID", execution["scheduler_job_id"]))
    attempt_name = "attempt-{}-{}".format(array_job_id, array_index)
    run_directory = (
        artifact_root
        / str(unit["unit_id"])
        / candidate_id
        / "replicate-1"
        / attempt_name
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    job_id = str(execution["scheduler_job_id"])
    shared_initialisation_directory = (
        artifact_root
        / "_shared_initialisations"
        / str(unit["unit_id"])
        / "replicate-1"
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
    rank_path = run_directory / "validation_ranks.tsv"
    write_bytes_new_or_identical(rank_path, rank_table, job_id)
    rank_artifact = {
        "path": str(rank_path),
        "sha256": sha256_file(rank_path),
        "size_bytes": rank_path.stat().st_size,
        "rows": int(2 * len(data["valid"])),
    }
    checkpoint = save_checkpoint(
        {
            "schema_version": 1,
            "stage": STAGE,
            "replicate": REPLICATE,
            "dataset": dataset,
            "model": model,
            "candidate_id": candidate_id,
            "parameters": parameters,
            "epochs": EPOCHS,
            "entity_embeddings": entity.detach().cpu(),
            "relation_embeddings": relation.detach().cpu(),
        },
        run_directory / "checkpoint.pt",
        scratch=scratch,
        job_id=job_id,
    )
    report = {
        "schema_version": 1,
        "status": "hpo_stage_1_run_completed",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "stage": STAGE,
        "replicate": REPLICATE,
        "array_index": array_index,
        "array_job_id": array_job_id,
        "dataset": dataset,
        "model": model,
        "unit_id": unit["unit_id"],
        "candidate_id": candidate_id,
        "candidate_configuration_sha256": candidate["configuration_sha256"],
        "parameters": parameters,
        "epochs": EPOCHS,
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
            "entity_matrix_raw_sha256": entity_initialisation_sha,
            "relation_matrix_raw_sha256": relation_initialisation_sha,
            "entity_statistics": entity_initialisation_statistics,
            "relation_statistics": relation_initialisation_statistics,
            "entity_artifact": entity_initialisation_artifact,
            "relation_artifact": relation_initialisation_artifact,
        },
        "training": {
            "batches": len(batch_losses),
            "training_sequence_sha256": sequence_digest.hexdigest(),
            "epoch_history": epoch_history,
            "runtime_seconds": float(training_runtime),
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
        "validation": validation,
        "validation_runtime_seconds": float(evaluation_runtime),
        "artifacts": {
            "checkpoint": checkpoint,
            "batch_loss_history": loss_artifact,
            "validation_ranks": rank_artifact,
        },
        "hpo_design_sha256": sha256_file(design_path),
        "hardware_preflight_report_sha256": sha256_file(preflight_path),
        "protocol_sha256": sha256_file(protocol_path),
        "execution": {
            **execution,
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "training_environment": environment,
        "git": git,
        "run_directory": str(run_directory),
    }
    del entity, relation, optimiser
    torch.cuda.empty_cache()
    gc.collect()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
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
        report = run_stage1(
            args.design.resolve(),
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
            "status": "hpo_stage_1_run_failed",
            "official_result": False,
            "test_data_accessed": False,
            "text_conditions_accessed": False,
            "stage": STAGE,
            "replicate": REPLICATE,
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
    print("HPO Stage 1 run recorded in {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
