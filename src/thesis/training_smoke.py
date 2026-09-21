"""Run a non-official end-to-end training/evaluation smoke test on frozen data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .kg_core import (
    KGCoreError,
    make_training_schedule,
    ranking_metrics,
    realistic_filtered_rank,
    score_candidates,
    train_embeddings,
)
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object
from .text_encoder import ENCODED_COLUMNS


PREFIX = "Integrating External Evidence to Reduce Uncertainty in Link Prediction"


class TrainingSmokeError(RuntimeError):
    """Raised when a non-official training smoke invariant fails."""


def seed32(label: str) -> int:
    digest = hashlib.sha256((PREFIX + "|" + label).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def load_indexed_tsv(path: Path) -> Any:
    import numpy as np

    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise TrainingSmokeError("Processed triple row is not three-column")
        try:
            rows.append(tuple(int(value) for value in fields))
        except ValueError as error:
            raise TrainingSmokeError("Processed triple row is non-integer") from error
    if not rows:
        raise TrainingSmokeError("Processed split is empty")
    return np.asarray(rows, dtype=np.int64)


def load_entity_ids(path: Path) -> List[str]:
    values = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != str(index):
            raise TrainingSmokeError("Entity mapping is not contiguous")
        values.append(fields[1])
    return values


def load_encoded_hashes(path: Path, expected_rows: int) -> Tuple[List[str], List[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != ENCODED_COLUMNS:
        raise TrainingSmokeError("Encoded evidence table schema is invalid")
    evidence_hashes = []
    vector_hashes = []
    for index, line in enumerate(lines[1:]):
        fields = line.split("\t")
        if len(fields) != len(ENCODED_COLUMNS):
            raise TrainingSmokeError("Encoded evidence table row is invalid")
        record = dict(zip(ENCODED_COLUMNS, fields))
        if record["entity_index"] != str(index):
            raise TrainingSmokeError("Encoded evidence ordering is invalid")
        evidence_hashes.append(record["evidence_sha256"])
        vector_hashes.append(record["normalised_vector_row_sha256"])
    if len(evidence_hashes) != expected_rows:
        raise TrainingSmokeError("Encoded evidence row count is invalid")
    return evidence_hashes, vector_hashes


def build_toy_graph(
    train: Any,
    valid: Any,
    *,
    full_entity_count: int,
    selected_training_entities: int,
    maximum_training_triples: int,
    maximum_validation_triples: int,
) -> Dict[str, Any]:
    import numpy as np

    frequency = Counter(map(int, train[:, 0]))
    frequency.update(map(int, train[:, 2]))
    selected = [
        entity
        for entity, _ in sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[
            :selected_training_entities
        ]
    ]
    selected_set = set(selected)
    train_seen_full = set(map(int, train[:, 0])) | set(map(int, train[:, 2]))
    truly_unseen = sorted(set(range(full_entity_count)) - train_seen_full)
    if not truly_unseen:
        raise TrainingSmokeError("Dataset has no training-unseen entity for the smoke test")
    candidate_global = sorted(selected_set | {truly_unseen[0]})
    global_to_toy = {entity: index for index, entity in enumerate(candidate_global)}

    induced_train = [
        tuple(map(int, triple))
        for triple in train
        if int(triple[0]) in selected_set and int(triple[2]) in selected_set
    ]
    positive_global = induced_train[:maximum_training_triples]
    if len(positive_global) < 100:
        raise TrainingSmokeError("Toy graph has too few positive training triples")
    relation_ids = sorted({relation for _, relation, _ in positive_global})
    relation_to_toy = {relation: index for index, relation in enumerate(relation_ids)}

    def remap(triple: Tuple[int, int, int]) -> Tuple[int, int, int]:
        head, relation, tail = triple
        return global_to_toy[head], relation_to_toy[relation], global_to_toy[tail]

    positives = np.asarray([remap(triple) for triple in positive_global], dtype=np.int64)
    full_train_true = {
        remap(triple)
        for triple in induced_train
        if triple[1] in relation_to_toy
    }
    valid_global = [
        tuple(map(int, triple))
        for triple in valid
        if int(triple[0]) in global_to_toy
        and int(triple[2]) in global_to_toy
        and int(triple[1]) in relation_to_toy
    ]
    if len(valid_global) < 10:
        raise TrainingSmokeError("Toy graph has too few validation triples")
    all_valid = np.asarray([remap(triple) for triple in valid_global], dtype=np.int64)
    evaluated_valid = all_valid[:maximum_validation_triples]
    replacement_entities = np.asarray(
        sorted(set(map(int, positives[:, 0])) | set(map(int, positives[:, 2]))),
        dtype=np.int64,
    )
    training_unseen = sorted(set(range(len(candidate_global))) - set(replacement_entities))
    if global_to_toy[truly_unseen[0]] not in training_unseen:
        raise TrainingSmokeError("Known training-unseen entity entered the toy training population")
    return {
        "candidate_global_indices": candidate_global,
        "relation_global_indices": relation_ids,
        "train": positives,
        "full_train_true": full_train_true,
        "all_valid": all_valid,
        "evaluated_valid": evaluated_valid,
        "replacement_entities": replacement_entities,
        "training_unseen": training_unseen,
        "selected_full_training_unseen_global_index": truly_unseen[0],
    }


def make_valid_shuffle(
    evidence_hashes: Sequence[str],
    vector_hashes: Sequence[str],
    *,
    seed: int,
    maximum_attempts: int,
) -> Tuple[Any, int, str]:
    import numpy as np

    if len(evidence_hashes) != len(vector_hashes):
        raise TrainingSmokeError("Shuffle hash populations differ")
    rng = np.random.default_rng(seed)
    size = len(evidence_hashes)
    indices = np.arange(size)
    for attempt in range(1, maximum_attempts + 1):
        permutation = rng.permutation(size)
        if np.any(permutation == indices):
            continue
        if any(evidence_hashes[source] == evidence_hashes[target] for target, source in enumerate(permutation)):
            continue
        if any(vector_hashes[source] == vector_hashes[target] for target, source in enumerate(permutation)):
            continue
        digest = hashlib.sha256(permutation.astype("<i8", copy=False).tobytes()).hexdigest()
        return permutation, attempt, digest
    raise TrainingSmokeError("No valid shuffled-text derangement was generated")


def random_unit_matrix(rows: int, columns: int, seed: int, device: str) -> Any:
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, columns)).astype(np.float32)
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(norms <= 1e-12):
        raise TrainingSmokeError("Random initialisation contains a zero norm")
    values = (values.astype(np.float64) / norms[:, None]).astype(np.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def build_filter_maps(
    train_true: Set[Tuple[int, int, int]],
    valid: Any,
) -> Tuple[Mapping[Tuple[int, int], Set[int]], Mapping[Tuple[int, int], Set[int]]]:
    tail_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    head_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for head, relation, tail in list(train_true) + [tuple(map(int, row)) for row in valid]:
        tail_map[(head, relation)].add(tail)
        head_map[(relation, tail)].add(head)
    return tail_map, head_map


def evaluate_validation(
    model: str,
    entity: Any,
    relation: Any,
    evaluated: Any,
    all_valid: Any,
    train_true: Set[Tuple[int, int, int]],
    *,
    transe_distance_norm: int,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    tail_map, head_map = build_filter_maps(train_true, all_valid)
    head_ranks = []
    tail_ranks = []
    with torch.inference_mode():
        for row in evaluated:
            triple = tuple(map(int, row))
            head, relation_id, tail = triple
            tail_scores = score_candidates(
                model,
                entity,
                relation,
                triple,
                direction="tail",
                transe_distance_norm=transe_distance_norm,
                chunk_size=127,
            ).cpu().numpy()
            head_scores = score_candidates(
                model,
                entity,
                relation,
                triple,
                direction="head",
                transe_distance_norm=transe_distance_norm,
                chunk_size=127,
            ).cpu().numpy()
            tail_ranks.append(
                realistic_filtered_rank(
                    tail_scores,
                    target_index=tail,
                    filtered_indices=tail_map[(head, relation_id)],
                )
            )
            head_ranks.append(
                realistic_filtered_rank(
                    head_scores,
                    target_index=head,
                    filtered_indices=head_map[(relation_id, tail)],
                )
            )
    combined = head_ranks + tail_ranks
    if len(combined) != 2 * len(evaluated) or not np.isfinite(combined).all():
        raise TrainingSmokeError("Validation query count/ranks are invalid")
    return {
        "head": ranking_metrics(head_ranks),
        "tail": ranking_metrics(tail_ranks),
        "combined": ranking_metrics(combined),
    }


def run_smoke(config_path: Path, *, project_root: Path, scratch: Path) -> Dict[str, Any]:
    import numpy as np
    import torch

    execution = require_scheduled_execution(scratch.resolve())
    config = read_json_object(config_path)
    if config.get("status") != "development_training_smoke_configuration" or config.get("official_result") is not False:
        raise TrainingSmokeError("Training smoke configuration is invalid")
    git = git_state(project_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TrainingSmokeError("Exactly one scheduler-visible CUDA device is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda"
    dataset = config["dataset"]
    processed_root = project_root / "data" / "processed" / dataset
    train = load_indexed_tsv(processed_root / "train.tsv")
    valid = load_indexed_tsv(processed_root / "valid.tsv")
    entity_ids = load_entity_ids(processed_root / "entities.tsv")
    selection = config["selection"]
    toy = build_toy_graph(
        train,
        valid,
        full_entity_count=len(entity_ids),
        selected_training_entities=selection["most_frequent_training_entities"],
        maximum_training_triples=selection["maximum_positive_training_triples"],
        maximum_validation_triples=selection["maximum_validation_triples"],
    )
    embedding_manifest_path = project_root / "data" / "manifests" / (dataset + "_text_embeddings.json")
    embedding_manifest = read_json_object(embedding_manifest_path)
    embedding_artifact = embedding_manifest["artifacts"]["normalised_embeddings"]
    embedding_path = project_root / "artifacts" / "embeddings" / dataset / embedding_artifact["filename"]
    if sha256_file(embedding_path) != embedding_artifact["sha256"]:
        raise TrainingSmokeError("Normalised text embedding artifact hash mismatch")
    full_text = np.load(embedding_path, allow_pickle=False)
    if full_text.shape != (len(entity_ids), 384) or full_text.dtype != np.float32:
        raise TrainingSmokeError("Normalised text embedding shape/dtype mismatch")
    candidate_global = toy["candidate_global_indices"]
    correct_numpy = full_text[candidate_global].copy()
    correct = torch.as_tensor(correct_numpy, device=device)
    encoded_table = project_root / "artifacts" / "descriptions" / dataset / "evidence_encoded.tsv"
    encoded_artifact = embedding_manifest["artifacts"]["encoded_evidence_table"]
    if sha256_file(encoded_table) != encoded_artifact["sha256"]:
        raise TrainingSmokeError("Encoded evidence table hash mismatch")
    evidence_all, vector_all = load_encoded_hashes(encoded_table, len(entity_ids))
    evidence_hashes = [evidence_all[index] for index in candidate_global]
    vector_hashes = [vector_all[index] for index in candidate_global]
    for local_index, row in enumerate(correct_numpy):
        if hashlib.sha256(row.tobytes(order="C")).hexdigest() != vector_hashes[local_index]:
            raise TrainingSmokeError("Vector row hash differs from encoded evidence table")
    shuffle_seed = seed32(config["development_seed_label"] + "|shuffle")
    permutation, shuffle_attempt, permutation_sha = make_valid_shuffle(
        evidence_hashes,
        vector_hashes,
        seed=shuffle_seed,
        maximum_attempts=config["shuffle"]["maximum_attempts"],
    )
    shuffled = correct[torch.as_tensor(permutation, dtype=torch.long, device=device)]
    training = config["training"]
    reports = []
    sequence_hashes = {}
    for model in config["models"]:
        relation_seed = seed32(config["development_seed_label"] + "|" + model + "|relation")
        random_seed = seed32(config["development_seed_label"] + "|" + model + "|random_entity")
        batch_seed = seed32(config["development_seed_label"] + "|" + model + "|batch")
        negative_seed = seed32(config["development_seed_label"] + "|" + model + "|negative")
        relation_initial = random_unit_matrix(
            len(toy["relation_global_indices"]), 384, relation_seed, device
        )
        random_initial = random_unit_matrix(len(candidate_global), 384, random_seed, device)
        schedule, sequence_sha = make_training_schedule(
            toy["train"],
            epochs=training["epochs"],
            batch_size=training["batch_size"],
            negatives_per_positive=training["negatives_per_positive"],
            replacement_entities=toy["replacement_entities"],
            true_training_triples=toy["full_train_true"],
            batch_seed=batch_seed,
            negative_seed=negative_seed,
            maximum_attempts=training["maximum_negative_sampling_attempts"],
        )
        sequence_hashes[model] = sequence_sha
        initialisations = {
            "random": random_initial,
            "correct_text": correct,
            "shuffled_text": shuffled,
        }
        model_runs = []
        for condition in config["conditions"]:
            initial_entity = initialisations[condition].clone()
            unseen_initial = initial_entity[toy["training_unseen"]].clone()
            final_entity, final_relation, losses = train_embeddings(
                model,
                initial_entity,
                relation_initial.clone(),
                schedule,
                learning_rate=training["learning_rate"],
                regularisation_coefficient=training["regularisation_coefficient"],
                transe_distance_norm=training["transe_distance_norm"],
                adam_beta1=training["adam_beta1"],
                adam_beta2=training["adam_beta2"],
                adam_epsilon=training["adam_epsilon"],
                weight_decay=training["weight_decay"],
            )
            unseen_deviation = float(
                torch.max(torch.abs(final_entity[toy["training_unseen"]] - unseen_initial)).cpu()
            )
            if unseen_deviation != 0.0:
                raise TrainingSmokeError("Training-unseen entity embedding changed")
            metrics = evaluate_validation(
                model,
                final_entity,
                final_relation,
                toy["evaluated_valid"],
                toy["all_valid"],
                toy["full_train_true"],
                transe_distance_norm=training["transe_distance_norm"],
            )
            model_runs.append(
                {
                    "condition": condition,
                    "sequence_sha256": sequence_sha,
                    "steps": len(losses),
                    "first_ten_loss_mean": float(np.mean(losses[:10])),
                    "last_ten_loss_mean": float(np.mean(losses[-10:])),
                    "training_unseen_rows": len(toy["training_unseen"]),
                    "training_unseen_maximum_absolute_change": unseen_deviation,
                    "validation": metrics,
                }
            )
        if len({run["sequence_sha256"] for run in model_runs}) != 1:
            raise TrainingSmokeError("Paired conditions used different training sequences")
        reports.append({"model": model, "runs": model_runs})
    return {
        "schema_version": 1,
        "status": "training_smoke_passed_pending_hardware_preflight",
        "official_result": False,
        "configuration_sha256": sha256_file(config_path),
        "embedding_manifest_sha256": sha256_file(embedding_manifest_path),
        "execution": execution,
        "git": git,
        "toy_graph": {
            "candidate_entities": len(candidate_global),
            "relations": len(toy["relation_global_indices"]),
            "positive_training_triples": len(toy["train"]),
            "validation_triples_evaluated": len(toy["evaluated_valid"]),
            "training_unseen_entities": len(toy["training_unseen"]),
            "selected_full_training_unseen_global_index": toy[
                "selected_full_training_unseen_global_index"
            ],
        },
        "shuffle": {
            "seed": shuffle_seed,
            "accepted_attempt": shuffle_attempt,
            "permutation_sha256": permutation_sha,
            "self_assignments": 0,
            "equal_evidence_hash_assignments": 0,
            "equal_vector_hash_assignments": 0,
        },
        "training_sequence_sha256_by_model": sequence_hashes,
        "models": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    try:
        report = run_smoke(
            args.config.resolve(),
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "training_smoke_failed",
            "official_result": False,
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
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
    print("Training smoke recorded in {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
