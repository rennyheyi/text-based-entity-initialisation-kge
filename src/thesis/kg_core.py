"""Minimal KGE scoring, sampling, training, and filtered-ranking primitives."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


class KGCoreError(RuntimeError):
    """Raised when a core KGE invariant is violated."""


def score_triples(
    model: str,
    entity_embeddings: Any,
    relation_embeddings: Any,
    triples: Any,
    *,
    transe_distance_norm: int = 1,
) -> Any:
    """Score integer triples with higher values denoting greater plausibility."""

    import torch

    if triples.ndim != 2 or triples.shape[1] != 3:
        raise KGCoreError("Triples must have shape [N, 3]")
    head = entity_embeddings[triples[:, 0]]
    relation = relation_embeddings[triples[:, 1]]
    tail = entity_embeddings[triples[:, 2]]
    if model == "TransE":
        if transe_distance_norm not in {1, 2}:
            raise KGCoreError("TransE distance norm must be one or two")
        return -torch.linalg.vector_norm(
            head + relation - tail,
            ord=transe_distance_norm,
            dim=-1,
        )
    if model == "DistMult":
        return (head * relation * tail).sum(dim=-1)
    raise KGCoreError("Unknown KGE model")


def score_candidates(
    model: str,
    entity_embeddings: Any,
    relation_embeddings: Any,
    triple: Sequence[int],
    *,
    direction: str,
    transe_distance_norm: int = 1,
    chunk_size: int | None = None,
) -> Any:
    """Score the complete entity vocabulary for one head or tail query."""

    import torch

    if direction not in {"head", "tail"}:
        raise KGCoreError("Prediction direction must be head or tail")
    entity_count = int(entity_embeddings.shape[0])
    if chunk_size is None:
        chunk_size = entity_count
    if chunk_size <= 0:
        raise KGCoreError("Candidate chunk size must be positive")
    head, relation, tail = map(int, triple)
    outputs = []
    for start in range(0, entity_count, chunk_size):
        stop = min(start + chunk_size, entity_count)
        candidates = torch.arange(start, stop, device=entity_embeddings.device)
        if direction == "tail":
            triples = torch.stack(
                (
                    torch.full_like(candidates, head),
                    torch.full_like(candidates, relation),
                    candidates,
                ),
                dim=1,
            )
        else:
            triples = torch.stack(
                (
                    candidates,
                    torch.full_like(candidates, relation),
                    torch.full_like(candidates, tail),
                ),
                dim=1,
            )
        outputs.append(
            score_triples(
                model,
                entity_embeddings,
                relation_embeddings,
                triples,
                transe_distance_norm=transe_distance_norm,
            )
        )
    return torch.cat(outputs, dim=0)


def realistic_filtered_rank(
    scores: Any,
    *,
    target_index: int,
    filtered_indices: Iterable[int],
) -> float:
    """Compute exact realistic rank from one scored candidate vector."""

    import numpy as np

    values = np.asarray(scores)
    if values.ndim != 1 or not 0 <= target_index < len(values):
        raise KGCoreError("Candidate scores/target index are invalid")
    if not np.isfinite(values).all():
        raise KGCoreError("Candidate scores contain a non-finite value")
    keep = np.ones(len(values), dtype=bool)
    for index in filtered_indices:
        if not 0 <= int(index) < len(values):
            raise KGCoreError("Filtered entity index is outside the vocabulary")
        if int(index) != target_index:
            keep[int(index)] = False
    keep[target_index] = True
    if not keep[target_index] or not keep.any():
        raise KGCoreError("Filtered candidate set lost its target")
    target_score = values[target_index]
    kept = values[keep]
    greater = int(np.sum(kept > target_score))
    equal_other = int(np.sum(kept == target_score)) - 1
    if equal_other < 0:
        raise KGCoreError("Target score is absent from the filtered candidates")
    return 1.0 + greater + 0.5 * equal_other


def ranking_metrics(ranks: Sequence[float]) -> Dict[str, float | int]:
    import numpy as np

    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise KGCoreError("Ranks must be a non-empty finite vector")
    return {
        "queries": int(len(values)),
        "mrr": float(np.mean(1.0 / values)),
        "hits_at_1": float(np.mean(values <= 1.0)),
        "hits_at_3": float(np.mean(values <= 3.0)),
        "hits_at_10": float(np.mean(values <= 10.0)),
    }


def sample_negative_triples(
    positives: Any,
    *,
    negatives_per_positive: int,
    replacement_entities: Any,
    true_training_triples: Set[Tuple[int, int, int]],
    rng: Any,
    maximum_attempts: int,
) -> Any:
    """Uniformly corrupt one side and reject all known training positives."""

    import numpy as np

    positives = np.asarray(positives, dtype=np.int64)
    population = np.asarray(replacement_entities, dtype=np.int64)
    if positives.ndim != 2 or positives.shape[1] != 3 or len(positives) == 0:
        raise KGCoreError("Positive triples are invalid")
    if negatives_per_positive <= 0 or len(population) < 2:
        raise KGCoreError("Negative-sampling population/size is invalid")
    negatives = np.empty(
        (len(positives), negatives_per_positive, 3), dtype=np.int64
    )
    for row_index, positive in enumerate(positives):
        for negative_index in range(negatives_per_positive):
            accepted = None
            for _ in range(maximum_attempts):
                candidate = positive.copy()
                side = int(rng.integers(0, 2))
                column = 0 if side == 0 else 2
                replacement = int(population[int(rng.integers(0, len(population)))])
                if replacement == int(positive[column]):
                    continue
                candidate[column] = replacement
                candidate_tuple = tuple(map(int, candidate))
                if candidate_tuple in true_training_triples:
                    continue
                accepted = candidate
                break
            if accepted is None:
                raise KGCoreError("Negative sampling exceeded its attempt limit")
            negatives[row_index, negative_index] = accepted
    return negatives


def triple_keys(
    triples: Any, *, entity_count: int, relation_count: int
) -> Any:
    """Encode integer triples as collision-free signed 64-bit keys."""

    import numpy as np

    values = np.asarray(triples, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise KGCoreError("Triples for key encoding must have shape [N, 3]")
    if entity_count <= 0 or relation_count <= 0:
        raise KGCoreError("Vocabulary sizes for key encoding must be positive")
    if len(values) and (
        np.any(values[:, 0] < 0)
        or np.any(values[:, 0] >= entity_count)
        or np.any(values[:, 1] < 0)
        or np.any(values[:, 1] >= relation_count)
        or np.any(values[:, 2] < 0)
        or np.any(values[:, 2] >= entity_count)
    ):
        raise KGCoreError("A triple index is outside the frozen vocabulary")
    maximum_key = (
        (entity_count - 1) * relation_count * entity_count
        + (relation_count - 1) * entity_count
        + entity_count
        - 1
    )
    if maximum_key > np.iinfo(np.int64).max:
        raise KGCoreError("Triple-key encoding exceeds signed 64-bit range")
    return (
        values[:, 0] * relation_count * entity_count
        + values[:, 1] * entity_count
        + values[:, 2]
    )


def sample_negative_triples_vectorized(
    positives: Any,
    *,
    negatives_per_positive: int,
    replacement_entities: Any,
    sorted_true_training_keys: Any,
    entity_count: int,
    relation_count: int,
    rng: Any,
    maximum_attempts: int,
) -> Any:
    """Vectorised uniform corruption with exact training-truth rejection."""

    import numpy as np

    positives = np.asarray(positives, dtype=np.int64)
    population = np.asarray(replacement_entities, dtype=np.int64)
    truth = np.asarray(sorted_true_training_keys, dtype=np.int64)
    if positives.ndim != 2 or positives.shape[1] != 3 or not len(positives):
        raise KGCoreError("Positive triples are invalid")
    if negatives_per_positive <= 0 or len(population) < 2:
        raise KGCoreError("Negative-sampling population/size is invalid")
    if maximum_attempts <= 0 or np.any(population < 0) or np.any(
        population >= entity_count
    ):
        raise KGCoreError("Negative-sampling bounds are invalid")
    if truth.ndim != 1 or len(truth) == 0 or np.any(truth[1:] < truth[:-1]):
        raise KGCoreError("True-training triple keys must be non-empty and sorted")
    repeated = np.repeat(positives, negatives_per_positive, axis=0)
    output = repeated.copy()
    unresolved = np.ones(len(repeated), dtype=bool)
    for _ in range(maximum_attempts):
        indices = np.flatnonzero(unresolved)
        if not len(indices):
            break
        base = repeated[indices]
        sides = rng.integers(0, 2, size=len(indices), dtype=np.int8)
        columns = np.where(sides == 0, 0, 2)
        replacements = population[
            rng.integers(0, len(population), size=len(indices))
        ]
        candidates = base.copy()
        candidates[np.arange(len(indices)), columns] = replacements
        distinct = replacements != base[np.arange(len(indices)), columns]
        keys = triple_keys(
            candidates, entity_count=entity_count, relation_count=relation_count
        )
        locations = np.searchsorted(truth, keys)
        in_truth = np.zeros(len(keys), dtype=bool)
        inside = locations < len(truth)
        in_truth[inside] = truth[locations[inside]] == keys[inside]
        accepted = distinct & ~in_truth
        accepted_indices = indices[accepted]
        output[accepted_indices] = candidates[accepted]
        unresolved[accepted_indices] = False
    if np.any(unresolved):
        raise KGCoreError("Vectorised negative sampling exceeded its attempt limit")
    return output.reshape(len(positives), negatives_per_positive, 3)


def batch_training_loss(
    model: str,
    entity: Any,
    relation: Any,
    positives: Any,
    negatives: Any,
    *,
    regularisation_coefficient: float,
    transe_distance_norm: int,
) -> Tuple[Any, Any, Any]:
    """Return frozen pairwise ranking, batch-local L2, and total losses."""

    import torch

    if positives.ndim != 2 or positives.shape[1] != 3:
        raise KGCoreError("Positive training batch has an invalid shape")
    if (
        negatives.ndim != 3
        or negatives.shape[0] != positives.shape[0]
        or negatives.shape[2] != 3
    ):
        raise KGCoreError("Negative training batch has an invalid shape")
    positive_scores = score_triples(
        model,
        entity,
        relation,
        positives,
        transe_distance_norm=transe_distance_norm,
    )
    flattened_negatives = negatives.reshape(-1, 3)
    negative_scores = score_triples(
        model,
        entity,
        relation,
        flattened_negatives,
        transe_distance_norm=transe_distance_norm,
    ).reshape(negatives.shape[0], negatives.shape[1])
    ranking_loss = torch.nn.functional.softplus(
        negative_scores - positive_scores[:, None]
    ).mean()
    referenced = torch.cat((positives, flattened_negatives), dim=0)
    unique_entities = torch.unique(referenced[:, [0, 2]])
    unique_relations = torch.unique(referenced[:, 1])
    regularisation = (
        entity[unique_entities].square().sum(dim=1).mean()
        + relation[unique_relations].square().sum(dim=1).mean()
    )
    total = ranking_loss + regularisation_coefficient * regularisation
    return ranking_loss, regularisation, total


def make_training_schedule(
    triples: Any,
    *,
    epochs: int,
    batch_size: int,
    negatives_per_positive: int,
    replacement_entities: Any,
    true_training_triples: Set[Tuple[int, int, int]],
    batch_seed: int,
    negative_seed: int,
    maximum_attempts: int,
) -> Tuple[List[Tuple[Any, Any]], str]:
    """Precompute a paired condition-independent batch/negative trajectory."""

    import numpy as np

    triples = np.asarray(triples, dtype=np.int64)
    if epochs <= 0 or batch_size <= 0:
        raise KGCoreError("Epoch count and batch size must be positive")
    batch_rng = np.random.default_rng(batch_seed)
    negative_rng = np.random.default_rng(negative_seed)
    digest = hashlib.sha256()
    schedule = []
    for epoch in range(epochs):
        order = batch_rng.permutation(len(triples))
        digest.update(epoch.to_bytes(8, "big"))
        digest.update(order.astype("<i8", copy=False).tobytes())
        for start in range(0, len(order), batch_size):
            positives = triples[order[start : start + batch_size]].copy()
            negatives = sample_negative_triples(
                positives,
                negatives_per_positive=negatives_per_positive,
                replacement_entities=replacement_entities,
                true_training_triples=true_training_triples,
                rng=negative_rng,
                maximum_attempts=maximum_attempts,
            )
            digest.update(positives.astype("<i8", copy=False).tobytes())
            digest.update(negatives.astype("<i8", copy=False).tobytes())
            schedule.append((positives, negatives))
    return schedule, digest.hexdigest()


def train_embeddings(
    model: str,
    entity_initialisation: Any,
    relation_initialisation: Any,
    schedule: Sequence[Tuple[Any, Any]],
    *,
    learning_rate: float,
    regularisation_coefficient: float,
    transe_distance_norm: int,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    weight_decay: float,
) -> Tuple[Any, Any, List[float]]:
    """Train entity/relation matrices using the frozen pairwise objective."""

    import torch

    entity = torch.nn.Parameter(entity_initialisation.clone())
    relation = torch.nn.Parameter(relation_initialisation.clone())
    optimiser = torch.optim.Adam(
        [entity, relation],
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
    )
    losses = []
    for positives_cpu, negatives_cpu in schedule:
        positives = torch.as_tensor(
            positives_cpu, dtype=torch.long, device=entity.device
        )
        negatives = torch.as_tensor(
            negatives_cpu, dtype=torch.long, device=entity.device
        )
        _, _, loss = batch_training_loss(
            model,
            entity,
            relation,
            positives,
            negatives,
            regularisation_coefficient=regularisation_coefficient,
            transe_distance_norm=transe_distance_norm,
        )
        if not torch.isfinite(loss):
            raise KGCoreError("Training loss is non-finite")
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if not torch.isfinite(entity).all() or not torch.isfinite(relation).all():
            raise KGCoreError("Training produced non-finite embeddings")
        losses.append(float(loss.detach().cpu()))
    return entity.detach(), relation.detach(), losses
