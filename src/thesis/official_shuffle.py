"""Materialise the 15 frozen official shuffled-text permutations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .data_freeze import git_state, write_bytes_new_or_identical
from .official_design import validate_design_invariants
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object
from .training_smoke import load_encoded_hashes, make_valid_shuffle


class OfficialShuffleError(RuntimeError):
    """Raised when an official shuffle cannot be materialised exactly."""


OFFICIAL_DESIGN_SHA256 = (
    "48a82e2bf835942f1d76ec40ce0b9b52499aa3a0c16f40ab31c04652215a6ee5"
)
MAXIMUM_ATTEMPTS = 1000


def numpy_bytes(array: Any) -> bytes:
    import numpy as np

    handle = io.BytesIO()
    np.save(handle, array, allow_pickle=False)
    return handle.getvalue()


def verify_permutation(
    permutation: Any,
    evidence_hashes: Sequence[str],
    vector_hashes: Sequence[str],
) -> Dict[str, Any]:
    import numpy as np

    values = np.asarray(permutation, dtype=np.int64)
    size = len(evidence_hashes)
    if len(vector_hashes) != size or values.shape != (size,):
        raise OfficialShuffleError("Shuffle populations or shape differ")
    indices = np.arange(size, dtype=np.int64)
    if not np.array_equal(np.sort(values), indices):
        raise OfficialShuffleError("Official shuffle is not a bijection")
    self_assignments = int(np.sum(values == indices))
    equal_evidence = sum(
        evidence_hashes[int(source)] == evidence_hashes[target]
        for target, source in enumerate(values)
    )
    equal_vectors = sum(
        vector_hashes[int(source)] == vector_hashes[target]
        for target, source in enumerate(values)
    )
    if self_assignments or equal_evidence or equal_vectors:
        raise OfficialShuffleError("Official shuffle violates derangement/hash rules")
    if sorted(vector_hashes[int(index)] for index in values) != sorted(vector_hashes):
        raise OfficialShuffleError("Correct and shuffled vector multisets differ")
    return {
        "rows": size,
        "self_assignments": self_assignments,
        "equal_evidence_hash_assignments": equal_evidence,
        "equal_vector_hash_assignments": equal_vectors,
        "correct_and_shuffled_vector_multisets_identical": True,
        "permutation_raw_sha256": hashlib.sha256(
            values.astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest(),
    }


def load_dataset_evidence(
    project_root: Path,
    dataset: str,
    resource: Mapping[str, Any],
) -> Tuple[Sequence[str], Sequence[str], Mapping[str, Any]]:
    import numpy as np

    manifest_path = project_root / str(resource["text_embedding_manifest"])
    if sha256_file(manifest_path) != resource["text_embedding_manifest_sha256"]:
        raise OfficialShuffleError("Text embedding manifest hash differs for " + dataset)
    manifest = read_json_object(manifest_path)
    artifact = manifest["artifacts"]["normalised_embeddings"]
    embedding_path = project_root / "artifacts" / "embeddings" / dataset / artifact["filename"]
    if artifact["sha256"] != resource["normalised_embeddings_sha256"]:
        raise OfficialShuffleError("Manifest/design embedding hash differs for " + dataset)
    if sha256_file(embedding_path) != artifact["sha256"]:
        raise OfficialShuffleError("Normalised embedding artifact hash differs for " + dataset)
    matrix = np.load(embedding_path, allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != 384 or matrix.dtype != np.float32:
        raise OfficialShuffleError("Normalised embedding shape/dtype differs for " + dataset)
    evidence_table = project_root / "artifacts" / "descriptions" / dataset / "evidence_encoded.tsv"
    encoded = manifest["artifacts"]["encoded_evidence_table"]
    if sha256_file(evidence_table) != encoded["sha256"]:
        raise OfficialShuffleError("Encoded evidence table hash differs for " + dataset)
    evidence_hashes, vector_hashes = load_encoded_hashes(evidence_table, len(matrix))
    for index, row in enumerate(matrix):
        if hashlib.sha256(row.tobytes(order="C")).hexdigest() != vector_hashes[index]:
            raise OfficialShuffleError("Embedding row hash differs for " + dataset)
    return evidence_hashes, vector_hashes, {
        "manifest_path": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "embedding_path": str(embedding_path.relative_to(project_root)),
        "embedding_sha256": sha256_file(embedding_path),
        "encoded_evidence_table": str(evidence_table.relative_to(project_root)),
        "encoded_evidence_table_sha256": sha256_file(evidence_table),
        "rows": len(matrix),
    }


def materialise_official_shuffles(
    design_path: Path,
    *,
    project_root: Path,
    scratch: Path,
) -> Dict[str, Any]:
    import numpy as np

    execution = require_scheduled_execution(scratch.resolve())
    if sha256_file(design_path) != OFFICIAL_DESIGN_SHA256:
        raise OfficialShuffleError("Official design hash differs")
    design = read_json_object(design_path)
    validate_design_invariants(design)
    git = git_state(project_root)
    records = []
    for dataset in design["dataset_order"]:
        evidence_hashes, vector_hashes, source = load_dataset_evidence(
            project_root, dataset, design["dataset_resources"][dataset]
        )
        specifications = [
            row for row in design["shuffle_artifacts"] if row["dataset"] == dataset
        ]
        if len(specifications) != 5:
            raise OfficialShuffleError("Official dataset does not have five shuffles")
        for specification in specifications:
            seed = int(specification["derivation"]["seed"])
            permutation, accepted_attempt, generated_sha = make_valid_shuffle(
                evidence_hashes,
                vector_hashes,
                seed=seed,
                maximum_attempts=MAXIMUM_ATTEMPTS,
            )
            permutation = np.asarray(permutation, dtype=np.int64)
            invariants = verify_permutation(
                permutation, evidence_hashes, vector_hashes
            )
            if generated_sha != invariants["permutation_raw_sha256"]:
                raise OfficialShuffleError("Generated permutation hash differs")
            artifact_path = project_root / specification["permutation_artifact"]
            artifact_bytes = numpy_bytes(permutation)
            write_bytes_new_or_identical(
                artifact_path,
                artifact_bytes,
                str(execution["scheduler_job_id"]),
            )
            manifest = {
                "schema_version": 1,
                "status": "official_shuffle_materialised_pending_training",
                "official_result": False,
                "shuffle_id": specification["shuffle_id"],
                "dataset": dataset,
                "official_replicate": specification["official_replicate"],
                "seed_derivation": specification["derivation"],
                "maximum_attempts": MAXIMUM_ATTEMPTS,
                "accepted_attempt": accepted_attempt,
                "source": source,
                "permutation_artifact": {
                    "path": specification["permutation_artifact"],
                    "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                    "size_bytes": len(artifact_bytes),
                    "dtype": "int64",
                    "shape": [len(permutation)],
                },
                "invariants": invariants,
                "official_design_sha256": OFFICIAL_DESIGN_SHA256,
            }
            manifest_path = project_root / specification["permutation_manifest"]
            manifest_bytes = canonical_json_bytes(manifest)
            write_bytes_new_or_identical(
                manifest_path,
                manifest_bytes,
                str(execution["scheduler_job_id"]),
            )
            records.append(
                {
                    "shuffle_id": specification["shuffle_id"],
                    "dataset": dataset,
                    "official_replicate": specification["official_replicate"],
                    "permutation_artifact": specification["permutation_artifact"],
                    "permutation_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                    "permutation_manifest": specification["permutation_manifest"],
                    "permutation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "accepted_attempt": accepted_attempt,
                    **invariants,
                }
            )
    if len(records) != 15:
        raise OfficialShuffleError("Official shuffle count differs from 15")
    return {
        "schema_version": 1,
        "status": "official_shuffles_materialised_pending_training",
        "official_result": False,
        "test_data_accessed": False,
        "official_design_sha256": OFFICIAL_DESIGN_SHA256,
        "shuffle_count": len(records),
        "shuffle_manifest_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
        "shuffles": records,
        "execution": execution,
        "git": git,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    try:
        report = materialise_official_shuffles(
            args.design.resolve(),
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "official_shuffle_materialisation_failed",
            "official_result": False,
            "test_data_accessed": False,
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
    print("Official shuffle materialisation recorded in {}".format(output))
    print(json.dumps({"status": report["status"], "shuffle_count": 15}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
