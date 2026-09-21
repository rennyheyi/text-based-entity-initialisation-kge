"""Freeze and run the official text encoder on audited entity evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_freeze import git_state, sha256_bytes, write_bytes_new_or_identical
from .source_audit import FULL_SHA_RE, require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object


class TextEncoderError(RuntimeError):
    """Raised when encoder identity, runtime, or output is not auditable."""


DATASETS = ("FB15k-237", "WN18RR", "CoDEx-M")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRE_ENCODER_COLUMNS = (
    "dataset",
    "entity_id",
    "entity_index",
    "source_artifact",
    "source_identifier",
    "mapping_status",
    "evidence_status",
    "evidence_text",
    "character_length",
    "pretoken_word_count",
    "evidence_sha256",
)
ENCODED_COLUMNS = PRE_ENCODER_COLUMNS + (
    "token_length",
    "truncated",
    "normalised_vector_row_sha256",
)


def validate_encoder_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "frozen_text_encoder_configuration"
    ):
        raise TextEncoderError("Unsupported encoder configuration")
    model = config.get("model")
    runtime = config.get("runtime")
    environment = config.get("environment")
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise TextEncoderError("Encoder model/runtime configuration is invalid")
    if not isinstance(environment, dict) or not environment:
        raise TextEncoderError("Encoder environment lock is missing")
    if model.get("repo_id") != "sentence-transformers/all-MiniLM-L6-v2":
        raise TextEncoderError("Unexpected text encoder repository")
    revision = model.get("revision")
    if not isinstance(revision, str) or not FULL_SHA_RE.fullmatch(revision):
        raise TextEncoderError("Encoder revision must be a full commit SHA")
    if model.get("trust_remote_code") is not False:
        raise TextEncoderError("Remote model code must remain disabled")
    required_files = model.get("required_files")
    if (
        not isinstance(required_files, list)
        or not required_files
        or len(required_files) != len(set(required_files))
        or any(not isinstance(value, str) or not value for value in required_files)
    ):
        raise TextEncoderError("Required model-file list is invalid")
    required_runtime = {
        "device": "cuda",
        "batch_size": 256,
        "embedding_dimension": 384,
        "raw_dtype": "float32",
        "maximum_sequence_length": 256,
        "truncation": True,
        "pooling": "mean_tokens",
        "normalise_during_encoding": False,
        "post_encoding_l2_normalisation": True,
        "minimum_raw_norm": 1e-12,
        "unit_norm_absolute_tolerance": 1e-6,
    }
    if runtime != required_runtime:
        raise TextEncoderError("Encoder runtime differs from the frozen protocol")


def runtime_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
    }
    for distribution in (
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
        "huggingface-hub",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise TextEncoderError("Required package is absent: {}".format(distribution)) from error
    try:
        import torch
    except ImportError as error:
        raise TextEncoderError("PyTorch runtime cannot be imported") from error
    versions["torch_runtime"] = str(torch.__version__)
    return versions


def require_locked_environment(
    expected: Mapping[str, Any], observed: Mapping[str, str]
) -> None:
    for package, version in expected.items():
        if observed.get(package) != version:
            raise TextEncoderError(
                "Environment version mismatch for {}: expected {!r}, observed {!r}".format(
                    package, version, observed.get(package)
                )
            )


def load_pre_encoder_table(
    project_root: Path,
    dataset: str,
    *,
    protocol_sha256: str,
) -> Tuple[List[Dict[str, str]], Mapping[str, Any], Path]:
    manifest_path = (
        project_root / "data" / "manifests" / (dataset + "_evidence_pre_encoder.json")
    )
    manifest = read_json_object(manifest_path)
    if manifest.get("status") != "evidence_verified_pending_encoder_tokenization":
        raise TextEncoderError("Evidence manifest is not ready for encoding")
    if manifest.get("dataset") != dataset or manifest.get("protocol_sha256") != protocol_sha256:
        raise TextEncoderError("Evidence manifest identity differs from encoder lock")
    table_path = (
        project_root
        / "artifacts"
        / "descriptions"
        / dataset
        / "evidence_pre_encoder.tsv"
    )
    audit_table = manifest.get("audit_table", {})
    if not table_path.is_file() or sha256_file(table_path) != audit_table.get("sha256"):
        raise TextEncoderError("Pre-encoder evidence table hash mismatch")
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
        )
        if tuple(reader.fieldnames or ()) != PRE_ENCODER_COLUMNS:
            raise TextEncoderError("Pre-encoder evidence columns differ from the schema")
        records = [dict(row) for row in reader]
    if len(records) != audit_table.get("rows"):
        raise TextEncoderError("Pre-encoder evidence row count mismatch")
    for index, record in enumerate(records):
        text = record["evidence_text"]
        observed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if (
            record["dataset"] != dataset
            or record["entity_index"] != str(index)
            or not record["entity_id"]
            or not text
            or observed_hash != record["evidence_sha256"]
        ):
            raise TextEncoderError(
                "Pre-encoder evidence row alignment/hash mismatch for {} row {} "
                "entity {!r}: expected hash {!r}, observed hash {!r}".format(
                    dataset,
                    index,
                    record.get("entity_id"),
                    record.get("evidence_sha256"),
                    observed_hash,
                )
            )
    return records, manifest, manifest_path


def snapshot_inventory(snapshot_path: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(snapshot_path.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(snapshot_path).parts:
            continue
        records.append(
            {
                "path": path.relative_to(snapshot_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise TextEncoderError("Downloaded encoder snapshot is empty")
    return records


def tokenizer_lengths(tokenizer: Any, texts: Sequence[str], batch_size: int) -> List[int]:
    lengths: List[int] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != len(batch):
            raise TextEncoderError("Tokenizer did not return one ID sequence per text")
        lengths.extend(len(values) for values in input_ids)
    return lengths


def normalise_embedding_matrix(
    raw: Any,
    *,
    minimum_raw_norm: float,
    unit_norm_tolerance: float,
) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np

    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 384 or raw.dtype != np.float32:
        raise TextEncoderError("Raw embedding matrix shape/dtype is invalid")
    if not np.isfinite(raw).all():
        raise TextEncoderError("Raw embedding matrix contains a non-finite value")
    raw_norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    if not np.isfinite(raw_norms).all() or np.any(raw_norms <= minimum_raw_norm):
        raise TextEncoderError("Raw embedding matrix contains a zero/invalid norm")
    normalised = (raw.astype(np.float64) / raw_norms[:, None]).astype(np.float32)
    if not np.isfinite(normalised).all():
        raise TextEncoderError("Normalised embedding matrix contains a non-finite value")
    final_norms = np.linalg.norm(normalised.astype(np.float64), axis=1)
    maximum_deviation = float(np.max(np.abs(final_norms - 1.0)))
    if maximum_deviation > unit_norm_tolerance:
        raise TextEncoderError("Normalised embedding rows violate unit-norm tolerance")
    statistics = {
        "raw_norm": {
            "minimum": float(raw_norms.min()),
            "maximum": float(raw_norms.max()),
            "mean": float(raw_norms.mean()),
            "standard_deviation": float(raw_norms.std()),
        },
        "normalised_norm": {
            "minimum": float(final_norms.min()),
            "maximum": float(final_norms.max()),
            "mean": float(final_norms.mean()),
            "standard_deviation": float(final_norms.std()),
            "maximum_absolute_deviation_from_one": maximum_deviation,
        },
    }
    return normalised, statistics


def numpy_bytes(array: Any) -> bytes:
    import numpy as np

    handle = io.BytesIO()
    np.save(handle, array, allow_pickle=False)
    return handle.getvalue()


def encoded_table_bytes(
    records: Sequence[Mapping[str, str]],
    token_lengths: Sequence[int],
    row_hashes: Sequence[str],
    *,
    maximum_sequence_length: int,
) -> bytes:
    if not (len(records) == len(token_lengths) == len(row_hashes)):
        raise TextEncoderError("Encoded table inputs have different row counts")
    lines = ["\t".join(ENCODED_COLUMNS)]
    for record, token_length, row_hash in zip(records, token_lengths, row_hashes):
        if token_length <= 0 or not SHA256_RE.fullmatch(row_hash):
            raise TextEncoderError("Encoded table token length or row hash is invalid")
        value = dict(record)
        value.update(
            {
                "token_length": str(token_length),
                "truncated": str(token_length > maximum_sequence_length).lower(),
                "normalised_vector_row_sha256": row_hash,
            }
        )
        fields = [str(value[column]) for column in ENCODED_COLUMNS]
        if any(
            "\t" in field or "\n" in field or "\r" in field
            for field in fields
        ):
            raise TextEncoderError("Encoded evidence table contains an unsafe field")
        lines.append("\t".join(fields))
    return ("\n".join(lines) + "\n").encode("utf-8")


def encode_all(
    config_path: Path,
    *,
    project_root: Path,
    scratch: Path,
) -> Dict[str, Any]:
    execution = require_scheduled_execution(scratch.resolve())
    config = read_json_object(config_path)
    validate_encoder_config(config)
    git = git_state(project_root)
    observed_versions = runtime_versions()
    require_locked_environment(config["environment"], observed_versions)

    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise TextEncoderError("Frozen encoder environment cannot be imported") from error

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TextEncoderError("Exactly one scheduler-visible CUDA device is required")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_config = config["model"]
    runtime = config["runtime"]
    snapshot_path = scratch / "encoder_snapshot"
    resolved_path = Path(
        snapshot_download(
            repo_id=model_config["repo_id"],
            revision=model_config["revision"],
            local_dir=snapshot_path,
            allow_patterns=model_config["required_files"],
        )
    ).resolve()
    if resolved_path != snapshot_path.resolve():
        raise TextEncoderError("Encoder snapshot resolved outside job scratch")
    inventory = snapshot_inventory(snapshot_path)
    observed_files = {record["path"] for record in inventory}
    if observed_files != set(model_config["required_files"]):
        raise TextEncoderError("Downloaded encoder file set differs from the lock")

    model = SentenceTransformer(
        str(snapshot_path),
        device="cuda",
        trust_remote_code=False,
    )
    model.eval()
    if model.get_sentence_embedding_dimension() != runtime["embedding_dimension"]:
        raise TextEncoderError("Sentence embedding dimension differs from the lock")
    if model.max_seq_length != runtime["maximum_sequence_length"]:
        raise TextEncoderError("Maximum sequence length differs from the lock")
    pooling_config = read_json_object(snapshot_path / "1_Pooling" / "config.json")
    if not (
        pooling_config.get("pooling_mode_mean_tokens") is True
        and pooling_config.get("pooling_mode_cls_token") is False
        and pooling_config.get("pooling_mode_max_tokens") is False
    ):
        raise TextEncoderError("Pooling configuration is not mean-token pooling")

    snapshot_manifest = {
        "schema_version": 1,
        "status": "frozen_text_encoder_runtime",
        "official_result": False,
        "protocol_sha256": config["protocol_sha256"],
        "encoder_lock_sha256": sha256_file(config_path),
        "model": model_config,
        "runtime": runtime,
        "environment": observed_versions,
        "snapshot_files": inventory,
        "snapshot_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "tokenizer": {
            "class": type(model.tokenizer).__name__,
            "vocabulary_size": len(model.tokenizer),
            "model_max_length": model.tokenizer.model_max_length,
            "padding_side": model.tokenizer.padding_side,
            "truncation_side": model.tokenizer.truncation_side,
        },
        "pooling_config": pooling_config,
        "cuda": {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0),
        },
    }
    encoder_manifest_bytes = canonical_json_bytes(snapshot_manifest)
    encoder_manifest_sha = sha256_bytes(encoder_manifest_bytes)

    prepared = []
    for dataset in DATASETS:
        records, evidence_manifest, evidence_manifest_path = load_pre_encoder_table(
            project_root,
            dataset,
            protocol_sha256=config["protocol_sha256"],
        )
        texts = [record["evidence_text"] for record in records]
        lengths = tokenizer_lengths(model.tokenizer, texts, runtime["batch_size"])
        with torch.inference_mode():
            raw = model.encode(
                texts,
                batch_size=runtime["batch_size"],
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        raw = np.asarray(raw)
        if raw.shape != (len(records), runtime["embedding_dimension"]):
            raise TextEncoderError("Encoded matrix row/dimension count is invalid")
        normalised, norm_statistics = normalise_embedding_matrix(
            raw,
            minimum_raw_norm=runtime["minimum_raw_norm"],
            unit_norm_tolerance=runtime["unit_norm_absolute_tolerance"],
        )
        raw_bytes = numpy_bytes(raw)
        normalised_bytes = numpy_bytes(normalised)
        row_hashes = [
            hashlib.sha256(row.tobytes(order="C")).hexdigest()
            for row in normalised
        ]
        final_table = encoded_table_bytes(
            records,
            lengths,
            row_hashes,
            maximum_sequence_length=runtime["maximum_sequence_length"],
        )
        truncated = sum(length > runtime["maximum_sequence_length"] for length in lengths)
        manifest = {
            "schema_version": 1,
            "status": "text_embeddings_frozen_pending_training_pipeline",
            "official_result": False,
            "dataset": dataset,
            "protocol_sha256": config["protocol_sha256"],
            "encoder_lock_sha256": sha256_file(config_path),
            "encoder_runtime_manifest_sha256": encoder_manifest_sha,
            "pre_encoder_evidence_manifest_sha256": sha256_file(evidence_manifest_path),
            "pre_encoder_evidence_table_sha256": evidence_manifest["audit_table"]["sha256"],
            "rows": len(records),
            "embedding_dimension": runtime["embedding_dimension"],
            "dtype": runtime["raw_dtype"],
            "token_length": {
                "minimum": min(lengths),
                "maximum": max(lengths),
                "total": sum(lengths),
                "truncated_rows": truncated,
                "truncated_proportion": truncated / len(lengths),
                "maximum_sequence_length": runtime["maximum_sequence_length"],
            },
            "norm_statistics": norm_statistics,
            "artifacts": {
                "raw_embeddings": {
                    "filename": "raw_text_embeddings.npy",
                    "sha256": sha256_bytes(raw_bytes),
                    "size_bytes": len(raw_bytes),
                },
                "normalised_embeddings": {
                    "filename": "normalised_text_embeddings.npy",
                    "sha256": sha256_bytes(normalised_bytes),
                    "size_bytes": len(normalised_bytes),
                },
                "encoded_evidence_table": {
                    "filename": "evidence_encoded.tsv",
                    "sha256": sha256_bytes(final_table),
                    "size_bytes": len(final_table),
                },
            },
        }
        prepared.append(
            (dataset, raw_bytes, normalised_bytes, final_table, manifest)
        )

    job_id = str(execution["scheduler_job_id"])
    encoder_manifest_path = project_root / "data" / "manifests" / "text_encoder_runtime.json"
    write_bytes_new_or_identical(encoder_manifest_path, encoder_manifest_bytes, job_id)
    dataset_reports = []
    for dataset, raw_bytes, normalised_bytes, final_table, manifest in prepared:
        embedding_root = project_root / "artifacts" / "embeddings" / dataset
        description_root = project_root / "artifacts" / "descriptions" / dataset
        write_bytes_new_or_identical(
            embedding_root / "raw_text_embeddings.npy", raw_bytes, job_id
        )
        write_bytes_new_or_identical(
            embedding_root / "normalised_text_embeddings.npy",
            normalised_bytes,
            job_id,
        )
        write_bytes_new_or_identical(
            description_root / "evidence_encoded.tsv", final_table, job_id
        )
        manifest_path = project_root / "data" / "manifests" / (dataset + "_text_embeddings.json")
        write_bytes_new_or_identical(
            manifest_path, canonical_json_bytes(manifest), job_id
        )
        dataset_reports.append(
            {
                "dataset": dataset,
                "rows": manifest["rows"],
                "truncated_rows": manifest["token_length"]["truncated_rows"],
                "raw_embeddings_sha256": manifest["artifacts"]["raw_embeddings"]["sha256"],
                "normalised_embeddings_sha256": manifest["artifacts"]["normalised_embeddings"]["sha256"],
                "manifest": str(manifest_path.relative_to(project_root)),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return {
        "schema_version": 1,
        "status": "text_embeddings_frozen_pending_training_pipeline",
        "official_result": False,
        "encoder_lock_sha256": sha256_file(config_path),
        "encoder_runtime_manifest": str(encoder_manifest_path.relative_to(project_root)),
        "encoder_runtime_manifest_sha256": sha256_file(encoder_manifest_path),
        "execution": execution,
        "git": git,
        "datasets": dataset_reports,
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
        report = encode_all(
            args.config.resolve(),
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "text_encoding_failed",
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
    print("Text encoding recorded in {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
