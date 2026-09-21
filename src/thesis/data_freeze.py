"""Materialize pinned benchmark sources and verify canonical split identity."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .source_audit import (
    download_file,
    require_scheduled_execution,
    sha256_file,
    validate_archive_member_name,
)
from .source_lock import canonical_json_bytes, read_json_object


SPLITS = ("train", "valid", "test")
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


class DataFreezeError(RuntimeError):
    """Raised when pinned data cannot be materialized without ambiguity."""


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _safe_relative(value: str) -> Path:
    normalised = value.replace("\\", "/")
    candidate = PurePosixPath(normalised)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise DataFreezeError("Unsafe relative artifact path: {!r}".format(value))
    return Path(*candidate.parts)


def validate_plan(plan: Mapping[str, Any], source_lock: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != 1 or plan.get("status") != "pinned_dataset_materialization_plan":
        raise DataFreezeError("Unsupported materialization plan")
    if source_lock.get("schema_version") != 1 or source_lock.get("status") != "source_lock_pending_materialization":
        raise DataFreezeError("Source lock is not ready for materialization")
    sources = source_lock.get("sources")
    datasets = plan.get("datasets")
    if not isinstance(sources, list) or not isinstance(datasets, list) or not datasets:
        raise DataFreezeError("Source lock and plan must contain lists")
    source_ids = {source.get("id") for source in sources if isinstance(source, dict)}
    if len(source_ids) != len(sources):
        raise DataFreezeError("Source lock contains duplicate or invalid IDs")
    dataset_names = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise DataFreezeError("Dataset plan must be an object")
        name = dataset.get("name")
        if not isinstance(name, str) or not DATASET_NAME_RE.fullmatch(name) or name in dataset_names:
            raise DataFreezeError("Dataset names must be unique and path-safe")
        dataset_names.add(name)
        providers = [dataset.get("canonical")] + list(dataset.get("crosschecks", []))
        for provider in providers:
            if not isinstance(provider, dict) or provider.get("source_id") not in source_ids:
                raise DataFreezeError("Dataset {} references an unknown source".format(name))
            split_specs = provider.get("splits")
            if not isinstance(split_specs, dict) or set(split_specs) != set(SPLITS):
                raise DataFreezeError("Dataset {} provider must define exactly three splits".format(name))


def _artifact_relative_path(source_id: str, artifact: Mapping[str, Any]) -> Path:
    locator = artifact.get("repository_path") or artifact.get("label")
    if not isinstance(locator, str) or not locator:
        raise DataFreezeError("Source {} has an artifact without a stable locator".format(source_id))
    return Path(source_id) / _safe_relative(locator)


def _all_locked_artifacts(source: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    artifacts = []
    metadata = source.get("metadata_artifact")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise DataFreezeError("Source metadata artifact is invalid")
        artifacts.append(metadata)
    source_artifacts = source.get("artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise DataFreezeError("Source has no locked artifacts")
    artifacts.extend(source_artifacts)
    return artifacts


def download_locked_sources(
    source_lock: Mapping[str, Any], scratch: Path
) -> Dict[str, List[Dict[str, Any]]]:
    downloaded: Dict[str, List[Dict[str, Any]]] = {}
    for source in source_lock["sources"]:
        source_id = str(source["id"])
        downloaded[source_id] = []
        for artifact in _all_locked_artifacts(source):
            expected_size = artifact.get("size_bytes")
            expected_sha = artifact.get("sha256")
            if not isinstance(expected_size, int) or expected_size <= 0:
                raise DataFreezeError("Locked artifact size is invalid")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise DataFreezeError("Locked artifact SHA-256 is invalid")
            relative = _artifact_relative_path(source_id, artifact)
            destination = scratch / "downloads" / relative
            observed = download_file(
                str(artifact["requested_url"]), destination, expected_size
            )
            if observed["size_bytes"] != expected_size or observed["sha256"] != expected_sha:
                raise DataFreezeError("Pinned artifact mismatch for {}".format(relative))
            downloaded[source_id].append(
                {"lock": artifact, "path": destination, "relative_path": relative}
            )
    return downloaded


def _find_download(
    downloaded: Mapping[str, List[Mapping[str, Any]]],
    source_id: str,
    location: Mapping[str, Any],
) -> Mapping[str, Any]:
    if source_id not in downloaded:
        raise DataFreezeError("No downloaded source named {}".format(source_id))
    selector_keys = [key for key in ("label", "repository_path") if key in location]
    if len(selector_keys) != 1:
        raise DataFreezeError("Every split location needs exactly one artifact selector")
    key = selector_keys[0]
    value = location[key]
    matches = [entry for entry in downloaded[source_id] if entry["lock"].get(key) == value]
    if len(matches) != 1:
        raise DataFreezeError("Split artifact selector is not unique")
    return matches[0]


def _read_archive_member(path: Path, archive_format: str, member_name: str) -> bytes:
    validate_archive_member_name(member_name)
    if archive_format == "zip":
        with zipfile.ZipFile(path, "r") as archive:
            try:
                info = archive.getinfo(member_name)
            except KeyError as error:
                raise DataFreezeError("Missing ZIP member {}".format(member_name)) from error
            if info.is_dir():
                raise DataFreezeError("ZIP split member is a directory")
            return archive.read(info)
    if archive_format == "tar.gz":
        with tarfile.open(path, "r:gz") as archive:
            try:
                info = archive.getmember(member_name)
            except KeyError as error:
                raise DataFreezeError("Missing tar member {}".format(member_name)) from error
            if not info.isfile():
                raise DataFreezeError("Tar split member is not a regular file")
            handle = archive.extractfile(info)
            if handle is None:
                raise DataFreezeError("Cannot read tar member {}".format(member_name))
            return handle.read()
    raise DataFreezeError("Unsupported archive format {}".format(archive_format))


def read_split_location(
    downloaded: Mapping[str, List[Mapping[str, Any]]],
    source_id: str,
    location: Mapping[str, Any],
) -> bytes:
    entry = _find_download(downloaded, source_id, location)
    member = location.get("archive_member")
    archive = entry["lock"].get("archive")
    if member is None:
        if archive is not None:
            raise DataFreezeError("Archive split location lacks an archive member")
        return entry["path"].read_bytes()
    if not isinstance(member, str) or not isinstance(archive, dict):
        raise DataFreezeError("Invalid archive split location")
    return _read_archive_member(entry["path"], str(archive["archive_format"]), member)


def provider_splits(
    provider: Mapping[str, Any],
    downloaded: Mapping[str, List[Mapping[str, Any]]],
) -> Dict[str, bytes]:
    source_id = str(provider["source_id"])
    return {
        split: read_split_location(downloaded, source_id, provider["splits"][split])
        for split in SPLITS
    }


def require_identical_crosschecks(
    canonical: Mapping[str, bytes],
    crosschecks: Sequence[Tuple[str, Mapping[str, bytes]]],
) -> List[Dict[str, Any]]:
    records = []
    for source_id, candidate in crosschecks:
        per_split = {}
        for split in SPLITS:
            identical = candidate[split] == canonical[split]
            per_split[split] = {
                "identical": identical,
                "sha256": sha256_bytes(candidate[split]),
                "size_bytes": len(candidate[split]),
            }
            if not identical:
                raise DataFreezeError(
                    "Cross-check source {} differs for {}".format(source_id, split)
                )
        records.append({"source_id": source_id, "splits": per_split})
    return records


def parse_split(raw: bytes, *, dataset: str, split: str) -> List[Tuple[str, str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataFreezeError("{} {} is not UTF-8".format(dataset, split)) from error
    rows: List[Tuple[str, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise DataFreezeError("{} {} contains a blank line".format(dataset, split))
        fields = line.split("\t")
        if len(fields) != 3 or any(not field or field != field.strip() for field in fields):
            raise DataFreezeError(
                "{} {} line {} is not a strict three-field TSV row".format(
                    dataset, split, line_number
                )
            )
        rows.append((fields[0], fields[1], fields[2]))
    if not rows:
        raise DataFreezeError("{} {} is empty".format(dataset, split))
    return rows


def audit_dataset(
    dataset: str,
    raw_splits: Mapping[str, bytes],
    expected: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[str, str, str]]]]:
    triples = {
        split: parse_split(raw_splits[split], dataset=dataset, split=split)
        for split in SPLITS
    }
    sets = {split: set(triples[split]) for split in SPLITS}
    split_audits: Dict[str, Any] = {}
    for split in SPLITS:
        duplicate_rows = len(triples[split]) - len(sets[split])
        if duplicate_rows:
            raise DataFreezeError("{} {} contains duplicate rows".format(dataset, split))
        entities = {head for head, _, _ in triples[split]} | {
            tail for _, _, tail in triples[split]
        }
        relations = {relation for _, relation, _ in triples[split]}
        expected_rows = expected.get(split + "_rows")
        if len(triples[split]) != expected_rows:
            raise DataFreezeError("{} {} row count differs from plan".format(dataset, split))
        split_audits[split] = {
            "rows": len(triples[split]),
            "unique_rows": len(sets[split]),
            "duplicate_rows": duplicate_rows,
            "entities": len(entities),
            "relations": len(relations),
            "size_bytes": len(raw_splits[split]),
            "sha256": sha256_bytes(raw_splits[split]),
        }
    overlap = {
        "train_valid": len(sets["train"] & sets["valid"]),
        "train_test": len(sets["train"] & sets["test"]),
        "valid_test": len(sets["valid"] & sets["test"]),
    }
    if any(overlap.values()):
        raise DataFreezeError("{} benchmark splits overlap".format(dataset))
    all_entities = {
        entity
        for split in SPLITS
        for head, _, tail in triples[split]
        for entity in (head, tail)
    }
    all_relations = {
        relation for split in SPLITS for _, relation, _ in triples[split]
    }
    train_entities = {
        entity for head, _, tail in triples["train"] for entity in (head, tail)
    }
    if len(all_entities) != expected.get("entities"):
        raise DataFreezeError("{} entity count differs from plan".format(dataset))
    if len(all_relations) != expected.get("relations"):
        raise DataFreezeError("{} relation count differs from plan".format(dataset))
    audit = {
        "splits": split_audits,
        "overlap_rows": overlap,
        "vocabulary": {
            "entities": len(all_entities),
            "relations": len(all_relations),
            "train_entities": len(train_entities),
            "entities_not_in_train": len(all_entities - train_entities),
            "valid_entities_not_in_train": len(
                {
                    entity
                    for head, _, tail in triples["valid"]
                    for entity in (head, tail)
                }
                - train_entities
            ),
            "test_entities_not_in_train": len(
                {
                    entity
                    for head, _, tail in triples["test"]
                    for entity in (head, tail)
                }
                - train_entities
            ),
        },
    }
    return audit, triples


def build_processed_files(
    triples: Mapping[str, Sequence[Tuple[str, str, str]]]
) -> Tuple[Dict[str, bytes], Dict[str, int]]:
    entities = sorted(
        {
            entity
            for split in SPLITS
            for head, _, tail in triples[split]
            for entity in (head, tail)
        }
    )
    relations = sorted(
        {relation for split in SPLITS for _, relation, _ in triples[split]}
    )
    entity_to_id = {entity: index for index, entity in enumerate(entities)}
    relation_to_id = {relation: index for index, relation in enumerate(relations)}
    outputs: Dict[str, bytes] = {
        "entities.tsv": "".join(
            "{}\t{}\n".format(index, entity) for index, entity in enumerate(entities)
        ).encode("utf-8"),
        "relations.tsv": "".join(
            "{}\t{}\n".format(index, relation)
            for index, relation in enumerate(relations)
        ).encode("utf-8"),
    }
    for split in SPLITS:
        outputs[split + ".tsv"] = "".join(
            "{}\t{}\t{}\n".format(
                entity_to_id[head], relation_to_id[relation], entity_to_id[tail]
            )
            for head, relation, tail in triples[split]
        ).encode("utf-8")
    return outputs, {"entities": len(entities), "relations": len(relations)}


def write_bytes_new_or_identical(path: Path, value: bytes, job_id: str) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise DataFreezeError("Refusing to overwrite different file {}".format(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + job_id)
    with temporary.open("xb") as handle:
        handle.write(value)
    try:
        os.link(str(temporary), str(path))
    except FileExistsError:
        if path.read_bytes() != value:
            raise DataFreezeError("Concurrent different output at {}".format(path))
    finally:
        temporary.unlink(missing_ok=True)


def copy_file_new_or_identical(
    source: Path, destination: Path, expected_sha256: str, expected_size: int, job_id: str
) -> None:
    if destination.exists():
        if destination.stat().st_size != expected_size or sha256_file(destination) != expected_sha256:
            raise DataFreezeError("Refusing to overwrite different artifact {}".format(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + job_id)
    with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    try:
        os.link(str(temporary), str(destination))
    except FileExistsError:
        if destination.stat().st_size != expected_size or sha256_file(destination) != expected_sha256:
            raise DataFreezeError("Concurrent different artifact at {}".format(destination))
    finally:
        temporary.unlink(missing_ok=True)


def git_state(project_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    revision = run("rev-parse", "HEAD")
    tracked_status = run("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise DataFreezeError("Tracked working tree must be clean before materialization")
    return {"commit": revision, "tracked_worktree_clean": True}


def materialize(
    source_lock_path: Path,
    plan_path: Path,
    *,
    project_root: Path,
    scratch: Path,
) -> Dict[str, Any]:
    execution = require_scheduled_execution(scratch.resolve())
    source_lock = read_json_object(source_lock_path)
    plan = read_json_object(plan_path)
    validate_plan(plan, source_lock)
    source_lock_sha = sha256_file(source_lock_path)
    plan_sha = sha256_file(plan_path)
    git = git_state(project_root)
    downloaded = download_locked_sources(source_lock, scratch)

    prepared = []
    for dataset_plan in plan["datasets"]:
        name = str(dataset_plan["name"])
        canonical = provider_splits(dataset_plan["canonical"], downloaded)
        crosscheck_values = [
            (str(provider["source_id"]), provider_splits(provider, downloaded))
            for provider in dataset_plan["crosschecks"]
        ]
        crosschecks = require_identical_crosschecks(canonical, crosscheck_values)
        audit, triples = audit_dataset(name, canonical, dataset_plan["expected"])
        processed, vocabulary = build_processed_files(triples)
        if vocabulary != {
            "entities": dataset_plan["expected"]["entities"],
            "relations": dataset_plan["expected"]["relations"],
        }:
            raise DataFreezeError("{} processed vocabulary differs from plan".format(name))
        manifest = {
            "schema_version": 1,
            "status": "pinned_dataset_materialized_pending_evidence_audit",
            "official_result": False,
            "dataset": name,
            "protocol_sha256": source_lock["protocol_sha256"],
            "source_lock_sha256": source_lock_sha,
            "materialization_plan_sha256": plan_sha,
            "canonical_source_id": dataset_plan["canonical"]["source_id"],
            "crosschecks": crosschecks,
            "mapping_rule": plan["mapping_rule"],
            "audit": audit,
            "raw_split_sha256": {
                split: sha256_bytes(canonical[split]) for split in SPLITS
            },
            "processed_sha256": {
                filename: sha256_bytes(value)
                for filename, value in sorted(processed.items())
            },
        }
        prepared.append((name, canonical, processed, manifest))

    job_id = str(execution["scheduler_job_id"])
    persisted_artifacts = []
    for source_id, entries in downloaded.items():
        for entry in entries:
            artifact = entry["lock"]
            destination = project_root / "data" / "raw" / "source_artifacts" / entry["relative_path"]
            copy_file_new_or_identical(
                entry["path"],
                destination,
                str(artifact["sha256"]),
                int(artifact["size_bytes"]),
                job_id,
            )
            persisted_artifacts.append(
                {
                    "source_id": source_id,
                    "path": str(destination.relative_to(project_root)),
                    "size_bytes": artifact["size_bytes"],
                    "sha256": artifact["sha256"],
                }
            )

    dataset_records = []
    for name, canonical, processed, manifest in prepared:
        for split in SPLITS:
            write_bytes_new_or_identical(
                project_root / "data" / "raw" / "datasets" / name / (split + ".txt"),
                canonical[split],
                job_id,
            )
        for filename, value in processed.items():
            write_bytes_new_or_identical(
                project_root / "data" / "processed" / name / filename,
                value,
                job_id,
            )
        manifest_path = project_root / "data" / "manifests" / (name + "_dataset.json")
        write_bytes_new_or_identical(manifest_path, canonical_json_bytes(manifest), job_id)
        dataset_records.append(
            {
                "dataset": name,
                "manifest": str(manifest_path.relative_to(project_root)),
                "manifest_sha256": sha256_file(manifest_path),
                "vocabulary": manifest["audit"]["vocabulary"],
                "split_rows": {
                    split: manifest["audit"]["splits"][split]["rows"] for split in SPLITS
                },
            }
        )

    return {
        "schema_version": 1,
        "status": "pinned_datasets_materialized_pending_evidence_audit",
        "official_result": False,
        "source_lock_sha256": source_lock_sha,
        "materialization_plan_sha256": plan_sha,
        "execution": execution,
        "git": git,
        "persisted_source_artifacts": persisted_artifacts,
        "datasets": dataset_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    try:
        report = materialize(
            args.source_lock.resolve(),
            args.plan.resolve(),
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "pinned_dataset_materialization_failed",
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
    print("Pinned dataset materialization recorded in {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
