"""Audit complete-vocabulary textual evidence from pinned local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_freeze import (
    _read_archive_member,
    git_state,
    sha256_bytes,
    write_bytes_new_or_identical,
)
from .source_audit import require_scheduled_execution, sha256_file
from .source_lock import canonical_json_bytes, read_json_object
from .wordnet30 import (
    WordNet30Error,
    mapping_gloss_alignment,
    normalise_text,
    parse_data_file,
    parse_sense_index,
    verify_mapping_row,
)
from .wordnet_mapping import WordNetMappingRow, parse_definition_line


class EvidenceAuditError(RuntimeError):
    """Raised when evidence is missing, ambiguous, or incompatible."""


def _source_by_id(source_lock: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    matches = [source for source in source_lock["sources"] if source.get("id") == source_id]
    if len(matches) != 1:
        raise EvidenceAuditError("Expected exactly one locked source {}".format(source_id))
    return matches[0]


def locked_artifact_path(
    project_root: Path,
    source_lock: Mapping[str, Any],
    location: Mapping[str, Any],
) -> Tuple[Path, Mapping[str, Any]]:
    source_id = str(location["source_id"])
    source = _source_by_id(source_lock, source_id)
    selector_keys = [key for key in ("repository_path", "label") if key in location]
    if len(selector_keys) != 1:
        raise EvidenceAuditError("Evidence location needs exactly one artifact selector")
    key = selector_keys[0]
    value = location[key]
    artifacts = list(source.get("artifacts", []))
    metadata = source.get("metadata_artifact")
    if metadata is not None:
        artifacts.append(metadata)
    matches = [artifact for artifact in artifacts if artifact.get(key) == value]
    if len(matches) != 1:
        raise EvidenceAuditError("Evidence artifact selector is not unique")
    artifact = matches[0]
    relative = artifact.get("repository_path") or artifact.get("label")
    path = project_root / "data" / "raw" / "source_artifacts" / source_id / str(relative)
    if not path.is_file():
        raise EvidenceAuditError("Pinned evidence artifact is missing locally")
    if path.stat().st_size != artifact["size_bytes"] or sha256_file(path) != artifact["sha256"]:
        raise EvidenceAuditError("Pinned evidence artifact hash/size mismatch")
    return path, artifact


def load_vocabulary(
    project_root: Path,
    dataset: str,
    source_lock_sha256: str,
) -> Tuple[List[str], Mapping[str, Any]]:
    manifest_path = project_root / "data" / "manifests" / (dataset + "_dataset.json")
    manifest = read_json_object(manifest_path)
    if manifest.get("status") != "pinned_dataset_materialized_pending_evidence_audit":
        raise EvidenceAuditError("Dataset manifest is not ready for evidence audit")
    if manifest.get("source_lock_sha256") != source_lock_sha256:
        raise EvidenceAuditError("Dataset manifest uses a different source lock")
    entities_path = project_root / "data" / "processed" / dataset / "entities.tsv"
    expected_hash = manifest.get("processed_sha256", {}).get("entities.tsv")
    if not isinstance(expected_hash, str) or sha256_file(entities_path) != expected_hash:
        raise EvidenceAuditError("Entity mapping differs from its dataset manifest")
    entities = []
    for line_number, line in enumerate(entities_path.read_text(encoding="utf-8").splitlines()):
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != str(line_number) or not fields[1]:
            raise EvidenceAuditError("Entity mapping is not contiguous and two-column")
        entities.append(fields[1])
    if len(entities) != manifest["audit"]["vocabulary"]["entities"]:
        raise EvidenceAuditError("Entity mapping count differs from dataset manifest")
    return entities, manifest


def parse_two_field_mapping(path: Path, *, language_wrapped: bool = False) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise EvidenceAuditError("Malformed two-field mapping line {}".format(line_number))
        identifier, value = fields
        if identifier in result:
            raise EvidenceAuditError("Duplicate identifier in evidence mapping")
        if language_wrapped:
            value = value.strip()
            if not value.startswith('"') or not value.endswith('"@en'):
                raise EvidenceAuditError("Description is not explicitly tagged as English")
            value = value[1:-4]
        value = normalise_text(value)
        if not value:
            raise EvidenceAuditError("Evidence mapping contains an empty value")
        result[identifier] = value
    if not result:
        raise EvidenceAuditError("Evidence mapping is empty")
    return result


def _human_readable(value: str, identifier: str) -> bool:
    return value != identifier and any(character.isalnum() for character in value)


def _record(
    *,
    dataset: str,
    entity_index: int,
    entity_id: str,
    source_artifact: str,
    mapping_status: str,
    evidence_status: str,
    evidence_text: str,
) -> Dict[str, Any]:
    evidence_text = normalise_text(evidence_text)
    if not evidence_text or "\t" in evidence_text or "\n" in evidence_text:
        raise EvidenceAuditError("Constructed evidence is empty or not TSV-safe")
    return {
        "dataset": dataset,
        "entity_id": entity_id,
        "entity_index": entity_index,
        "source_artifact": source_artifact,
        "source_identifier": entity_id,
        "mapping_status": mapping_status,
        "evidence_status": evidence_status,
        "evidence_text": evidence_text,
        "character_length": len(evidence_text),
        "pretoken_word_count": len(evidence_text.split()),
        "evidence_sha256": hashlib.sha256(evidence_text.encode("utf-8")).hexdigest(),
    }


def audit_freebase(
    dataset_plan: Mapping[str, Any],
    vocabulary: Sequence[str],
    *,
    project_root: Path,
    source_lock: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    name_path, name_artifact = locked_artifact_path(
        project_root, source_lock, dataset_plan["name_source"]
    )
    description_path, description_artifact = locked_artifact_path(
        project_root, source_lock, dataset_plan["description_source"]
    )
    names = parse_two_field_mapping(name_path)
    descriptions = parse_two_field_mapping(description_path, language_wrapped=True)
    vocabulary_set = set(vocabulary)
    records = []
    status_counts = Counter()
    for entity_index, entity_id in enumerate(vocabulary):
        name = normalise_text(names.get(entity_id, "").replace("_", " "))
        description = descriptions.get(entity_id, "")
        if not _human_readable(name, entity_id):
            raise EvidenceAuditError("FB15k-237 entity lacks a valid human-readable name")
        if description:
            evidence_status = "full"
            mapping_status = "verified_name_and_description"
            evidence_text = name + ". " + description
        else:
            evidence_status = "name_only"
            mapping_status = "verified_name_only"
            evidence_text = name
        status_counts[evidence_status] += 1
        records.append(
            _record(
                dataset="FB15k-237",
                entity_index=entity_index,
                entity_id=entity_id,
                source_artifact="{}|{}".format(
                    name_artifact["repository_path"],
                    description_artifact["repository_path"],
                ),
                mapping_status=mapping_status,
                evidence_status=evidence_status,
                evidence_text=evidence_text,
            )
        )
    return records, {
        "status_counts": dict(status_counts),
        "name_mapping_rows": len(names),
        "description_mapping_rows": len(descriptions),
        "extra_name_rows": len(set(names) - vocabulary_set),
        "extra_description_rows": len(set(descriptions) - vocabulary_set),
        "source_sha256": {
            "names": name_artifact["sha256"],
            "descriptions": description_artifact["sha256"],
        },
    }


def _load_wordnet_entries(archive_path: Path) -> Tuple[Dict[Any, Any], Mapping[Any, Any]]:
    entries: Dict[Any, Any] = {}
    for lookup_pos, filename in (
        ("n", "data.noun"),
        ("v", "data.verb"),
        ("a", "data.adj"),
        ("r", "data.adv"),
    ):
        raw = _read_archive_member(
            archive_path, "tar.gz", "WordNet-3.0/dict/" + filename
        )
        parsed = parse_data_file(raw, lookup_pos=lookup_pos)
        overlap = set(entries) & set(parsed)
        if overlap:
            raise WordNet30Error("Duplicate keys across WordNet data files")
        entries.update(parsed)
    senses = parse_sense_index(
        _read_archive_member(
            archive_path, "tar.gz", "WordNet-3.0/dict/index.sense"
        )
    )
    return entries, senses


def audit_wordnet(
    dataset_plan: Mapping[str, Any],
    vocabulary: Sequence[str],
    *,
    project_root: Path,
    source_lock: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mapping_path, mapping_artifact = locked_artifact_path(
        project_root, source_lock, dataset_plan["mapping_source"]
    )
    archive_path, wordnet_artifact = locked_artifact_path(
        project_root, source_lock, dataset_plan["wordnet_source"]
    )
    rows: Dict[str, WordNetMappingRow] = {}
    pos_counts = Counter()
    for line_number, line in enumerate(
        mapping_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = parse_definition_line(line)
        except ValueError as error:
            raise EvidenceAuditError(
                "Invalid WordNet mapping line {}".format(line_number)
            ) from error
        if row.entity_id in rows:
            raise EvidenceAuditError("Duplicate WN18RR mapping entity")
        rows[row.entity_id] = row
        pos_counts[row.source_pos_tag] += 1
    vocabulary_set = set(vocabulary)
    if set(rows) != vocabulary_set:
        raise EvidenceAuditError("WN18RR mapping IDs do not equal the complete vocabulary")
    entries, senses = _load_wordnet_entries(archive_path)
    records = []
    gloss_alignment_counts = Counter()
    for entity_index, entity_id in enumerate(vocabulary):
        row = rows[entity_id]
        entry = verify_mapping_row(row, entries, senses)
        gloss_alignment = mapping_gloss_alignment(row, entry)
        gloss_alignment_counts[gloss_alignment] += 1
        records.append(
            _record(
                dataset="WN18RR",
                entity_index=entity_index,
                entity_id=entity_id,
                source_artifact="{}|{}".format(
                    mapping_artifact["repository_path"], wordnet_artifact["label"]
                ),
                mapping_status="verified_offset_pos_lemma_sense_{}".format(
                    gloss_alignment
                ),
                evidence_status="full",
                evidence_text=row.lemma_text + ". " + entry.definition,
            )
        )
    return records, {
        "status_counts": {"full": len(records)},
        "mapping_rows": len(rows),
        "extra_mapping_rows": 0,
        "missing_mapping_rows": 0,
        "pos_counts": dict(sorted(pos_counts.items())),
        "compatibility": {
            "resolved_unique_offset_pos": len(records),
            "lemma_matches": len(records),
            "sense_index_matches": len(records),
            "mapping_gloss_alignment": dict(sorted(gloss_alignment_counts.items())),
            "resolved_wordnet_definitions": len(records),
            "try_all_pos_fallbacks": 0,
        },
        "source_sha256": {
            "mapping": mapping_artifact["sha256"],
            "wordnet_3_0": wordnet_artifact["sha256"],
        },
    }


def audit_codex(
    dataset_plan: Mapping[str, Any],
    vocabulary: Sequence[str],
    *,
    project_root: Path,
    source_lock: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata_path, artifact = locked_artifact_path(
        project_root, source_lock, dataset_plan["metadata_source"]
    )
    metadata = read_json_object(metadata_path)
    vocabulary_set = set(vocabulary)
    records = []
    status_counts = Counter()
    for entity_index, entity_id in enumerate(vocabulary):
        value = metadata.get(entity_id)
        if not isinstance(value, dict):
            raise EvidenceAuditError("CoDEx-M entity lacks metadata")
        label_value = value.get("label", "")
        description_value = value.get("description", "")
        if not isinstance(label_value, str) or not isinstance(description_value, str):
            raise EvidenceAuditError("CoDEx metadata fields must be strings")
        label = normalise_text(label_value)
        description = normalise_text(description_value)
        if not _human_readable(label, entity_id):
            raise EvidenceAuditError("CoDEx-M entity lacks a valid label")
        if description:
            evidence_status = "full"
            mapping_status = "verified_label_and_description"
            evidence_text = label + ". " + description
        else:
            evidence_status = "label_only"
            mapping_status = "verified_label_only"
            evidence_text = label
        status_counts[evidence_status] += 1
        records.append(
            _record(
                dataset="CoDEx-M",
                entity_index=entity_index,
                entity_id=entity_id,
                source_artifact=str(artifact["repository_path"]),
                mapping_status=mapping_status,
                evidence_status=evidence_status,
                evidence_text=evidence_text,
            )
        )
    return records, {
        "status_counts": dict(status_counts),
        "metadata_rows": len(metadata),
        "extra_metadata_rows": len(set(metadata) - vocabulary_set),
        "source_sha256": {"metadata": artifact["sha256"]},
    }


def evidence_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise EvidenceAuditError("Evidence record table is empty")
    hashes = Counter(str(record["evidence_sha256"]) for record in records)
    entity_ids_by_hash: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        entity_ids_by_hash[str(record["evidence_sha256"])].append(
            str(record["entity_id"])
        )
    duplicate_groups = [
        {
            "evidence_sha256": evidence_sha256,
            "multiplicity": len(entity_ids),
            "entity_ids": sorted(entity_ids),
        }
        for evidence_sha256, entity_ids in sorted(entity_ids_by_hash.items())
        if len(entity_ids) > 1
    ]
    character_lengths = [int(record["character_length"]) for record in records]
    word_counts = [int(record["pretoken_word_count"]) for record in records]
    return {
        "rows": len(records),
        "unique_evidence_texts": len(hashes),
        "duplicate_text_groups": sum(count > 1 for count in hashes.values()),
        "maximum_text_multiplicity": max(hashes.values()),
        "duplicate_groups": duplicate_groups,
        "character_length": {
            "minimum": min(character_lengths),
            "maximum": max(character_lengths),
            "total": sum(character_lengths),
        },
        "pretoken_word_count": {
            "minimum": min(word_counts),
            "maximum": max(word_counts),
            "total": sum(word_counts),
        },
    }


def validate_expected(
    dataset_plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    details: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> None:
    expected = dataset_plan["expected"]
    observed_status = Counter(record["evidence_status"] for record in records)
    if len(records) != expected["vocabulary"]:
        raise EvidenceAuditError("Evidence row count differs from the frozen expectation")
    for key in ("full", "name_only", "label_only", "missing"):
        if key in expected and observed_status.get(key, 0) != expected[key]:
            raise EvidenceAuditError("Evidence status count differs for {}".format(key))
    for key in (
        "unique_evidence_texts",
        "duplicate_text_groups",
        "maximum_text_multiplicity",
    ):
        if statistics[key] != expected[key]:
            raise EvidenceAuditError("Evidence text multiplicity differs for {}".format(key))
    if "pos_counts" in expected and details.get("pos_counts") != expected["pos_counts"]:
        raise EvidenceAuditError("WordNet POS counts differ from the frozen expectation")
    if "mapping_gloss_alignment" in expected:
        compatibility = details.get("compatibility", {})
        if compatibility.get("mapping_gloss_alignment") != expected["mapping_gloss_alignment"]:
            raise EvidenceAuditError(
                "WordNet mapping-gloss alignment differs from the frozen expectation"
            )


TABLE_COLUMNS = (
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


def table_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = ["\t".join(TABLE_COLUMNS)]
    for record in records:
        values = [str(record[column]) for column in TABLE_COLUMNS]
        if any("\t" in value or "\n" in value or "\r" in value for value in values):
            raise EvidenceAuditError("Evidence audit table contains an unsafe field")
        lines.append("\t".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_evidence_audit(
    source_lock_path: Path,
    plan_path: Path,
    *,
    project_root: Path,
    scratch: Path,
) -> Dict[str, Any]:
    execution = require_scheduled_execution(scratch.resolve())
    source_lock = read_json_object(source_lock_path)
    plan = read_json_object(plan_path)
    if source_lock.get("status") != "source_lock_pending_materialization":
        raise EvidenceAuditError("Source lock status is invalid")
    if plan.get("schema_version") != 1 or plan.get("status") != "pinned_evidence_audit_plan":
        raise EvidenceAuditError("Evidence plan status is invalid")
    source_lock_sha = sha256_file(source_lock_path)
    plan_sha = sha256_file(plan_path)
    git = git_state(project_root)
    prepared = []
    for dataset_plan in plan["datasets"]:
        dataset = str(dataset_plan["name"])
        vocabulary, dataset_manifest = load_vocabulary(
            project_root, dataset, source_lock_sha
        )
        kind = dataset_plan["kind"]
        if kind == "freebase_kgbert":
            records, details = audit_freebase(
                dataset_plan,
                vocabulary,
                project_root=project_root,
                source_lock=source_lock,
            )
        elif kind == "wordnet_3_0":
            records, details = audit_wordnet(
                dataset_plan,
                vocabulary,
                project_root=project_root,
                source_lock=source_lock,
            )
        elif kind == "codex_metadata":
            records, details = audit_codex(
                dataset_plan,
                vocabulary,
                project_root=project_root,
                source_lock=source_lock,
            )
        else:
            raise EvidenceAuditError("Unsupported evidence audit kind")
        statistics = evidence_statistics(records)
        validate_expected(dataset_plan, records, details, statistics)
        table = table_bytes(records)
        manifest = {
            "schema_version": 1,
            "status": "evidence_verified_pending_encoder_tokenization",
            "official_result": False,
            "dataset": dataset,
            "protocol_sha256": source_lock["protocol_sha256"],
            "source_lock_sha256": source_lock_sha,
            "evidence_plan_sha256": plan_sha,
            "dataset_manifest_sha256": sha256_file(
                project_root / "data" / "manifests" / (dataset + "_dataset.json")
            ),
            "dataset_vocabulary": dataset_manifest["audit"]["vocabulary"],
            "details": details,
            "statistics": statistics,
            "audit_table": {
                "filename": "evidence_pre_encoder.tsv",
                "sha256": sha256_bytes(table),
                "rows": len(records),
                "token_length_status": "pending_frozen_encoder_tokenizer",
            },
        }
        prepared.append((dataset, table, manifest))

    job_id = str(execution["scheduler_job_id"])
    dataset_records = []
    for dataset, table, manifest in prepared:
        table_path = (
            project_root
            / "artifacts"
            / "descriptions"
            / dataset
            / "evidence_pre_encoder.tsv"
        )
        manifest_path = (
            project_root
            / "data"
            / "manifests"
            / (dataset + "_evidence_pre_encoder.json")
        )
        write_bytes_new_or_identical(table_path, table, job_id)
        write_bytes_new_or_identical(
            manifest_path, canonical_json_bytes(manifest), job_id
        )
        dataset_records.append(
            {
                "dataset": dataset,
                "rows": manifest["statistics"]["rows"],
                "status_counts": manifest["details"]["status_counts"],
                "unique_evidence_texts": manifest["statistics"]["unique_evidence_texts"],
                "audit_table": str(table_path.relative_to(project_root)),
                "audit_table_sha256": sha256_file(table_path),
                "manifest": str(manifest_path.relative_to(project_root)),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return {
        "schema_version": 1,
        "status": "evidence_verified_pending_encoder_freeze",
        "official_result": False,
        "source_lock_sha256": source_lock_sha,
        "evidence_plan_sha256": plan_sha,
        "execution": execution,
        "git": git,
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
        report = run_evidence_audit(
            args.source_lock.resolve(),
            args.plan.resolve(),
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "evidence_audit_failed",
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
    print("Evidence audit recorded in {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
