"""Build an immutable source lock from a successful Gate A candidate audit."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from .source_audit import FULL_SHA_RE, sha256_file


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SourceLockError(RuntimeError):
    """Raised when an audit report cannot be promoted to a source lock."""


def read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceLockError("Cannot read JSON object from {}".format(path)) from error
    if not isinstance(value, dict):
        raise SourceLockError("Expected a JSON object in {}".format(path))
    return value


def _required_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SourceLockError("{} is not a full SHA-256 digest".format(label))
    return value


def _required_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SourceLockError("{} must be a positive integer".format(label))
    return value


def _lock_archive(archive: Any) -> Optional[Dict[str, Any]]:
    if archive is None:
        return None
    if not isinstance(archive, dict):
        raise SourceLockError("Archive audit must be an object")
    archive_format = archive.get("archive_format")
    if archive_format not in {"zip", "tar.gz"}:
        raise SourceLockError("Unsupported audited archive format")
    member_count = _required_positive_int(archive.get("member_count"), "archive member count")
    unpacked = _required_positive_int(
        archive.get("total_unpacked_bytes"), "archive unpacked byte count"
    )
    return {
        "archive_format": archive_format,
        "member_count": member_count,
        "total_unpacked_bytes": unpacked,
    }


def _lock_artifact(artifact: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(artifact, dict):
        raise SourceLockError("{} audit must be an object".format(label))
    requested_url = artifact.get("requested_url")
    resolved_url = artifact.get("resolved_url")
    if not isinstance(requested_url, str) or not requested_url.startswith("https://"):
        raise SourceLockError("{} requested URL is invalid".format(label))
    if not isinstance(resolved_url, str) or not resolved_url.startswith("https://"):
        raise SourceLockError("{} resolved URL is invalid".format(label))
    locked: Dict[str, Any] = {
        "label": artifact.get("label"),
        "requested_url": requested_url,
        "content_type": artifact.get("content_type"),
        "size_bytes": _required_positive_int(artifact.get("size_bytes"), label + " size"),
        "sha256": _required_sha256(artifact.get("sha256"), label + " SHA-256"),
    }
    for key in (
        "repository_path",
        "figshare_file_id",
        "figshare_reported_size_bytes",
        "figshare_supplied_md5",
        "figshare_computed_md5",
    ):
        if key in artifact:
            locked[key] = artifact[key]
    archive = _lock_archive(artifact.get("archive"))
    if archive is not None:
        locked["archive"] = archive
    return locked


def _require_exact_artifacts(
    configured: Sequence[Mapping[str, Any]],
    audited: Any,
    *,
    source_id: str,
) -> List[Dict[str, Any]]:
    if not isinstance(audited, list) or len(audited) != len(configured):
        raise SourceLockError("{} artifact count differs from candidate config".format(source_id))
    locked = []
    for index, (file_spec, artifact) in enumerate(zip(configured, audited)):
        if not isinstance(artifact, dict):
            raise SourceLockError("{} artifact {} is invalid".format(source_id, index))
        expected_path = file_spec.get("path")
        if expected_path is not None and artifact.get("repository_path") != expected_path:
            raise SourceLockError("{} repository artifact order/path mismatch".format(source_id))
        locked.append(_lock_artifact(artifact, label="{} artifact {}".format(source_id, index)))
    return locked


def _lock_source(configured: Mapping[str, Any], audited: Mapping[str, Any]) -> Dict[str, Any]:
    source_id = configured.get("id")
    if audited.get("id") != source_id:
        raise SourceLockError("Source IDs differ between config and audit")
    if audited.get("status") != "audited_candidate_requires_pin":
        raise SourceLockError("Source {} did not pass candidate audit".format(source_id))
    for key in ("dataset", "role", "kind"):
        if audited.get(key) != configured.get(key):
            raise SourceLockError("Source {} has mismatched {}".format(source_id, key))

    kind = configured.get("kind")
    locked: MutableMapping[str, Any] = {
        "id": source_id,
        "dataset": configured.get("dataset"),
        "role": configured.get("role"),
        "kind": kind,
    }
    if kind == "http_file":
        artifacts = _require_exact_artifacts([configured], audited.get("artifacts"), source_id=str(source_id))
        artifact = artifacts[0]
        if artifact["requested_url"] != configured.get("url"):
            raise SourceLockError("Source {} URL differs from candidate config".format(source_id))
        if artifact["label"] != configured.get("filename"):
            raise SourceLockError("Source {} filename differs from candidate config".format(source_id))
        locked.update(
            {
                "source_page": configured.get("source_page"),
                "declared_version": configured.get("declared_version"),
                "artifacts": artifacts,
            }
        )
    elif kind == "github_files":
        repository = configured.get("repository")
        if audited.get("repository") != repository:
            raise SourceLockError("Source {} repository differs from candidate config".format(source_id))
        revision = audited.get("resolved_revision")
        if not isinstance(revision, str) or not FULL_SHA_RE.fullmatch(revision):
            raise SourceLockError("Source {} lacks a full resolved Git revision".format(source_id))
        requested_ref = configured.get("ref")
        if audited.get("requested_ref") != requested_ref:
            raise SourceLockError("Source {} requested ref differs from candidate config".format(source_id))
        if isinstance(requested_ref, str) and FULL_SHA_RE.fullmatch(requested_ref) and revision != requested_ref:
            raise SourceLockError("Source {} changed an already pinned revision".format(source_id))
        locked.update(
            {
                "repository": repository,
                "requested_ref": requested_ref,
                "resolved_revision": revision,
                "artifacts": _require_exact_artifacts(
                    configured.get("files", []),
                    audited.get("artifacts"),
                    source_id=str(source_id),
                ),
            }
        )
    elif kind == "figshare_file":
        for key in ("article_id", "article_version"):
            if audited.get(key) != configured.get(key):
                raise SourceLockError("Source {} has mismatched {}".format(source_id, key))
        if audited.get("article_title") != configured.get("expected_title"):
            raise SourceLockError("Source {} has an unexpected Figshare title".format(source_id))
        artifacts = _require_exact_artifacts([configured], audited.get("artifacts"), source_id=str(source_id))
        if artifacts[0]["label"] != configured.get("filename"):
            raise SourceLockError("Source {} has an unexpected Figshare filename".format(source_id))
        metadata_artifact = _lock_artifact(
            audited.get("metadata_artifact"), label=str(source_id) + " metadata"
        )
        locked.update(
            {
                "source_page": configured.get("source_page"),
                "article_id": configured.get("article_id"),
                "article_version": configured.get("article_version"),
                "article_title": audited.get("article_title"),
                "article_doi": audited.get("article_doi"),
                "metadata_artifact": metadata_artifact,
                "artifacts": artifacts,
            }
        )
    else:
        raise SourceLockError("Unsupported source kind {}".format(kind))
    return dict(locked)


def build_source_lock(
    candidate_config: Mapping[str, Any],
    audit_report: Mapping[str, Any],
    *,
    candidate_config_sha256: str,
    audit_report_sha256: str,
    protocol_sha256: str,
) -> Dict[str, Any]:
    if candidate_config.get("status") != "candidate_sources_only_not_pinned":
        raise SourceLockError("Candidate configuration has an invalid status")
    if audit_report.get("status") != "candidate_sources_audited_requires_pin":
        raise SourceLockError("Audit report did not pass")
    if audit_report.get("official_result") is not False or audit_report.get("failure_count") != 0:
        raise SourceLockError("Audit report is not a clean candidate audit")
    if audit_report.get("config_sha256") != candidate_config_sha256:
        raise SourceLockError("Audit report was produced from a different candidate config")
    configured_sources = candidate_config.get("sources")
    audited_sources = audit_report.get("sources")
    if not isinstance(configured_sources, list) or not isinstance(audited_sources, list):
        raise SourceLockError("Sources must be lists")
    if len(configured_sources) != len(audited_sources):
        raise SourceLockError("Candidate and audited source counts differ")
    completed_at = audit_report.get("execution", {}).get("completed_at_utc")
    if not isinstance(completed_at, str) or not completed_at:
        raise SourceLockError("Audit report lacks a completion timestamp")
    return {
        "schema_version": 1,
        "status": "source_lock_pending_materialization",
        "official_result": False,
        "protocol_sha256": _required_sha256(protocol_sha256, "protocol SHA-256"),
        "candidate_config_sha256": _required_sha256(
            candidate_config_sha256, "candidate config SHA-256"
        ),
        "candidate_audit_report_sha256": _required_sha256(
            audit_report_sha256, "candidate audit report SHA-256"
        ),
        "candidate_audit_job_id": audit_report.get("execution", {}).get("scheduler_job_id"),
        "retrieved_at_utc": completed_at,
        "sources": [
            _lock_source(configured, audited)
            for configured, audited in zip(configured_sources, audited_sources)
        ],
    }


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> None:
    value = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != value:
            raise SourceLockError("Refusing to overwrite a different source lock")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
    os.replace(str(temporary), str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_path = args.candidate_config.resolve()
    report_path = args.audit_report.resolve()
    protocol_path = args.protocol.resolve()
    lock = build_source_lock(
        read_json_object(candidate_path),
        read_json_object(report_path),
        candidate_config_sha256=sha256_file(candidate_path),
        audit_report_sha256=sha256_file(report_path),
        protocol_sha256=sha256_file(protocol_path),
    )
    write_new_or_identical(args.output.resolve(), lock)
    print("Source lock written to {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
