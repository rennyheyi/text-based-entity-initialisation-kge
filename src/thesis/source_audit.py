"""Scheduled, read-only audit of candidate dataset and evidence sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
FIGSHARE_FILE_URL_PREFIXES = (
    "https://ndownloader.figshare.com/files/",
    "https://figshare.com/ndownloader/files/",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive_member_name(name: str) -> None:
    normalised = name.replace("\\", "/")
    candidate = PurePosixPath(normalised)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Unsafe archive member path: {!r}".format(name))


def github_raw_url(repository: str, revision: str, path: str) -> str:
    prefix = "https://github.com/"
    if not repository.startswith(prefix) or not repository.endswith(".git"):
        raise ValueError("GitHub repository must use canonical https://github.com/<owner>/<repo>.git form")
    slug = repository[len(prefix) : -4]
    if len(slug.split("/")) != 2:
        raise ValueError("Unexpected GitHub repository slug")
    if not FULL_SHA_RE.fullmatch(revision):
        raise ValueError("GitHub raw URL requires a full 40-character commit SHA")
    path_obj = PurePosixPath(path)
    if path_obj.is_absolute() or ".." in path_obj.parts:
        raise ValueError("Unsafe repository path")
    return "https://raw.githubusercontent.com/{}/{}/{}".format(slug, revision, path_obj.as_posix())


def parse_ls_remote(output: str, expected_ref: str) -> str:
    rows = []
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) == 2 and fields[1] == expected_ref:
            rows.append(fields[0])
    if len(rows) != 1 or not FULL_SHA_RE.fullmatch(rows[0]):
        raise ValueError("Expected exactly one full SHA for {}".format(expected_ref))
    return rows[0]


def resolve_repository_ref(repository: str, ref: str) -> str:
    if FULL_SHA_RE.fullmatch(ref):
        return ref
    expected_ref = ref if ref.startswith("refs/") else "refs/heads/{}".format(ref)
    completed = subprocess.run(
        ["git", "ls-remote", "--exit-code", repository, expected_ref],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return parse_ls_remote(completed.stdout, expected_ref)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported source-candidate schema version")
    if config.get("status") != "candidate_sources_only_not_pinned":
        raise ValueError("Candidate audit must not consume a configuration marked as official or pinned")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Configuration must contain a non-empty sources list")
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every source entry must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
            raise ValueError("Invalid source ID")
        if source_id in seen:
            raise ValueError("Duplicate source ID: {}".format(source_id))
        seen.add(source_id)
        if source.get("kind") not in {"http_file", "github_files", "figshare_file"}:
            raise ValueError("Unsupported source kind for {}".format(source_id))


def require_scheduled_execution(scratch: Path) -> Dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Gate A source audit is permitted on the cluster only inside a Slurm allocation")
    if not scratch.is_absolute() or not scratch.exists() or not scratch.is_dir():
        raise RuntimeError("A pre-created absolute job-scratch directory is required")
    if not os.access(str(scratch), os.W_OK):
        raise RuntimeError("Job-scratch directory is not writable")
    return {
        "execution_context": "slurm",
        "scheduler_job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "scratch_path": str(scratch),
        "started_at_utc": utc_now(),
    }


def download_file(url: str, destination: Path, max_bytes: int) -> Dict[str, Any]:
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS source URLs are permitted")
    request = urllib.request.Request(url, headers={"User-Agent": "thesis-source-audit/1.0"})
    digest = hashlib.sha256()
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("xb") as output:
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None and int(declared_length) > max_bytes:
            raise ValueError("Declared download size exceeds configured maximum")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("Download exceeds configured maximum")
            output.write(chunk)
            digest.update(chunk)
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type")
    if total == 0:
        raise ValueError("Downloaded artifact is empty")
    return {
        "requested_url": url,
        "resolved_url": final_url,
        "content_type": content_type,
        "size_bytes": total,
        "sha256": digest.hexdigest(),
    }


def inspect_archive(path: Path, archive_format: str, max_unpacked_bytes: int) -> Dict[str, Any]:
    members: List[Dict[str, Any]] = []
    total_unpacked = 0
    if archive_format == "zip":
        with zipfile.ZipFile(path, "r") as archive:
            for info in archive.infolist():
                validate_archive_member_name(info.filename)
                total_unpacked += info.file_size
                members.append(
                    {
                        "name": info.filename,
                        "size_bytes": info.file_size,
                        "crc32": "{:08x}".format(info.CRC),
                        "is_directory": info.is_dir(),
                    }
                )
    elif archive_format == "tar.gz":
        with tarfile.open(path, "r:gz") as archive:
            for info in archive.getmembers():
                validate_archive_member_name(info.name)
                total_unpacked += info.size
                members.append(
                    {
                        "name": info.name,
                        "size_bytes": info.size,
                        "type": repr(info.type),
                        "is_file": info.isfile(),
                    }
                )
    else:
        raise ValueError("Unsupported archive format: {}".format(archive_format))
    if total_unpacked > max_unpacked_bytes:
        raise ValueError("Archive exceeds configured maximum unpacked size")
    return {
        "archive_format": archive_format,
        "member_count": len(members),
        "total_unpacked_bytes": total_unpacked,
        "members": members,
    }


def audit_download(
    *,
    source_id: str,
    label: str,
    url: str,
    scratch: Path,
    max_download_bytes: int,
    archive_format: Optional[str] = None,
    max_unpacked_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    destination = scratch / source_id / safe_label
    record = download_file(url, destination, max_download_bytes)
    record["label"] = label
    if archive_format is not None:
        if max_unpacked_bytes is None:
            raise ValueError("Archive source requires max_unpacked_bytes")
        record["archive"] = inspect_archive(destination, archive_format, max_unpacked_bytes)
    return record


def figshare_version_api_url(article_id: int, article_version: int) -> str:
    if article_id <= 0 or article_version <= 0:
        raise ValueError("Figshare article ID and version must be positive integers")
    return "https://api.figshare.com/v2/articles/{}/versions/{}".format(
        article_id, article_version
    )


def select_figshare_file(
    metadata: Mapping[str, Any],
    *,
    article_id: int,
    expected_title: str,
    filename: str,
    max_download_bytes: int,
) -> Mapping[str, Any]:
    if metadata.get("id") != article_id:
        raise ValueError("Figshare metadata article ID does not match the requested article")
    if metadata.get("title") != expected_title:
        raise ValueError("Figshare metadata title does not match the expected title")
    files = metadata.get("files")
    if not isinstance(files, list):
        raise ValueError("Figshare metadata does not contain a file list")
    matches = [item for item in files if isinstance(item, dict) and item.get("name") == filename]
    if len(matches) != 1:
        raise ValueError("Expected exactly one Figshare file named {!r}".format(filename))
    selected = matches[0]
    file_id = selected.get("id")
    size = selected.get("size")
    download_url = selected.get("download_url")
    if not isinstance(file_id, int) or file_id <= 0:
        raise ValueError("Figshare file has an invalid file ID")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Figshare file has an invalid reported size")
    if size > max_download_bytes:
        raise ValueError("Figshare file exceeds the configured maximum download size")
    if not isinstance(download_url, str) or not download_url.startswith(
        FIGSHARE_FILE_URL_PREFIXES
    ):
        raise ValueError("Figshare file has an unexpected download URL")
    return selected


def audit_figshare_file(source: Mapping[str, Any], scratch: Path) -> Dict[str, Any]:
    source_id = str(source["id"])
    article_id = int(source["article_id"])
    article_version = int(source["article_version"])
    metadata_url = figshare_version_api_url(article_id, article_version)
    metadata_path = scratch / source_id / "article_metadata_v{}.json".format(article_version)
    metadata_record = download_file(
        metadata_url,
        metadata_path,
        int(source["max_metadata_bytes"]),
    )
    metadata_record["label"] = metadata_path.name
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Figshare article metadata is not valid UTF-8 JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("Figshare article metadata must be a JSON object")
    selected = select_figshare_file(
        metadata,
        article_id=article_id,
        expected_title=str(source["expected_title"]),
        filename=str(source["filename"]),
        max_download_bytes=int(source["max_download_bytes"]),
    )
    artifact = audit_download(
        source_id=source_id,
        label=str(source["filename"]),
        url=str(selected["download_url"]),
        scratch=scratch,
        max_download_bytes=int(source["max_download_bytes"]),
        archive_format=source.get("archive_format"),
        max_unpacked_bytes=source.get("max_unpacked_bytes"),
    )
    if artifact["size_bytes"] != selected["size"]:
        raise ValueError("Downloaded Figshare file size differs from article metadata")
    artifact["figshare_file_id"] = selected["id"]
    artifact["figshare_reported_size_bytes"] = selected["size"]
    artifact["figshare_supplied_md5"] = selected.get("supplied_md5")
    artifact["figshare_computed_md5"] = selected.get("computed_md5")
    return {
        "id": source_id,
        "dataset": source.get("dataset"),
        "role": source.get("role"),
        "kind": source.get("kind"),
        "status": "audited_candidate_requires_pin",
        "source_page": source.get("source_page"),
        "article_id": article_id,
        "article_version": article_version,
        "article_title": metadata.get("title"),
        "article_doi": metadata.get("doi"),
        "metadata_artifact": metadata_record,
        "artifacts": [artifact],
    }


def audit_source(source: Mapping[str, Any], scratch: Path) -> Dict[str, Any]:
    source_id = str(source["id"])
    result: Dict[str, Any] = {
        "id": source_id,
        "dataset": source.get("dataset"),
        "role": source.get("role"),
        "kind": source.get("kind"),
        "status": "in_progress",
        "artifacts": [],
    }
    if source["kind"] == "figshare_file":
        return audit_figshare_file(source, scratch)
    if source["kind"] == "http_file":
        result["source_page"] = source.get("source_page")
        result["declared_version"] = source.get("declared_version")
        result["artifacts"].append(
            audit_download(
                source_id=source_id,
                label=str(source["filename"]),
                url=str(source["url"]),
                scratch=scratch,
                max_download_bytes=int(source["max_download_bytes"]),
                archive_format=source.get("archive_format"),
                max_unpacked_bytes=source.get("max_unpacked_bytes"),
            )
        )
    elif source["kind"] == "github_files":
        repository = str(source["repository"])
        requested_ref = str(source["ref"])
        revision = resolve_repository_ref(repository, requested_ref)
        result["repository"] = repository
        result["requested_ref"] = requested_ref
        result["resolved_revision"] = revision
        for index, file_spec in enumerate(source["files"]):
            path = str(file_spec["path"])
            result["artifacts"].append(
                audit_download(
                    source_id=source_id,
                    label="{:02d}_{}".format(index, path),
                    url=github_raw_url(repository, revision, path),
                    scratch=scratch,
                    max_download_bytes=int(file_spec["max_download_bytes"]),
                    archive_format=file_spec.get("archive_format"),
                    max_unpacked_bytes=file_spec.get("max_unpacked_bytes"),
                )
            )
            result["artifacts"][-1]["repository_path"] = path
    else:  # guarded by validate_config(), retained for direct unit calls
        raise ValueError("Unsupported source kind for {}".format(source_id))
    result["status"] = "audited_candidate_requires_pin"
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    execution = require_scheduled_execution(args.scratch.resolve())
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)

    report: MutableMapping[str, Any] = {
        "schema_version": 1,
        "status": "candidate_source_audit_running",
        "official_result": False,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "execution": execution,
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git": subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip(),
        },
        "sources": [],
    }

    failures = 0
    for source in config["sources"]:
        try:
            report["sources"].append(audit_source(source, args.scratch.resolve()))
        except Exception as error:  # preserved in the audit report before failing the job
            failures += 1
            report["sources"].append(
                {
                    "id": source.get("id"),
                    "dataset": source.get("dataset"),
                    "role": source.get("role"),
                    "kind": source.get("kind"),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    report["execution"]["completed_at_utc"] = utc_now()
    if failures:
        report["status"] = "candidate_source_audit_failed"
        report["failure_count"] = failures
    else:
        report["status"] = "candidate_sources_audited_requires_pin"
        report["failure_count"] = 0
    atomic_write_json(output_path, report)
    print("Gate A candidate-source audit recorded in {}".format(output_path))
    if failures:
        print("Gate A candidate-source audit failed for {} source(s)".format(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
